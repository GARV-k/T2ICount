from torch import nn
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch
from models.diff_unet import UNetWrapper
import torch.nn.functional as F
from models.decoder import Upsample, Regressor


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


class Count(nn.Module):
    def __init__(self, config, sd_path, unet_config=dict()):
        super(Count, self).__init__()
        config = OmegaConf.load(f'{config}')
        sd_model = load_model_from_config(config, f"{sd_path}")
        self.vae = sd_model.first_stage_model
        # sd_model may be wrapped (e.g., DiffusionWrapper) or may contain the diffusion model at
        # sd_model.model.diffusion_model. Choose the actual UNet module robustly.
        unet_module = None
        # Case 1: checkpoint returned a wrapper that stores the diffusion model as .diffusion_model
        if hasattr(sd_model, 'diffusion_model'):
            unet_module = sd_model.diffusion_model
        # Case 2: sd_model.model may itself be a wrapper containing .diffusion_model
        elif hasattr(sd_model, 'model') and hasattr(sd_model.model, 'diffusion_model'):
            unet_module = sd_model.model.diffusion_model
        # Case 3: fallback to sd_model.model (original code path)
        elif hasattr(sd_model, 'model'):
            unet_module = sd_model.model

        if unet_module is None:
            raise RuntimeError('Could not locate UNet module inside loaded sd_model (checked diffusion_model and model)')

        # For this user's machine, hardcode 4-GPU partitioning across cuda:0..3
        devices = unet_config.get('devices') if isinstance(unet_config, dict) and 'devices' in unet_config else ['cuda:0','cuda:1','cuda:2','cuda:3']
        self.unet = UNetWrapper(unet_module, **unet_config, devices=devices) if isinstance(unet_config, dict) else UNetWrapper(unet_module, devices=devices)

        # Move VAE and CLIP to GPU:0 (the primary device) so their parameters live on the same device
        # as input tensors when encoding. UNetWrapper handles distributing UNet submodules across
        # multiple GPUs (device1/device2). Place the Decoder onto UNet's device1 so it can accept
        # the feature maps returned by UNet without device mismatch.
        self.clip = sd_model.cond_stage_model
        # delete VAE decoder if present to save memory (original behavior)
        try:
            del self.vae.decoder
        except Exception:
            pass

        # Move VAE and CLIP to primary GPU (cuda:0) when available
        if torch.cuda.is_available():
            device0 = torch.device('cuda:0')
            try:
                self.vae.to(device0)
            except Exception:
                pass
            try:
                self.clip.to(device0)
            except Exception:
                pass

        self.set_frozen_parameters()

        # Place decoder onto a device where some of the UNet output blocks live.
        # We'll choose the device that the last output block was placed on to receive the outputs.
        try:
            last_output_device = self.unet.devices[-1]
        except Exception:
            last_output_device = self.unet.device1
        self.decoder = Decoder(in_dim=[320, 640, 1280, 1280], out_dim=256).to(last_output_device)

    def set_frozen_parameters(self):
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.clip.parameters():
            param.requires_grad = False

    def set_train(self):
        self.unet.train()
        self.decoder.train()
        self.vae.eval()
        self.clip.eval()

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.clip.eval()
        self.decoder.eval()

    def extract_feat(self, img, prompt):
        with torch.no_grad():
            latents = self.vae.encode(img)
            class_info = self.clip.encode(prompt)
            c_crossattn = class_info

        latents = latents.mode().detach()

        t = torch.zeros((img.shape[0],), device=img.device).long()
        outs = self.unet(latents, t, c_crossattn=[c_crossattn])
        return outs, c_crossattn

    def forward(self, x, prompt, prompt_attn_mask):
        outs, text_feat = self.extract_feat(x, prompt)
        img_feat, attn12, attn24, attn48 = outs
        # Ensure attention maps, masks and text features live on the same device
        # to avoid device mismatch during element-wise ops.
        attn_device = attn12.device if attn12 is not None else self.unet.device1
        try:
            prompt_attn_mask = prompt_attn_mask.to(attn_device)
        except Exception:
            prompt_attn_mask = prompt_attn_mask.cuda(attn_device.index) if torch.cuda.is_available() else prompt_attn_mask
        try:
            text_feat = text_feat.to(attn_device)
        except Exception:
            pass
        attn12 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)(attn12)
        attn24 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(attn24)
        attn12 = (attn12 * prompt_attn_mask).sum(dim=1, keepdims=True) / prompt_attn_mask.sum(dim=1, keepdims=True)
        attn24 = (attn24 * prompt_attn_mask).sum(dim=1, keepdims=True) / prompt_attn_mask.sum(dim=1, keepdims=True)
        attn48 = (attn48 * prompt_attn_mask).sum(dim=1, keepdims=True) / prompt_attn_mask.sum(dim=1, keepdims=True)

        fused_attn = 0.1 * minmax_norm(attn48) + 0.3 * minmax_norm(attn24) + 0.6 * minmax_norm(attn12)
        text_feat = (text_feat * prompt_attn_mask.squeeze(3)).sum(dim=1) / prompt_attn_mask.squeeze(3).sum(dim=1) # B, 768
        # Ensure decoder inputs live on the same device as the decoder module.
        try:
            decoder_params = next(self.decoder.parameters())
            decoder_device = decoder_params.device
        except StopIteration:
            decoder_device = self.unet.device1

        # Move image feature maps to decoder device
        img_feat = [f.to(decoder_device) for f in img_feat]
        try:
            text_feat = text_feat.to(decoder_device)
        except Exception:
            pass

        den_map, sim_x2, sim_x1 = self.decoder(img_feat, text_feat)
        return den_map, sim_x2, sim_x1, fused_attn.float().detach()

    def forward_with_encoded_text(self, x, text_feat, prompt_attn_mask):
        """Forward method that takes pre-encoded text features (for DataParallel)"""
        with torch.no_grad():
            latents = self.vae.encode(x)
        latents = latents.mode().detach()
        
        t = torch.zeros((x.shape[0],), device=x.device).long()
        outs = self.unet(latents, t, c_crossattn=[text_feat])
        img_feat, attn12, attn24, attn48 = outs
        # Move masks and text features to attention tensors' device to avoid
        # cross-device operations when combining attention maps with masks.
        attn_device = attn12.device if attn12 is not None else self.unet.device1
        try:
            prompt_attn_mask = prompt_attn_mask.to(attn_device)
        except Exception:
            prompt_attn_mask = prompt_attn_mask.cuda(attn_device.index) if torch.cuda.is_available() else prompt_attn_mask
        try:
            text_feat = text_feat.to(attn_device)
        except Exception:
            pass

        attn12 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)(attn12)
        attn24 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(attn24)
        attn12 = (attn12 * prompt_attn_mask).sum(dim=1, keepdims=True) / prompt_attn_mask.sum(dim=1, keepdims=True)
        attn24 = (attn24 * prompt_attn_mask).sum(dim=1, keepdims=True) / prompt_attn_mask.sum(dim=1, keepdims=True)
        attn48 = (attn48 * prompt_attn_mask).sum(dim=1, keepdims=True) / prompt_attn_mask.sum(dim=1, keepdims=True)

        fused_attn = 0.1 * minmax_norm(attn48) + 0.3 * minmax_norm(attn24) + 0.6 * minmax_norm(attn12)
        text_feat = (text_feat * prompt_attn_mask.squeeze(3)).sum(dim=1) / prompt_attn_mask.squeeze(3).sum(dim=1) # B, 768
        try:
            decoder_params = next(self.decoder.parameters())
            decoder_device = decoder_params.device
        except StopIteration:
            decoder_device = self.unet.device1

        img_feat = [f.to(decoder_device) for f in img_feat]
        try:
            text_feat = text_feat.to(decoder_device)
        except Exception:
            pass

        den_map, sim_x2, sim_x1 = self.decoder(img_feat, text_feat)
        return den_map, sim_x2, sim_x1, fused_attn.float().detach()


class Decoder(nn.Module):
    def __init__(self, in_dim=[320, 717, 1357, 1280], out_dim=256):
        super(Decoder, self).__init__()
        self.upsample1 = Upsample(in_dim[3], in_dim[3]//2, in_dim[3]//2 + in_dim[2], in_dim[3]//2)
        self.upsample2 = Upsample(in_dim[3]//2, in_dim[3] // 4, in_dim[3]//4 + in_dim[1], in_dim[3] // 4)
        self.upsample3 = Upsample(in_dim[3]//4, out_dim, out_dim + in_dim[0], out_dim, False)

        self.regressor = Regressor(out_dim)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, outs, text_feat):
        x = outs
        feat, fused_text = self.upsample1(x[3], x[2], text_feat)
        sim_x2 = F.cosine_similarity(self.up(self.up(feat)), fused_text.unsqueeze(2).unsqueeze(3), dim=1).unsqueeze(1)
        feat, fused_text = self.upsample2(feat, x[1], text_feat,
                                          simi_map=F.interpolate(sim_x2, (24, 24), mode='bilinear', align_corners=False))
        sim_x1 = F.cosine_similarity(self.up(feat), fused_text.unsqueeze(2).unsqueeze(3), dim=1).unsqueeze(1)
        feat = self.upsample3(feat, x[0], simi_map=sim_x1)
        den = self.regressor(feat)
        return den, sim_x2, sim_x1


def minmax_norm(x):
    B, C, H, W = x.shape
    x = x.flatten(1)
    x_min = x.min(dim=1, keepdim=True)[0]
    x_max = x.max(dim=1, keepdim=True)[0]
    eps = 1e-7
    denominator = (x_max - x_min) + eps
    x_normalized = (x - x_min) / denominator
    x_normalized = x_normalized.reshape(B, C, H, W)
    return x_normalized

