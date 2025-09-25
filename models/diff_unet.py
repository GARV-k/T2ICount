from omegaconf import OmegaConf

import torch as th
import torch
import math
import abc

from torch import nn, einsum

from einops import rearrange, repeat
from transformers import CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextConfig, CLIPTextModel, CLIPTextTransformer
from inspect import isfunction


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        def forward(x, context=None, mask=None):
            h = self.heads

            q = self.to_q(x)
            is_cross = context is not None
            context = default(context, x)

            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            attn2 = rearrange(attn, '(b h) k c -> h b k c', h=h).mean(0)
            controller(attn2, is_cross, place_in_unet)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()

    for net in sub_nets:
        if "input_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "output_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "middle_block" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        attn = self.forward(attn, is_cross, place_in_unet)
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= (self.max_size) ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item for item in self.step_store[key]] for key in self.step_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, base_size=64, max_size=None):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.base_size = base_size
        if max_size is None:
            self.max_size = self.base_size // 2
        else:
            self.max_size = max_size


def register_hier_output(model, device1, device2):
    self = model
    from ldm.modules.diffusionmodules.util import checkpoint, timestep_embedding
    def forward(x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        h = x.type(self.dtype).to(device1)
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(h.device)
        emb = self.time_embed(t_emb)
        emb = emb.to(h.device)
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y.to(emb.device))
        # Support both the original "context" argument and the ddpm-style
        # keyword arg `c_crossattn` which is commonly passed as a list of
        # conditioning tensors. If `c_crossattn` is provided, concatenate
        # along the token dimension to build the final context tensor. This
        # mirrors the behaviour in ldm.models.diffusion.ddpm.DiffusionWrapper
        # and avoids falling back to self-attention (which can cause
        # query/context dimension mismatches when context is expected to be
        # text features of dimension 768).
        if 'c_crossattn' in kwargs and kwargs['c_crossattn'] is not None:
            cc = kwargs['c_crossattn']
            # cc may be a list like [tensor] or already a tensor
            if isinstance(cc, (list, tuple)):
                context = torch.cat(cc, 1).to(h.device)
            else:
                context = cc.to(h.device)
        else:
            if context is not None:
                context = context.to(h.device)
        # Input blocks
        for i, module in enumerate(self.input_blocks):
            target_device = device1 if i < len(self.input_blocks) // 2 else device2
            h = h.to(target_device)
            emb = emb.to(target_device)
            context = context.to(target_device) if context is not None else None
            h = module(h, emb, context)
            hs.append(h)
        # Middle block
        h = h.to(device2)
        emb = emb.to(device2)
        context = context.to(device2) if context is not None else None
        h = self.middle_block(h, emb, context)
        out_list = []
        # Output blocks
        for i_out, module in enumerate(self.output_blocks):
            target_device = device1 if i_out < len(self.output_blocks) // 2 else device2
            h = h.to(target_device)
            emb = emb.to(target_device)
            context = context.to(target_device) if context is not None else None
            h_cat = hs.pop().to(target_device)
            h = torch.cat([h, h_cat], dim=1)
            h = module(h, emb, context)
            if i_out in [1, 4, 7]:
                out_list.append(h)
        h = h.type(x.dtype)

        out_list.append(h)
        return out_list
    self.forward = forward


class UNetWrapper(nn.Module):
    def __init__(self, unet, use_attn=True, base_size=512, max_attn_size=None,
                 attn_selector='up_cross+down_cross', devices=None) -> None:
        super().__init__()
        self.unet = unet
        self.attention_store = AttentionStore(base_size=base_size // 8, max_size=max_attn_size)
        self.size12 = base_size // 32
        self.size24 = base_size // 16
        self.size48 = base_size // 8
        self.use_attn = use_attn
        # By default, partition across 4 GPUs for this user's request.
        # devices can be a list of device strings (e.g. ['cuda:0','cuda:1','cuda:2','cuda:3']).
        if devices is None:
            devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
        # convert to torch.device objects
        self.devices = [torch.device(d) for d in devices]

        # Balanced partitioning strategy:
        # - spread input_blocks across first half of devices, middle_block to the middle device,
        #   and output_blocks across second half, aiming for even distribution.
        n_devices = len(self.devices)
        n_input = len(unet.input_blocks)
        n_output = len(unet.output_blocks)

        # map input blocks round-robin across devices 0..n_devices-1
        for i, block in enumerate(unet.input_blocks):
            target = self.devices[i % n_devices]
            block.to(target)

        # place middle_block onto device index floor(n_devices/2)
        mid_idx = n_devices // 2
        unet.middle_block.to(self.devices[mid_idx])

        # map output blocks round-robin across devices starting from mid_idx (to interleave compute)
        for i, block in enumerate(unet.output_blocks):
            target = self.devices[(mid_idx + i) % n_devices]
            block.to(target)
        # Ensure time embedding and other small top-level modules live on device1 so
        # operations like self.time_embed(t_emb) don't mix CPU params with GPU tensors.
        # Move small top-level modules (time_embed/label_emb/conv_in) to a stable device.
        # Choose device 0 so early processing uses the earliest GPU and to balance memory.
        primary_top_device = self.devices[0]
        if hasattr(unet, 'time_embed'):
            try:
                unet.time_embed.to(primary_top_device)
            except Exception:
                pass
        if hasattr(unet, 'label_emb'):
            try:
                unet.label_emb.to(primary_top_device)
            except Exception:
                pass
        # conv_in / input conv may be used before blocks — place on primary_top_device
        if hasattr(unet, 'conv_in'):
            try:
                unet.conv_in.to(primary_top_device)
            except Exception:
                pass
        if self.use_attn:
            register_attention_control(unet, self.attention_store)
        # Register a forward on the unet that uses our devices mapping so the
        # runtime tensor movement matches how we placed modules above.
        self.device1 = self.devices[0]
        self.device2 = self.devices[1] if len(self.devices) > 1 else self.devices[0]

        from ldm.modules.diffusionmodules.util import checkpoint, timestep_embedding

        def forward_unet(x, timesteps=None, context=None, y=None, **kwargs):
            # x: input latents
            assert (y is not None) == (
                    getattr(unet, 'num_classes', None) is not None
            ), "must specify y if and only if the model is class-conditional"
            hs = []
            # start on device 0 for top-level ops
            h = x.type(unet.dtype).to(self.devices[0])
            t_emb = timestep_embedding(timesteps, unet.model_channels, repeat_only=False)
            t_emb = t_emb.to(h.device)
            emb = unet.time_embed(t_emb)
            emb = emb.to(h.device)
            if getattr(unet, 'num_classes', None) is not None:
                assert y.shape == (x.shape[0],)
                emb = emb + unet.label_emb(y.to(emb.device))

            # Support ddpm-style c_crossattn keyword
            if 'c_crossattn' in kwargs and kwargs['c_crossattn'] is not None:
                cc = kwargs['c_crossattn']
                if isinstance(cc, (list, tuple)):
                    context = torch.cat(cc, 1).to(h.device)
                else:
                    context = cc.to(h.device)
            else:
                if context is not None:
                    context = context.to(h.device)

            # Input blocks (round-robin placement)
            for i, module in enumerate(unet.input_blocks):
                target_device = self.devices[i % len(self.devices)]
                h = h.to(target_device)
                emb = emb.to(target_device)
                context = context.to(target_device) if context is not None else None
                h = module(h, emb, context)
                hs.append(h)

            # Middle block
            mid_idx = len(self.devices) // 2
            h = h.to(self.devices[mid_idx])
            emb = emb.to(self.devices[mid_idx])
            context = context.to(self.devices[mid_idx]) if context is not None else None
            h = unet.middle_block(h, emb, context)
            out_list = []

            # Output blocks (round-robin starting from mid_idx)
            for i_out, module in enumerate(unet.output_blocks):
                target_device = self.devices[(mid_idx + i_out) % len(self.devices)]
                h = h.to(target_device)
                emb = emb.to(target_device)
                context = context.to(target_device) if context is not None else None
                h_cat = hs.pop().to(target_device)
                h = torch.cat([h, h_cat], dim=1)
                h = module(h, emb, context)
                if i_out in [1, 4, 7]:
                    out_list.append(h)

            h = h.type(x.dtype)
            out_list.append(h)
            return out_list

        # Install our forward onto the unet instance
        unet.forward = forward_unet
        self.attn_selector = attn_selector.split('+')

    def forward(self, *args, **kwargs):
        if self.use_attn:
            self.attention_store.reset()
        out_list = self.unet(*args, **kwargs)
        if self.use_attn:
            avg_attn = self.attention_store.get_average_attention()
            attn12, attn24, attn48 = self.process_attn(avg_attn)
            return out_list[::-1], attn12, attn24, attn48
        else:
            return out_list[::-1]

    def process_attn(self, avg_attn):
        attns = {self.size12: [], self.size24: [], self.size48: []}
        for k in self.attn_selector:
            for up_attn in avg_attn[k]:
                # Ensure attention tensors are on a single device before stacking.
                try:
                    up_attn = up_attn.to(self.device1)
                except Exception:
                    # fallback: move to CPU if GPU move fails
                    up_attn = up_attn.cpu()
                size = int(math.sqrt(up_attn.shape[1]))
                attns[size].append(rearrange(up_attn, 'b (h w) c -> b c h w', h=size))
        attn12 = torch.stack(attns[self.size12]).mean(0)
        attn24 = torch.stack(attns[self.size24]).mean(0)
        if len(attns[self.size48]) > 0:
            attn48 = torch.stack(attns[self.size48]).mean(0)
        else:
            attn48 = None
        return attn12, attn24, attn48
