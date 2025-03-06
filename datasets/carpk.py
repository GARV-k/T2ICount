import torch.nn as nn
import glob
from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
from transformers import CLIPTokenizer
import torch


class CARPK(Dataset):
    def __init__(self, root, info):
        self.im_list = sorted(glob.glob(os.path.join(root, 'Images/*.png')))
                
        self.root = root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.label_list = []
        try:
            with open(info) as f:
                for i in f:
                    self.label_list.append(i.strip())
        except:
            raise Exception("The path to label list is incorrect.")
        
        split_list = []
        for i in self.im_list:
            if os.path.basename(i).replace('.png', '') in self.label_list:
                split_list.append(i)
        self.im_list = split_list
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):

        im_path = self.im_list[item]

        img = Image.open(im_path).convert('RGB')
        ann_path = im_path.replace('png', 'txt').replace('Images', 'Annotations')
        print(os.path.basename(im_path))

        with open(ann_path, 'r') as f:
            count = sum(1 for line in f)

        prompt = 'cars'

        attention_mask = torch.zeros(77)
        cls_name_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        cls_name_length = cls_name_tokens['input_ids'].shape[1]
        attention_mask[1: 1 + cls_name_length] = 1

        return self.transform(img), count, prompt, attention_mask, os.path.basename(im_path).split('.')[0]
