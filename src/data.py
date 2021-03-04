import torch
import torchvision

import numpy as np
from pathlib import Path

from PIL import Image


class LungDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        origin_mask_list,
        origins_folder,
        masks_folder,
        transforms=None,
        origin_mode="L",
        mask_threshold=128,
    ):
        self.origin_mask_list = list(origin_mask_list)
        self.origins_folder = Path(origins_folder)
        self.masks_folder = Path(masks_folder)
        self.transforms = transforms
        self.origin_mode = origin_mode
        self.mask_threshold = mask_threshold

        if not 0 <= self.mask_threshold <= 255:
            raise ValueError("mask_threshold must be in range [0, 255]")

        if not self.origins_folder.exists():
            raise FileNotFoundError(f"Origins folder not found: {self.origins_folder}")
        if not self.masks_folder.exists():
            raise FileNotFoundError(f"Masks folder not found: {self.masks_folder}")

    def _sample_paths(self, origin_name, mask_name):
        origin_path = self.origins_folder / f"{origin_name}.png"
        mask_path = self.masks_folder / f"{mask_name}.png"

        if not origin_path.exists():
            raise FileNotFoundError(f"Origin image not found: {origin_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask image not found: {mask_path}")

        return origin_path, mask_path
    
    def __getitem__(self, idx):
        origin_name, mask_name = self.origin_mask_list[idx]
        origin_path, mask_path = self._sample_paths(origin_name, mask_name)
        origin = Image.open(origin_path).convert(self.origin_mode)
        mask = Image.open(mask_path).convert("L")
        if self.transforms is not None:
            origin, mask = self.transforms((origin, mask))
            
        origin = torchvision.transforms.functional.to_tensor(origin) - 0.5
    
        mask = np.array(mask)
        mask = (torch.tensor(mask) > self.mask_threshold).long() 
        return origin, mask
        
    
    def __len__(self):
        return len(self.origin_mask_list)

    
class Pad():
    def __init__(self, max_padding):
        self.max_padding = max_padding
        
    def __call__(self, sample):
        origin, mask = sample
        padding = np.random.randint(0, self.max_padding)
#         origin = torchvision.transforms.functional.pad(origin, padding=padding, padding_mode="symmetric")
        origin = torchvision.transforms.functional.pad(origin, padding=padding, fill=0)
        mask = torchvision.transforms.functional.pad(mask, padding=padding, fill=0)
        return origin, mask


class Crop():
    def __init__(self, max_shift):
        self.max_shift = max_shift
        
    def __call__(self, sample):
        origin, mask = sample
        tl_shift = np.random.randint(0, self.max_shift)
        br_shift = np.random.randint(0, self.max_shift)
        origin_w, origin_h = origin.size
        crop_w = origin_w - tl_shift - br_shift
        crop_h = origin_h - tl_shift - br_shift
        
        origin = torchvision.transforms.functional.crop(origin, tl_shift, tl_shift,
                                                        crop_h, crop_w)
        mask = torchvision.transforms.functional.crop(mask, tl_shift, tl_shift,
                                                        crop_h, crop_w)
        return origin, mask


class Resize():
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample):
        origin, mask = sample
        origin = torchvision.transforms.functional.resize(origin, self.output_size)
        mask = torchvision.transforms.functional.resize(mask, self.output_size)
        
        return origin, mask


def blend(origin, mask1=None, mask2=None):
    img = torchvision.transforms.functional.to_pil_image(origin + 0.5).convert("RGB")
    if mask1 is not None:
        mask1 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.zeros_like(origin),
            torch.stack([mask1.float()]),
            torch.zeros_like(origin)
        ]))
        img = Image.blend(img, mask1, 0.2)
        
    if mask2 is not None:
        mask2 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.stack([mask2.float()]),
            torch.zeros_like(origin),
            torch.zeros_like(origin)
        ]))
        img = Image.blend(img, mask2, 0.2)
    
    return img
