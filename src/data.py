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
        if max_padding < 0:
            raise ValueError("max_padding must be non-negative")
        self.max_padding = max_padding
        
    def __call__(self, sample):
        origin, mask = sample
        if self.max_padding == 0:
            return origin, mask

        padding = np.random.randint(0, self.max_padding + 1)
        origin = torchvision.transforms.functional.pad(origin, padding=padding, fill=0)
        mask = torchvision.transforms.functional.pad(mask, padding=padding, fill=0)
        return origin, mask


class Crop():
    def __init__(self, max_shift):
        if max_shift < 0:
            raise ValueError("max_shift must be non-negative")
        self.max_shift = max_shift
        
    def __call__(self, sample):
        origin, mask = sample
        if self.max_shift == 0:
            return origin, mask

        top_shift = np.random.randint(0, self.max_shift + 1)
        bottom_shift = np.random.randint(0, self.max_shift + 1)
        left_shift = np.random.randint(0, self.max_shift + 1)
        right_shift = np.random.randint(0, self.max_shift + 1)
        origin_w, origin_h = origin.size
        crop_w = origin_w - left_shift - right_shift
        crop_h = origin_h - top_shift - bottom_shift
        if crop_w <= 0 or crop_h <= 0:
            return origin, mask

        origin = torchvision.transforms.functional.crop(
            origin,
            top_shift,
            left_shift,
            crop_h,
            crop_w,
        )
        mask = torchvision.transforms.functional.crop(
            mask,
            top_shift,
            left_shift,
            crop_h,
            crop_w,
        )
        return origin, mask


class Resize():
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample):
        origin, mask = sample
        origin = torchvision.transforms.functional.resize(
            origin,
            self.output_size,
            interpolation=Image.BILINEAR,
        )
        mask = torchvision.transforms.functional.resize(
            mask,
            self.output_size,
            interpolation=Image.NEAREST,
        )
        
        return origin, mask


class ComposePair():
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomHorizontalFlipPair():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        origin, mask = sample
        if np.random.random() < self.p:
            origin = torchvision.transforms.functional.hflip(origin)
            mask = torchvision.transforms.functional.hflip(mask)
        return origin, mask


class RandomVerticalFlipPair():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        origin, mask = sample
        if np.random.random() < self.p:
            origin = torchvision.transforms.functional.vflip(origin)
            mask = torchvision.transforms.functional.vflip(mask)
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
