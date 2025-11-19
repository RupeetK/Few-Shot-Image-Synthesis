
# get_ipython().system('pip install albumentations')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable
import random

class DiffAugment:
    def __init__(self,policy='color,translation,cutout', probability=0.8):
        # Parse augmentation policy and store probability
        self.policy=policy.split(',')
        self.probability = probability

    def __call__(self,x):
        # x should be a tensor
        # Skip augmentation randomly
        if torch.rand(1).item() > self.probability: 
            return x

        # Apply all enabled augmentation functions
        for p in self.policy:
            for f in AUGMENT_FNS[p]:
                x=f(x)
        return x


# per-channel augmentations

def rand_brightness(x):
    # Add random brightness shift per image
    magnitude = torch.rand(x.size(0),1,1,1,dtype=x.dtype,device=x.device) - 0.5
    x =x + magnitude
    return x

def rand_saturation(x):
    # Scale distance from mean to adjust saturation
    magnitude = torch.rand(x.size(0),1,1,1,dtype=x.dtype,device=x.device) * 2
    x_mean = x.mean(dim=1,keepdim=True)
    x = (x-x_mean) * magnitude + x_mean
    return x

def rand_contrast(x):
    # Adjust global contrast
    magnitude = torch.rand(x.size(0),1,1,1,dtype=x.dtype,device=x.device) + 0.5
    x_mean = x.mean(dim=[1,2,3],keepdim=True)
    x = (x-x_mean) * magnitude + x_mean
    return x

def rand_translation(x,ratio=0.125):
    # Randomly shift image spatially by fraction of size
    batch_size, _, height, width = x.shape
    shift_x = int(width * ratio + 0.5)
    shift_y = int(height * ratio + 0.5)

    # Random offsets for each image
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[batch_size,1,1],device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[batch_size,1,1],device=x.device)

    # Create coordinate grid
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(batch_size, dtype=torch.long, device=x.device),
        torch.arange(height, dtype=torch.long, device=x.device),
        torch.arange(width, dtype=torch.long, device=x.device),
        indexing = 'ij'
    )

    # Translate grid and clamp to ensure valid indices
    grid_x = torch.clamp(grid_x + translation_y, 0, height - 1)  
    grid_y = torch.clamp(grid_y + translation_x, 0, width - 1)
    
    x_pad = torch.nn.functional.pad(x, [1,1,1,1], mode='constant', value=0)
    
    # Add 1 to grid indices for padded tensor (since padding adds 1 pixel border)
    x = x_pad.permute(0,2,3,1).contiguous()[grid_batch, grid_x + 1, grid_y + 1].permute(0,3,1,2).contiguous()
    
    return x


def rand_cutout(x, ratio=0.5):
    # Remove a random rectangular region
    batch_size, _, height, width = x.shape

    # Compute cutout size
    cutout_h = max(1, int(height * ratio + 0.5))
    cutout_w = max(1, int(width * ratio + 0.5))

    # Random center of cutout (fully inside image)
    offset_y = torch.randint(
        cutout_h // 2, height - (cutout_h - (cutout_h // 2)) + 1, 
        size=(batch_size,1,1),
        device=x.device,
        dtype=torch.long
    )
    
    offset_x = torch.randint(
        cutout_w // 2, width - (cutout_w - (cutout_w // 2)) + 1, 
        size=(batch_size,1,1),
        device=x.device,
        dtype=torch.long
    )

    # Create grid for the cutout patch
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(batch_size, dtype=torch.long, device=x.device),    
        torch.arange(cutout_h, dtype=torch.long, device=x.device),       
        torch.arange(cutout_w, dtype=torch.long, device=x.device), 
        indexing = 'ij'
    )

    # Shift grid to offset 
    grid_x = grid_x + offset_y - cutout_h // 2
    grid_y = grid_y + offset_x - cutout_w // 2


    # Create mask and apply cutout
    mask = torch.ones(batch_size, height, width, dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)

    return x

AUGMENT_FNS= {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}


class AlbumentationsAugment:

    def __init__(self,img_size=1024, augment_prob=0.8):
        self.augment_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.4
                ),
                A.RandomCrop(
                    height = img_size//2,
                    width = img_size//2,
                    p=0.3
                ),
                # A.CoarseDropout(
                #     max_holes = 1,
                #     max_height = img_size//2,
                #     max_width = img_size//2,
                #     min_holes = 1,
                #     fill_value = 0,
                #     p=0.5
                # ),
                A.ColorJitter(
                    brightness = 0.1,
                    contrast = 0.1,
                    saturation = 0.15,
                    hue = 0.05,
                    p=0.3
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit = (3,5), p = 1.0),  # Mild blur
                        A.GaussNoise(var_limit = (10.0,30.0), p = 1.0),  # Low noise
                    ],p = 0.2
                ),
                A.Affine(
                    scale = (0.9,1.1),  # Mild scaling 
                    rotate = (-15,15),  # small rotation
                    p = 0.3
                ),
                A.ShiftScaleRotate(
                    shift_limit = 0.0625,
                    scale_limit = 0.1, 
                    rotate_limit = 10,
                    border_mode = 0,
                    p = 0.3
                ),    
            ], p = augment_prob
        )

        # Base transform (always applied - resize and normalize)
        self.base_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])

    def __call__(self,image):
        # Convert PIL Image to numpy array 
        if isinstance(image,Image.Image):
            image = np.array(image)

        # Apply transformations then normalization
        augmented = self.augment_transform(image=image)['image']
        final = self.base_transform(image=augmented)['image']

        return final
        


def Augment(img_size, training=True):
    # Standard lightweight training pipeline
    # Eval: no Horizontal Flip
    if training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


class FoodDataset(Dataset):
    def __init__(self,root_dir,img_size,augmentation_type=None,training=True):
        
        self.root_dir= Path(root_dir)
        self.img_size = img_size
        self.augmentation_type = augmentation_type
        self.training = training
       
        # Gather images from pizza/sushi/pasta folders
        self.images_paths = []
        self.labels=[]
        self.class_to_label = {}
        for label, class_name in enumerate (['pizza','sushi','pasta']):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                self.class_to_label[class_name] = label
                images = list(class_dir.glob('*.jpg'))

                self.images_paths.extend(images)
                self.labels.extend([label] * len(images))

        # Shuffle dataset
        random.seed(42)
        combined = list(zip(self.images_paths,self.labels))
        random.shuffle(combined)
        self.images_paths,self.labels = zip(*combined) if combined else ([],[])
       
        # Reverse mapping for display
        self.label_to_class = {v:k for k,v in self.class_to_label.items()}

        # Choose augmentation
        self._set_transform()

        print(f'{len(self.images_paths)} images, classes: {self.class_to_label}')


    def _set_transform(self):
        # Either Albumentations or Torchvision
        if self.augmentation_type == 'albumentations':
            augment_prob = 0.8 if self.training else 0.0
            self.transform = AlbumentationsAugment(
                img_size = self.img_size, 
                augment_prob=augment_prob
            )

        else:
            self.transform = Augment(img_size = self.img_size,training=self.training)

            

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self,idx):
        # Load image + label
        img_path = self.images_paths[idx]
        label = self.labels[idx]

        img=Image.open(img_path).convert('RGB')

        img = self.transform(img)

        return {
            'image' : img,
            'label' : label,
            'class_name' : self.label_to_class[label],
            'path' : str(img_path)
        }


def create_dataloaders(root_dir,class_name=None,img_size,batch_size,augmentation_type=None,num_workers=0):

    '''
    For GAN Training, augmentation_type=None
    For Diffusion, augmentation_type=albumentations
    '''

    # load only a single food category folder
    if class_name is not None:
        class_dir = os.path.join(root_dir,class_name)
        
        if not os.path.isdir(class_dir):
            raise ValueError(f"Class folder not found: {class_dir}")

        train_dataset = FoodDataset(
            class_dir,img_size=img_size,augmentation_type=augmentation_type,training=True
        )
    
        # No augmentation for evaluation
        eval_dataset = FoodDataset(
            class_dir,img_size=img_size,augmentation_type=None,training=False
        )

    # multi-class 
    else:
        train_dataset = FoodDataset(
            root_dir,img_size=img_size,augmentation_type=augmentation_type,training=True
        )
    
        # No augmentation for evaluation
        eval_dataset = FoodDataset(
            root_dir,img_size=img_size,augmentation_type=None,training=False
        )
    
    # drop_last = True
    train_loader = DataLoader(
        train_dataset,batch_size,shuffle=True,num_workers=num_workers,pin_memory=False,drop_last=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,batch_size,shuffle=False,num_workers=num_workers,pin_memory=False,drop_last=False
    )

    return train_loader, eval_loader
        


