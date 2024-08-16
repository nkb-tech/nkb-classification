import cv2
import torchvision.transforms as T
from PIL import Image
import os
import torch
import numpy as np

from configs.singletask_on_yolo_crops_config import mean, std


# Since your image is normalized, you need to unnormalize it. You have to do the reverse operations that you did during normalization. One way is
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def save_augs(dataset, image_dir='exp'):
    transform = T.ToPILImage() 
    num_samples = 50
    if len(dataset) < num_samples: num_samples = len(dataset)
    
    for i in range(num_samples):
        tensor_img,_ = dataset[i]
        # unorm = UnNormalize(torch.mean(tensor_img, dim=(1, 2)), torch.std(tensor_img, dim=(1, 2)))
        unorm = UnNormalize(mean=mean, std=std)
        image = transform(unorm(tensor_img))
        image_filename = os.path.join(image_dir, f'{i}.jpg')
        image.save(image_filename)
        image = cv2.imread(image_filename)
        Cimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_filename, Cimg)
        

