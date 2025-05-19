# Defines custom transforms

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import cv2
import random
from typing import Any

def convert_contrast(image_16bit, kernel='linear', gamma=1.0, convert_to='uint8', convert_to_bgr=False):
    if image_16bit.dtype != np.uint16:
        raise ValueError("Input image must be 16-bit numpy array")
    lower = np.min(image_16bit)
    upper = np.max(image_16bit)

    normalized = (image_16bit - lower) / (upper - lower)
    
    if kernel == 'linear':
        enhanced = normalized
    elif kernel == 'log':
        enhanced = np.log1p(normalized * 1000) / np.log1p(1000)
    elif kernel == 'gamma':
        enhanced = np.power(normalized, gamma)
    else:
        raise ValueError("Unknown kernel")
    
    enhanced = np.clip(enhanced, 0, 1)
    if convert_to == 'uint8':
        result = (enhanced * 255).astype(np.uint8)
    elif convert_to == 'uint16':
        result = (enhanced * 65535).astype(np.uint16)
    else:
        raise ValueError("Unknown output type")
    
    if convert_to_bgr:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

class ContrastTransform(ImageOnlyTransform):

    def __init__(self, kernel='linear', gamma=1.0, convert_to='uint8', convert_to_bgr=True):
        super().__init__(p=1.0)
        self.kernel = kernel
        self.gamma = gamma
        self.convert_to = convert_to
        self.convert_to_bgr = convert_to_bgr

    def apply(self, img, **params):
        return convert_contrast(img, self.kernel, self.gamma, self.convert_to, self.convert_to_bgr)

