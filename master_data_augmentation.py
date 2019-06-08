# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:30:45 2019

@author: Max
"""

from brightness_augmentation import add_uniform_lighting, add_noise_lighting
from elastic_transformation_augmentation import elastic_transform
from sunspot_augmentation import generate_sunspot
import cv2


def all_augmentation(img, n_augmentations):
    
    full_img = cv2.imdecode(img, 1)
    
    aug_arr = []
    
    for i in range(n_augmentations):
        if (i % 4 == 0):
            new_img = generate_sunspot(full_img)
        elif (i % 4 == 1):
            new_img = elastic_transform(full_img)
        elif (i % 4 == 2):
            new_img = add_uniform_lighting(full_img)
        elif (i % 4 == 3):
            new_img = add_noise_lighting(full_img)
            
        (res, comp_img) = cv2.imencode('.jpg', new_img)
            
        aug_arr.append(comp_img)
            
    return aug_arr