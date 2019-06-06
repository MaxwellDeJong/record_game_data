# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:30:45 2019

@author: Max
"""

from brightness_augmentation import add_uniform_lighting, add_noise_lighting
from elastic_transformation_augmentation import elastic_transform
from sunspot_augmentation import generate_sunspot


def all_augmentation(img, n_augmentations):
    
    aug_arr = []
    
    for i in n_augmentations:
        if (i % 4 == 0):
            aug_arr.append(generate_sunspot(img))
        elif (i % 4 == 1):
            aug_arr.append(elastic_transform(img))
        elif (i % 4 == 2):
            aug_arr.append(add_uniform_lighting(img))
        elif (i % 4 == 3):
            aug_arr.append(add_noise_lighting(img))
            
    return aug_arr