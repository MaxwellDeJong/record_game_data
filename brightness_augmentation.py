# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 21:18:41 2019

@author: Max
"""
import numpy as np
import cv2

def add_uniform_lighting(img):
    
    val = np.random.randint(-40, 40)
    
    mask = val * np.ones(img.shape)
    
    return add_arrays(img, mask)


def add_noise_lighting(img):
    
    mean = np.random.randint(-40, 40)
    std = np.random.randint(5, 40)
    
    noise = np.random.normal(mean, std, img.shape)
    
    return add_arrays(img, noise)
    
    
def add_arrays(original_img_arr, mask_arr):
    
    img_arr = np.copy(original_img_arr)
    
    (max_x, max_y, n_channel) = np.shape(img_arr)
    
    for i in range(max_x):
        for j in range(max_y):
            for k in range(n_channel):
                img_arr[i, j, k] = int(max(min(img_arr[i, j, k] + mask_arr[i, j, k], 254), 0))
            
    return img_arr
   
    
def test_noise():
    
    img = np.load('D:/steep_training/ski-race/training_data-3--aug.npy')[89][0]
    
    full_img = cv2.imdecode(img, 1)
    cv2.imshow('Original', full_img)
    
    uniform_img = add_uniform_lighting(full_img)
    noise_img = add_noise_lighting(full_img)
    
    cv2.imshow('Uniform', uniform_img)
    cv2.imshow('Noise', noise_img)
    
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        cv2.destroyAllWindows()