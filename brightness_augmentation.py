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
    
    
def add_arrays(img_arr, mask_arr):
    
    return np.clip(img_arr + mask_arr, 0, 255).astype('uint8')
   
    
def test_noise():
    
    img = np.load('D:/steep_training/ski-race/training_data-3.npy')[89][0]
    
    full_img = cv2.imdecode(img, 1)
    cv2.imshow('Original', full_img)
    
    np.save('D:/steep_training/original.npy', full_img)
    
    uniform_img = add_uniform_lighting(full_img)
    noise_img = add_noise_lighting(full_img)
    
    cv2.imshow('Uniform', uniform_img)
    cv2.imshow('Noise', noise_img)
    
    np.save('D:/steep_training/uniform.npy', uniform_img)
    np.save('D:/steep_training/noise.npy', noise_img)
    
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        
#test_noise()