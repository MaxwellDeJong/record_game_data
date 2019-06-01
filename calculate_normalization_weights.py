# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:57:50 2019

@author: Max
"""

import numpy as np
import cv2
import os


def parse_file(filename):
    
    training_data = np.load(filename)
    
    img_arr = [i[0] for i in training_data]
    
    img_decomp_arr = [cv2.imdecode(img, 1) for img in img_arr]
    
    means = np.mean(img_decomp_arr, axis=(0, 1, 2))
    stds = np.std(img_decomp_arr, axis=(0, 1, 2))
    
    return (len(img_arr), means, stds)
    

def calc_weight_dict():
    
    weight_dict = {}
    idx = 0
    
    while True:
        filename = 'D:/steep_training/ski-race/balanced/training_data-{}.npy'.format(idx)
        
        if (os.path.isfile(filename)):
            (length, means, stds) = parse_file(filename)
            
            weight_dict[idx] = (length, means, stds)
            idx += 1
            
            print('Finished analyzing file ', idx)
            
        else:
            return weight_dict
        
        
def count_n_frames(weight_dict):
    
    n_frames = 0
    
    for key in weight_dict:
        n_frames += weight_dict[key][0]
        
    return n_frames
        
        
def calc_overall_statistics(weight_dict, n_frames):
    
    means = np.zeros(3)
    variances = np.zeros(3)
    
    for key in weight_dict:
        
        (local_size, local_means, local_stds) = weight_dict[key]
        
        weight = local_size / n_frames
        
        means += weight * local_means
        
        # Not precisely correct with different population sizes and means
        variances += local_stds * local_stds
        
    return (means / 255, np.sqrt(variances) / 255)
    
    
def calculate_normalization_coefficients():
    
    weight_dict = calc_weight_dict()
    n_frames = count_n_frames(weight_dict)
    
    (means, stds) = calc_overall_statistics(weight_dict, n_frames)
    
    stats_filename = 'D:/steep_training/ski-race/balanced/normalization_weights.npy'
    
    np.save(stats_filename, [means, stds])
    
calculate_normalization_coefficients()