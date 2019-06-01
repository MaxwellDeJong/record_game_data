# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:57:50 2019

@author: Max
"""

import numpy as np
import cv2
import os


def save_individual_file(count, img_arr, one_hot_arr):

    for i in range(len(img_arr)):
            
        filename = 'D:/steep_training/ski-race/balanced/training_frame-{}.npy'.format(count + i)
        np.save(filename, [img_arr[i], one_hot_arr[i]])


def parse_file(filename, global_count):
    
    training_data = np.load(filename)
    
    img_arr = [i[0] for i in training_data]
    one_hot_arr = [i[1] for i in training_data]
    
    save_individual_file(global_count, img_arr, one_hot_arr)
    
    img_decomp_arr = [cv2.imdecode(img, 1) for img in img_arr]
    
    means = np.mean(img_decomp_arr, axis=(0, 1, 2))
    stds = np.std(img_decomp_arr, axis=(0, 1, 2))
    
    return (len(img_arr), means, stds)
    

def calc_weight_dict():
    
    weight_dict = {}
    idx = 0
    global_count = 0
    
    delete_bulk = False
    
    while True:
        filename = 'D:/steep_training/ski-race/balanced/training_data-{}.npy'.format(idx)
        
        if (os.path.isfile(filename)):
            (length, means, stds) = parse_file(filename, global_count)
            
            weight_dict[idx] = (length, means, stds)
            idx += 1
            global_count += length
            
            print('Finished analyzing file ', idx)
            
            if delete_bulk:
                os.remove(filename)
            
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


def delete_bulk_balanced_data():
    
    
def calculate_normalization_coefficients():
    
    weight_dict = calc_weight_dict()
    n_frames = count_n_frames(weight_dict)
    
    (means, stds) = calc_overall_statistics(weight_dict, n_frames)
    
    stats_filename = 'D:/steep_training/ski-race/balanced/normalization_weights.npy'
    
    np.save(stats_filename, [means, stds])
    
calculate_normalization_coefficients()