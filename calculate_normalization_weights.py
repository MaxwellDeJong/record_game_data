# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:57:50 2019

@author: Max
"""

import numpy as np
import cv2
import os
import pickle
import random

def count_training_sets():
    
    idx = 0
 
    while True:
        filename = 'D:/steep_training/ski-race/training_data-{}.npy'.format(idx)
        
        if (os.path.isfile(filename)):
            idx += 1
        else:
            return idx


def save_individual_file(count, img_arr, one_hot_arr, label_dict):

    for i in range(len(img_arr)):
            
        filename = 'D:/steep_training/ski-race/balanced/training_frame-{}.npy'.format(count + i)
        label_dict[count + i] = one_hot_arr[i]
        np.save(filename, img_arr[i])


def parse_file(filename, n_files_avg, n_files, global_count, label_dict):
    
    training_data = np.load(filename)
    
    img_arr = [i[0] for i in training_data]
    one_hot_arr = [i[1] for i in training_data]
    
    save_individual_file(global_count, img_arr, one_hot_arr, label_dict)
    
    n_files_per_set = n_files_avg / n_files
    
    samples_to_avg = random.sample(range(2000), n_files_per_set)
    
    minibatch_img_arr = [img_arr[i] for i in samples_to_avg]
    img_decomp_arr = [cv2.imdecode(img, 1) for img in minibatch_img_arr]
    
    means = np.mean(img_decomp_arr, axis=(0, 1, 2))
    stds = np.std(img_decomp_arr, axis=(0, 1, 2))
    
    return (len(img_arr), means, stds)
    

def calc_weight_dict(n_files_avg):
    
    weight_dict = {}
    label_dict = {}
    idx = 0
    global_count = 0
    
    n_training_sets = count_training_sets()
    
    delete_bulk = False
    
    while True:
        filename = 'D:/steep_training/ski-race/balanced/training_data-{}.npy'.format(idx)
        
        if (os.path.isfile(filename)):
            (length, means, stds) = parse_file(filename, n_files_avg, n_training_sets, global_count, label_dict)
            
            weight_dict[idx] = (length, means, stds)
            idx += 1
            global_count += length
            
            print('Finished analyzing file ', idx)
            
            if delete_bulk:
                os.remove(filename)
            
        else:
            return (weight_dict, label_dict)
        
        
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
    
    (weight_dict, label_dict) = calc_weight_dict(20000)
    n_frames = count_n_frames(weight_dict)
    
    (means, stds) = calc_overall_statistics(weight_dict, n_frames)
    
    stats_filename = 'D:/steep_training/ski-race/balanced/normalization_weights.npy'   
    np.save(stats_filename, [means, stds])
    
    label_filename = 'D:/steep_training/ski-race/balanced/label_dict.pkl'
    
    with open(label_filename, 'wb') as handle:
        pickle.dump(label_dict, handle)
    
calculate_normalization_coefficients()