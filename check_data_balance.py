# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:37:19 2019

@author: Max
"""
import numpy as np
import os

from balance_training_data import load_dicts, count_desired_frames

def calc_expected_size(count_dict):
    
    n_sig_keys = len(count_dict)
    desired_frames = count_desired_frames(count_dict)
    
    return (desired_frames, n_sig_keys * desired_frames)


def get_key(one_hot, one_hot_dict):
    
    for key in one_hot_dict:
        if (one_hot_dict[key] == one_hot):
            return key
        
    print('No key found!')


def count_balanced_keys(one_hot_dict):
    
    idx = 0
    n_balanced_files = 0
    
    balanced_count_dict = {}
    
    while True:
        filename = 'D:/steep_training/ski-race/balanced/training_data-{}.npy'.format(idx)
        
        if (os.path.isfile(filename)):
            
            training_data = np.load(filename)
            one_hots = [i[1] for i in training_data]
            
            for one_hot in one_hots:
                key = get_key(one_hot, one_hot_dict)
                
                if key in balanced_count_dict:
                    balanced_count_dict[key] += 1
                else:
                    balanced_count_dict[key] = 1
                    
                n_balanced_files += 1
                    
            idx += 1
            
        else:
            return (n_balanced_files, balanced_count_dict)
   
    
def diagnose_balance():

    (count_dict, new_one_hot_dict, _) = load_dicts()
    
    (desired_frames, expected_size) = calc_expected_size(count_dict)
    (actual_size, balanced_count_dict) = count_balanced_keys(new_one_hot_dict)
    
    print('Original counts: ', count_dict)
    print('Expecting ', desired_frames, ' for each key, for a total of ', expected_size, ' entries.')
    
    print('Balanced counts: ', balanced_count_dict)
    print('Total of ', actual_size, ' entries in balanced set.')
    
diagnose_balance()