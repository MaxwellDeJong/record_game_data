# -*- coding: utf-8 -*-
"""
Created on Thu May 30 01:03:30 2019

@author: Max
"""
import pickle
import numpy as np
import os
from random import shuffle

def deep_dict_copy(dict_to_copy):
    
    new_dict = {}
    
    for key in dict_to_copy:
        new_dict[key] = dict_to_copy[key][:]
        
    return new_dict


def load_one_hot_dict():
        
    with open('D:/steep_training/ski-race/one_hot_dict.pkl', 'rb') as handle:
        one_hot_dict = pickle.load(handle)
        
    return one_hot_dict
    
    
def initialize_count_dict(one_hot_dict):
    
    count_dict = {}
        
    for key in one_hot_dict:
        count_dict[key] = 0
        
    return count_dict


def get_key(one_hot_vec, one_hot_dict):
    
    for (key, one_hot) in one_hot_dict.items():
        if (one_hot == one_hot_vec):
            return key
        
    print('Could not find matching vector for ', one_hot_vec)
    print('One hot dictionary: ', one_hot_dict)


def count_file(filename, one_hot_dict, count_dict):
    
    training_data = np.load(filename)
    print(filename, ' had ', len(training_data), ' entries')
    
    one_hot_arr = [i[1] for i in training_data]
    
    for one_hot in one_hot_arr:
        key = get_key(one_hot, one_hot_dict)
        count_dict[key] += 1
        
        
def count_files(idx, one_hot_dict, count_dict, aug_count_dict):
    
    filename = 'D:/steep_training/ski-race/training_data-{}.npy'.format(idx)
    aug_filename = 'D:/steep_training/ski-race/training_data-{}--aug.npy'.format(idx)
    
    count_file(filename, one_hot_dict, count_dict)
    count_file(aug_filename, one_hot_dict, aug_count_dict)
    
    next_filename = 'D:/steep_training/ski-race/training_data-{}.npy'.format(idx+1)
    
    valid_next_file = os.path.isfile(next_filename)
    
    return valid_next_file
    
    
def count_training_files():
    
    one_hot_dict = load_one_hot_dict()
    
    count_dict = initialize_count_dict(one_hot_dict)
    aug_count_dict = initialize_count_dict(one_hot_dict)
    
    n_files = 0
    
    idx = 0   
    filename = 'D:/steep_training/ski-race/training_data-{}.npy'.format(idx)
                                                         
    valid_file = os.path.isfile(filename)
    
    while valid_file:
        
        print('Counting keystrokes for file ', idx)
        
        valid_file = count_files(idx, one_hot_dict, count_dict, aug_count_dict)
        n_files += 1
        idx += 1
        
#    for key in count_dict:
#        print('For key ', key, ' we count ', count_dict[key], 
#              ' in training data\nand ', aug_count_dict[key], 
#              ' in augmented file')
        
    return (n_files, one_hot_dict, count_dict, aug_count_dict)


def find_keys_to_remove(cutoff, count_dict):
    
    tot_count = 0
    
    for key in count_dict:
        tot_count += count_dict[key]
        
    keys_to_remove = []
    
    for key in count_dict:
        if (count_dict[key] < cutoff * tot_count):
            keys_to_remove.append(key)
            
    return keys_to_remove


def rebase_one_hot_dict(cutoff, count_dict, original_one_hot_dict):
    '''If a key combination is present in less than 'cutoff' files, remove
    it from our possible labels.'''
              
    new_one_hot_dict = deep_dict_copy(original_one_hot_dict)

    keys_to_remove = find_keys_to_remove(cutoff, count_dict)
    nonzero_elem_list = []
    
    for key in keys_to_remove:
        one_hot = original_one_hot_dict[key]
        nonzero_elem_list.append(one_hot.index(1))
        
        del new_one_hot_dict[key]
        
    elems_to_delete = sorted(nonzero_elem_list)[::-1]
    
    for key in new_one_hot_dict:
        for elem in elems_to_delete:
            del new_one_hot_dict[key][elem]
            
    return new_one_hot_dict


def count_raw_percentages(new_one_hot_dict, count_dict):
    '''Counts the occurences of the significant keypresses for creating weights
    for use after neural network is trained.'''
    
    raw_pct_dict = {}
    
    tot_count = 0
    for key in new_one_hot_dict:
        tot_count += count_dict[key]
        
    for key in new_one_hot_dict:
        raw_pct_dict[key] = count_dict[key] / tot_count
        
    return raw_pct_dict


def count_frames_to_delete(new_one_hot_dict, count_dict, aug_count_dict):

    min_n_frames = int(1e8)
    
    tot_count_dict = {}
    
    for key in new_one_hot_dict:
        tot_count = count_dict[key] + aug_count_dict[key]
        
        tot_count_dict[key] = tot_count
        
        if (tot_count < min_n_frames):
            min_n_frames = tot_count
            
    frames_to_del_dict = {}
    
    for key in new_one_hot_dict:
        frames_to_del_dict[key] = tot_count_dict[key] - min_n_frames
        
    print('Count dict: ', tot_count_dict)
    print('Minimum number of frames among significant keys: ', min_n_frames)
    print('Frames to delete: ', frames_to_del_dict)
        
    return frames_to_del_dict


def calc_single_file_frames_to_del(nfiles, new_one_hot_dict, count_dict, aug_count_dict):
    
    frames_to_del_dict = count_frames_to_delete(new_one_hot_dict, count_dict, aug_count_dict)
    
    single_file_frames_to_del = {}
    
    for key in frames_to_del_dict:
        single_file_frames_to_del[key] = int(frames_to_del_dict[key] / nfiles)
        
    return single_file_frames_to_del


def save_balanced_data(idx, original_one_hot_dict, new_one_hot_dict, single_file_frames_to_del):
    
    local_frames_to_del = single_file_frames_to_del.copy()
    
    filename = 'D:/steep_training/ski-race/training_data-{}.npy'.format(idx)
    aug_filename = 'D:/steep_training/ski-race/training_data-{}--aug.npy'.format(idx)
    
    valid_file = os.path.isfile(filename)
    
    if not valid_file:
        return (False, {})
    
    train_data = list(np.load(filename))
    aug_train_data = list(np.load(aug_filename))
    
    composite_data = train_data + aug_train_data
    print('Total composite data loaded: ', len(composite_data))
    shuffle(composite_data)
    
    balanced_img_data = []
    balanced_one_hot_data = []
    
    for (img, one_hot) in composite_data:
        
        key = get_key(one_hot, original_one_hot_dict)
        
        if (key in new_one_hot_dict):
            
            if (local_frames_to_del[key] != 0):
                local_frames_to_del[key] -= 1
                
            else:
                
                new_one_hot = new_one_hot_dict[key]
            
                balanced_img_data.append(img)
                balanced_one_hot_data.append(new_one_hot)
                
    balanced_data = list(zip(balanced_img_data, balanced_one_hot_data))
    print('Balanced data contains ', len(balanced_data), ' labeled frames')
                
    composite_filename = 'D:/steep_training/ski-race/balanced/training_data-{}.npy'.format(idx)
    
    np.save(composite_filename, balanced_data)
    
    return (True, local_frames_to_del)


def update_single_file_frames(single_file_frames_to_del, local_frames_correction):
    
    for key in local_frames_correction:
        single_file_frames_to_del[key] += local_frames_correction[key]


def balance_training_data():
    
    (n_files, one_hot_dict, count_dict, aug_count_dict) = count_training_files()
    new_one_hot_dict = rebase_one_hot_dict(0.03, count_dict, one_hot_dict)
    
    raw_pct_dict = count_raw_percentages(new_one_hot_dict, count_dict)    
    single_file_frames_to_del = calc_single_file_frames_to_del(n_files, new_one_hot_dict, count_dict, aug_count_dict)
    
    valid_file = True
    idx = 0
    
    print('Single file frames to delete: ', single_file_frames_to_del)
    
    if not os.path.exists('D:/steep_training/ski-race/balanced'):
        os.mkdir('D:/steep_training/ski-race/balanced')
    
    while valid_file:
        (valid_file, local_frames_correction) = save_balanced_data(idx, one_hot_dict, new_one_hot_dict, single_file_frames_to_del)
        update_single_file_frames(single_file_frames_to_del, local_frames_correction)
        
        idx += 1
          
    new_one_hot_filename = 'D:/steep_training/ski-race/balanced/one_hot_dict.pkl'
    original_key_weights_filename = 'D:/steep_training/ski-race/balanced/original_key_weights.pkl'
    
    with open(new_one_hot_filename, 'wb') as handle:
        pickle.dump(new_one_hot_dict, handle)
        
    with open(original_key_weights_filename, 'wb') as handle:
        pickle.dump(raw_pct_dict, handle)

balance_training_data()