# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:36:42 2019

@author: Max
"""

import pickle
import numpy as np
import os

def deep_dict_copy(dict_to_copy):
    '''Makes a new copy of dictionary and a deep copy of its arrays.'''
    
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
    '''Transforms from a one-hot array to the associated key press.'''
    
    for (key, one_hot) in one_hot_dict.items():
        if (one_hot == one_hot_vec):
            return key
        
    print('Could not find matching vector for ', one_hot_vec)
    print('One hot dictionary: ', one_hot_dict)


def count_file(filename, one_hot_dict, count_dict):
    '''Counts the number of all keys pressed in a single file and updates
    the count dictionary.'''
    
    training_data = np.load(filename)
    print(filename, ' had ', len(training_data), ' entries')
    
    one_hot_arr = [i[1] for i in training_data]
    
    for one_hot in one_hot_arr:
        key = get_key(one_hot, one_hot_dict)
        count_dict[key] += 1
        
        
def count_files(idx, one_hot_dict, count_dict):
    '''Counts the number of all key presses in a training file and its
    associated augmented file.'''
    
    filename = 'D:/steep_training/ski-race/training_data-{}.npy'.format(idx)
    
    count_file(filename, one_hot_dict, count_dict)
    
    next_filename = 'D:/steep_training/ski-race/training_data-{}.npy'.format(idx+1)
    
    valid_next_file = os.path.isfile(next_filename)
    
    return valid_next_file
    
    
def count_training_files():
    '''Counts all key presses across all training and augmented files.'''
    
    one_hot_dict = load_one_hot_dict()
    
    count_dict = initialize_count_dict(one_hot_dict)
    
    n_files = 0
    
    idx = 0   
    filename = 'D:/steep_training/ski-race/training_data-{}.npy'.format(idx)
                                                         
    valid_file = os.path.isfile(filename)
    
    while valid_file:
        
        print('Counting keystrokes for file ', idx)
        
        valid_file = count_files(idx, one_hot_dict, count_dict)
        n_files += 1
        idx += 1
        
    return (n_files, one_hot_dict, count_dict)


def find_keys_to_remove(cutoff, count_dict):
    '''Determines which keys occur too infrequently to be considered choices
    by the neural network.'''
    
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
    tot_count_dict = {}
    
    tot_count = 0
    for key in new_one_hot_dict:
        tot_count += count_dict[key]
        
    for key in new_one_hot_dict:
        raw_pct_dict[key] = count_dict[key] / tot_count
        tot_count_dict[key] = count_dict[key]
        
    return (raw_pct_dict, tot_count_dict)


def process_raw_training_data():
    
    (n_files, one_hot_dict, raw_count_dict) = count_training_files()
    new_one_hot_dict = rebase_one_hot_dict(0.01, raw_count_dict, one_hot_dict)
    
    (raw_pct_dict, sig_count_dict) = count_raw_percentages(new_one_hot_dict, raw_count_dict)   
    
    if not os.path.exists('D:/steep_training/ski-race/balanced'):
        os.mkdir('D:/steep_training/ski-race/balanced')
    
    new_one_hot_filename = 'D:/steep_training/ski-race/balanced/one_hot_dict.pkl'
    original_one_hot_filename = 'D:/steep_training/ski-race/balanced/original_one_hot_dict.pkl'
    original_key_weights_filename = 'D:/steep_training/ski-race/balanced/original_key_weights.pkl'
    tot_counts_filename = 'D:/steep_training/ski-race/balanced/significant_count_dict.pkl'
    
    with open(new_one_hot_filename, 'wb') as handle:
        pickle.dump(new_one_hot_dict, handle)
        
    with open(original_key_weights_filename, 'wb') as handle:
        pickle.dump(raw_pct_dict, handle)
        
    with open(tot_counts_filename, 'wb') as handle:
        pickle.dump(sig_count_dict, handle)

    with open(original_one_hot_filename, 'wb') as handle:
        pickle.dump(one_hot_dict, handle)


if __name__ == '__main__':
    process_raw_training_data()
