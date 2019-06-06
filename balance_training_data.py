# -*- coding: utf-8 -*-
"""
Created on Thu May 30 01:03:30 2019

@author: Max
"""
import pickle
from random import shuffle
import numpy as np
import os
from master_data_augmentation import all_augmentation

def get_key(one_hot_vec, one_hot_dict):
    '''Transforms from a one-hot array to the associated key press.'''
    
    for (key, one_hot) in one_hot_dict.items():
        if (one_hot == one_hot_vec):
            return key
        
    print('Could not find matching vector for ', one_hot_vec)
    print('One hot dictionary: ', one_hot_dict)
    
    
def load_count_dict():
    
    with open('D:/steep_training/ski-race/balanced/tot_count_dict.pkl', 'rb') as handle:
        count_dict = pickle.load(handle)
        
    return count_dict


def count_desired_frames(count_dict):
    
    desired_frames = min(count_dict['AW'], count_dict['DW'])
    
    return desired_frames


def count_frame_diff(new_one_hot_dict, count_dict, n_desired_frames):
    
    frame_diff_dict = {}
    
    for key in new_one_hot_dict:
        
        diff = count_dict[key] - n_desired_frames        
        frame_diff_dict[key] = diff
        
    return frame_diff_dict


def calc_single_file_frame_diff(nfiles, new_one_hot_dict, count_dict):
    
    n_desired_frames = count_desired_frames(count_dict)
    
    frame_diff_dict = count_frame_diff(new_one_hot_dict, count_dict, n_desired_frames)
    
    single_file_frame_diff = {}
    
    for key in frame_diff_dict:
        single_file_frame_diff[key] = frame_diff_dict[key] / nfiles
        
    return single_file_frame_diff


def save_balanced_data(idx, original_one_hot_dict, new_one_hot_dict, single_file_frame_diff, running_single_frame_diff):
    
    filename = 'D:/steep_training/ski-race/training_data-{}.npy'.format(idx)
    
    valid_file = os.path.isfile(filename)
    
    if not valid_file:
        return (False, {})
    
    train_data = list(np.load(filename))
    shuffle(train_data)
    
    balanced_img_data = []
    balanced_one_hot_data = []
    
    for (img, old_one_hot) in train_data:
        
        key = get_key(old_one_hot, original_one_hot_dict)
        
        if (key in new_one_hot_dict):
            
            one_hot = new_one_hot_dict[key]
            
            frame_diff = running_single_file_frame_diff[key]
            int_frame_diff = int(frame_diff)
            
            if (int_frame_diff != 0):
                
                if (int_frame_diff > 0):
                    running_single_file_frame_diff[key] -= 1
                    
                else:
                    
                    balanced_img_data.append(img)
                    balanced_one_hot_data.append(one_hot)
                    
                    augmented_images = all_augmentation[single_file_frame_diff[key] - 1]
                    running_single_frame_diff[key] += len(augmented_images)
                    
                    for aug_img in augmented_images:
                        balanced_img_data.append(aug_img)
                        balanced_one_hot_data.append(one_hot)
            
            else:
                balanced_img_data.append(img)
                balanced_ont_hot_data.append(one_hot)
                        

                
    balanced_data = list(zip(balanced_img_data, balanced_one_hot_data))
    shuffle(balanced_data)
    print('Balanced data contains ', len(balanced_data), ' labeled frames')
                
    composite_filename = 'D:/steep_training/ski-race/balanced/training_data-{}.npy'.format(idx)
    
    np.save(composite_filename, balanced_data)
    
    return (True, local_frames_to_del)


def update_single_file_frames(single_file_frames_to_del, local_frames_correction):
    
    for key in local_frames_correction:
        single_file_frames_to_del[key] += local_frames_correction[key]


def balance_training_data():
    
    (n_files, one_hot_dict, count_dict, aug_count_dict) = count_training_files()
    new_one_hot_dict = rebase_one_hot_dict(0.01, count_dict, one_hot_dict)
    
    print(new_one_hot_dict)
    print(count_dict)
    
#    raw_pct_dict = count_raw_percentages(new_one_hot_dict, count_dict)    
#    single_file_frames_to_del = calc_single_file_frames_to_del(n_files, new_one_hot_dict, count_dict, aug_count_dict)
#    
#    valid_file = True
#    idx = 0
#    
#    print('Single file frames to delete: ', single_file_frames_to_del)
#    
#    if not os.path.exists('D:/steep_training/ski-race/balanced'):
#        os.mkdir('D:/steep_training/ski-race/balanced')
#    
#    while valid_file:
#        (valid_file, local_frames_correction) = save_balanced_data(idx, one_hot_dict, new_one_hot_dict, single_file_frames_to_del)
#        update_single_file_frames(single_file_frames_to_del, local_frames_correction)
#        
#        idx += 1
#          
#    new_one_hot_filename = 'D:/steep_training/ski-race/balanced/one_hot_dict.pkl'
#    original_key_weights_filename = 'D:/steep_training/ski-race/balanced/original_key_weights.pkl'
#    
#    with open(new_one_hot_filename, 'wb') as handle:
#        pickle.dump(new_one_hot_dict, handle)
#        
#    with open(original_key_weights_filename, 'wb') as handle:
#        pickle.dump(raw_pct_dict, handle)

balance_training_data()