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


def save_balanced_data(idx, original_one_hot_dict, new_one_hot_dict, running_single_file_frame_diff):
    
    filename = 'D:/steep_training/ski-race/training_data-{}.npy'.format(idx)
    
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

            # If we have no difference between our needed and expected number of frames,
            # add the data to our training set
            if (int_frame_diff == 0):

                balanced_img_data.append(img)
                balanced_ont_hot_data.append(one_hot)

            # If we have too many frames, don't add them to our training set
            elif (int_frame_diff > 0):
            
                running_single_file_frame_diff[key] -= 1
                    
            # If we don't have enough frames, augment the ones we do have
            else:
                    
                balanced_img_data.append(img)
                balanced_one_hot_data.append(one_hot)

                augmented_images = all_augmentation[running_single_file_frame_diff[key] - 1]
                running_single_file_frame_diff[key] += len(augmented_images)
                    
                for aug_img in augmented_images:
                    balanced_img_data.append(aug_img)
                    balanced_one_hot_data.append(one_hot)
            
    balanced_data = list(zip(balanced_img_data, balanced_one_hot_data))
    shuffle(balanced_data)
    print('Balanced data contains ', len(balanced_data), ' labeled frames')
                
    balanced_filename = 'D:/steep_training/ski-race/balanced/training_data-{}.npy'.format(idx)
    
    np.save(balanced_filename, balanced_data)
    

def load_dicts():

    with open('D:/steep_training/ski-race/balanced/significant_count_dict.pkl', 'rb') as handle:
        count_dict = pickle.load(handle)

    with open('D:/steep_training/ski-race/balanced/one_hot_dict.pkl', 'rb') as handle:
        new_one_hot_dict = pickle.load(handle)

    with open('D:/steep_training/ski-race/balanced/original_one_hot_dict.pkl', 'rb') as handle:
        original_one_hot_dict = pickle.load(handle)

    return (count_dict, new_one_hot_dict, original_one_hot_dict)


def count_training_files():

    idx = 0

    while True:

        filename = 'D:/steep-training/ski-race/training_data-{}.npy'.format(idx)

        if (os.path.isfile(filename)):
            idx += 1
        else:
            return idx


def update_single_file_frame_diff(single_file_frame_diff, running_single_file_frame_diff):

    if (running_single_file_frame_diff is None):

        running_single_file_frame_diff = {}

        for key in single_file_frame_diff:
            running_single_file_frame_diff[key] = single_file_frame_diff[key]

    else:

        for key in single_file_frame_diff:

            running_single_file_frame_diff[key] += single_file_frame_diff[key]
    

def balance_training_data():

    (count_dict, new_one_hot_dict, original_one_hot_dict) = load_dicts()
    n_training_files = count_training_files()

    single_file_frame_diff = calc_single_file_frame_diff(nfiles, new_one_hot_dict, count_dict)
    running_file_frame_diff = None

    for idx in range(n_training_files):
    
        update_single_file_frame_diff(single_file_frame_dff, running_single_file_frame_diff)
        save_balanced_data(idx, one_hot_dict, new_one_hot_dict, running_single_file_frame_diff)


if __name__ == '__main__':
    balance_training_data()
