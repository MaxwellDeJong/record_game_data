# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:42:26 2019

@author: Max
"""

import numpy as np
import cv2
import pickle
import os


def get_mirror_one_hot_dict(one_hot_dict):
    
    mirror_one_hot_dict = {}
    
    for key in one_hot_dict.keys():
        
        mirror_key = ''
        
        if (key == 'AJ'):
            mirror_key = 'DL'
        elif (key == 'DL'):
            mirror_key = 'AJ'
        elif (key == 'A'):
            mirror_key = 'D'
        elif (key == 'AW'):
            mirror_key = 'DW'
        elif (key == 'D'):
            mirror_key = 'A'
        elif (key == 'DW'):
            mirror_key = 'AW'
        elif (key == 'AJW'):
            mirror_key = 'DLW'
        elif (key == 'DLW'):
            mirror_key = 'AJW'
            
        if not (mirror_key == ''):
            mirror_one_hot_dict[key] = one_hot_dict[mirror_key]
            
    return mirror_one_hot_dict


def get_mirror_one_hot(one_hot_vec, one_hot_dict, mirror_one_hot_dict):
    
    for (key, one_hot) in one_hot_dict.items():
        if (one_hot == one_hot_vec):
            return mirror_one_hot_dict[key]
            

def is_symmetric(one_hot_vec, one_hot_dict):
    
    key_found = False
    
    for key, one_hot in one_hot_dict.items():
        if (one_hot_vec == one_hot):
            key_found = True
            break
        
    if not key_found:
        return False
    
    if ('A' in key) or ('D' in key):
        return True
    
    if ('J' in key) or ('L' in key):
        return True
    
    return False


def augment_turns(train_data, one_hot_dict, mirror_one_hot_dict, compressed=False):
    
    aug_img_arr = []
    aug_one_hot_arr = []
    
    for i in train_data:
        
        if compressed:
            img = cv2.imdecode(i[0], 1)
        else:
            img = i[0]
            
        one_hot = i[1]
        
        if (is_symmetric(one_hot, mirror_one_hot_dict)):
            
            img_mirror = cv2.flip(img, 1)
            
            if compressed:
                res, img_mirror_comp = cv2.imencode('.jpg', img_mirror)
            else:
                res = True
                img_mirror_comp = img_mirror
            
            mirror_one_hot = get_mirror_one_hot(one_hot, one_hot_dict, mirror_one_hot_dict)
            if res:
                aug_img_arr.append(img_mirror_comp)
                aug_one_hot_arr.append(mirror_one_hot)
            else:
                print('ERROR: could not flip image')
    
    augmented_data = list(zip(aug_img_arr, aug_one_hot_arr))
    
    return augmented_data


def augment_training_file(filename, idx, one_hot_dict, mirror_one_hot_dict, dirname='ski-race'):
    
    new_filename = 'D:/steep_training/' + dirname + '/training_data-{}--aug.npy'.format(idx)
    
    if not os.path.isfile(new_filename):
    
        train_data = np.load(filename)
        
        augmented_data = augment_turns(train_data, one_hot_dict, mirror_one_hot_dict)
        
        np.save(new_filename, augmented_data)


def augment_all_data(dirname='ski-race'):
    
    with open('D:/steep_training/' + dirname + '/one_hot_dict.pkl', 'rb') as handle:
        one_hot_dict = pickle.load(handle)
        
    mirror_one_hot_dict = get_mirror_one_hot_dict(one_hot_dict)
    
    idx = 0   
    filename = 'D:/steep_training/' + dirname + '/training_data-{}.npy'.format(idx)
    
    valid_file = (os.path.isfile(filename))
    
    while valid_file:
        
        augment_training_file(filename, idx, one_hot_dict, mirror_one_hot_dict, dirname=dirname)
        
        idx += 1      
        filename = 'D:/steep_training/' + dirname + '/training_data-{}.npy'.format(idx)
        
        valid_file = os.path.isfile(filename)

if __name__ == '__main__':
    augment_all_data(dirname='wing-suit')        