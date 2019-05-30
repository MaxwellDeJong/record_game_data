#! "D:\anaconda\python.exe

from sklearn.preprocessing import LabelBinarizer
import numpy as np
import cv2
import time
import os

from getkeys import key_check
from grabscreen import grab_screen


def get_one_hot(mode):
    
    if (mode == 'wing_suit'):
        relevant_keys = ['W', 'A', 'S', 'D', 'WA', 'WD', 'AJ', 'DL', 'nk']
    elif (mode == 'ski'):
        relevant_keys = ['W', 'A', 'S', 'D', 'AW', 'DW', 'space', 'spaceW', 'AJW', 'ADW', 'K', 'AJ', 'DL', 'nk']        
    enc = LabelBinarizer()
    one_hot_np = enc.fit_transform(relevant_keys)
    one_hot_np = one_hot_np.astype('int')

    one_hot = [list(i) for i in one_hot_np]

    one_hot_dict = dict(zip(relevant_keys, one_hot))

    return one_hot_dict


def get_full_keys_str(keys_str, contains_space, contains_shift):
    
    full_keys_str = keys_str[:]
    
    if contains_space:
        full_keys_str += 'space'
    if contains_shift:
        full_keys_str += 'shift'
        
    return full_keys_str


def filtered_keys_output(keys_str, contains_space, contains_shift, one_hot_dict):
    
    relevant_keys = one_hot_dict.keys()
    relevant_keys = ''.join(relevant_keys)
    
    filtered_keys_str = ''
    
    # Remove any characters that are not relevant for game commands
    for key in keys_str:
        if (key in relevant_keys):
            filtered_keys_str += key
            
    filtered_keys_str = ''.join(sorted(filtered_keys_str))
    
    full_keys_str = get_full_keys_str(filtered_keys_str, contains_space, contains_shift)
    
    # If there is nothing left, return 'no keys'
    if (full_keys_str == ''):
        return one_hot_dict['nk']
    # If the filtered string is a valid key, return the proper vector
    elif (full_keys_str in one_hot_dict.keys()):
        return one_hot_dict[full_keys_str]
    # If removing the shift and space make the key valid, return the
    # resulting vector
    elif (filtered_keys_str in one_hot_dict.keys()):
        return one_hot_dict[filtered_keys_str]
    
    # Otherwise remove keys until we have a valid key
    while filtered_keys_str not in one_hot_dict.keys():
        if (filtered_keys_str == ''):
            return one_hot_dict['nk']
        else:
            filtered_keys_str = filtered_keys_str[:-1]
    
    return one_hot_dict[filtered_keys_str]

        
def keys_to_output(keys, one_hot_dict):
    
    if (keys == []):
        return one_hot_dict['nk']
    
    contains_space = False
    contains_shift = False
    
    if (' ' in keys):
        contains_space = True
        keys.remove(' ')
    if ('\xa0' in keys):
        contains_shift = True
        keys.remove('\xa0')
        
    keys_str = ''.join(sorted(keys))
    
    full_keys_str = keys_str[:]
            
    if contains_space:
        full_keys_str = 'space' + full_keys_str
    if contains_shift:
        full_keys_str = 'shift' + full_keys_str
    
    # If we receive input that doesn't match a key, first strip away characters not used
    
    try:
        output = one_hot_dict[full_keys_str]
    except:
        output = filtered_keys_output(keys_str, contains_shift, contains_space, one_hot_dict)
    
    return output


def get_training_file():

    starting_value = 0

    while True:

        file_name = 'D:/steep_training/ski-race/training_data-{}.npy'.format(starting_value)

        if os.path.isfile(file_name):
            starting_value += 1
        else:
            print('Will start saving to training_data-', starting_value)  
            return (starting_value, file_name)


def gather_data():
    
    mode = 'ski'

    one_hot_dict = get_one_hot(mode)
    (training_idx, training_file) = get_training_file()

    training_data = []
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    title_bar_offset = 30
    n_reset_frames = 100

    last_time = time.time()
    
    paused = False
    compress = True
    print('Starting data collection')
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    
    keys = key_check()
    keys = key_check()

    while(True):
        
        if not paused:

            screen = grab_screen(region=(0, title_bar_offset, 1260, 710))
            screen = cv2.resize(screen, (512, 289))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            
            #keys = key_check()
            output = keys_to_output(keys, one_hot_dict)
            
            (result, comp_screen) = cv2.imencode('.jpg', screen, encode_param)
            training_data.append([comp_screen, output])

            if len(training_data) % 500 == 0:

                curr_time = time.time()

                print('Last 500 samples averaged: ', 500 / (curr_time - last_time), ' FPS')

                last_time = curr_time

                if len(training_data) == 2000:
                    
                    np.save(training_file, training_data)
                    print('SAVED training file ', training_idx)

                    training_data = []
                    training_idx += 1

                    training_file = 'D:/steep_training/ski-race/training_data-{}.npy'.format(training_idx)

        if 'T' in keys:

            if paused:

                paused = False
                print('unpaused!')
                time.sleep(1)

            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
                
            keys = key_check()

        elif '\t' in keys:
            
            if not paused:
           
                if (len(training_data) > n_reset_frames):
                    training_data = training_data[:-n_reset_frames]
                else:
                    training_data = []
                print('Deleted crash data')
                         
            time.sleep(3)
            # Reset recorded keys to avoid duplicate tab
            keys = key_check()
            
        keys = key_check()
 
    
if __name__ == '__main__':
    gather_data()