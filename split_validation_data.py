import os
import numpy as np
import pickle
import cv2


def load_dicts(dirname='ski-race'):

    old_one_hot_filename = 'D:/steep_training/' + dirname + '/balanced/original_one_hot_dict.pkl'
    new_one_hot_filename = 'D:/steep_training/' + dirname + '/balanced/one_hot_dict.pkl'

    with open(old_one_hot_filename, 'rb') as handle:
        old_one_hot_dict = pickle.load(handle)

    with open(new_one_hot_filename, 'rb') as handle:
        new_one_hot_dict = pickle.load(handle)

    return (old_one_hot_dict, new_one_hot_dict)


def get_key(one_hot_vec, one_hot_dict):
    '''Transforms from a one-hot array to the associated key press.'''
    
    for (key, one_hot) in one_hot_dict.items():
        if (one_hot == one_hot_vec):
            return key
        
    print('Could not find matching vector for ', one_hot_vec)
    print('One hot dictionary: ', one_hot_dict)


def transform_one_hot(one_hot_vec, old_one_hot_dict, new_one_hot_dict):

    key = get_key(one_hot_vec, old_one_hot_dict)
    
    if key in new_one_hot_dict:
        return new_one_hot_dict[key]
    
    return None


def save_individual_file(count, img_arr, one_hot_arr, old_one_hot_dict, \
                         new_one_hot_dict, label_dict, dirname='ski-race'):

    idx = 0
    for i in range(len(img_arr)):
            
        filename = 'D:/steep_training/' + dirname + \
            '/balanced/validation_frame-{}.npy'.format(count + idx)
        label = transform_one_hot(one_hot_arr[i], old_one_hot_dict, new_one_hot_dict)
        
        if not label is None:
            label_dict[count + idx] = label
            idx += 1

            np.save(filename, img_arr[i])
            
    return idx


def parse_file(filename, global_count, old_one_hot_dict, new_one_hot_dict, \
               label_dict, compressed=False, dirname='ski-race'):
    
    training_data = np.load(filename)
    
    if compressed:
        img_arr = [cv2.imdecode(i[0], 1) for i in training_data]
    else:
        img_arr = [i[0] for i in training_data]
    one_hot_arr = [i[1] for i in training_data]
    
    n_valid_frames = save_individual_file(global_count, img_arr, one_hot_arr, \
        old_one_hot_dict, new_one_hot_dict, label_dict, dirname=dirname)

    return n_valid_frames


def calc_label_dict(dirname='ski-race'):
    
    label_dict = {}
    idx = 0
    global_count = 0

    (old_one_hot_dict, new_one_hot_dict) = load_dicts(dirname=dirname)
    
    delete_bulk = False
    
    while True:
        filename = 'D:/steep_training/' + dirname + '/validation_data-{}.npy'.format(idx)
        
        if (os.path.isfile(filename)):

            n_samples = parse_file(filename, global_count, old_one_hot_dict, \
                                   new_one_hot_dict, label_dict, dirname=dirname)
            global_count += n_samples
            
            print('Finished analyzing file ', idx)
            
            if delete_bulk:
                os.remove(filename)
                
            idx += 1
            
        else:

            label_filename = 'D:/steep_training/' + dirname + '/balanced/validation_label_dict.pkl'

            with open(label_filename, 'wb') as handle:
                pickle.dump(label_dict, handle)

            break

if __name__ == '__main__':
    calc_label_dict(dirname='wing-suit')
