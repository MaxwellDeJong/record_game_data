import numpy as np
from scipy.stats import multivariate_normal
import random
import cv2


def generate_2d_gaussian(x_mu, y_mu, std_x, std_y, intensity):
    
    mean = [x_mu, y_mu]
    cov = [[std_x * std_x, 0], [0, std_y * std_y]]
    
    x, y = np.mgrid[0:289:1, 0:512:1]
    pos = np.empty(x.shape + (2,))
    
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    
    rv = multivariate_normal(mean, cov)
    gaussian = rv.pdf(pos)
    
    scaling = intensity / np.max(gaussian)
    
    return (x, y, scaling * gaussian)


def generate_sunspot(full_img):
    
    x_mu = random.randint(20, 269)
    y_mu = random.randint(200, 492)
    
    std_x = random.randint(15, 40)
    std_y = random.randint(15, 40)
    
    intensity = random.randint(120, 200)
    
    (_, _, gaussian_mask) = generate_2d_gaussian(x_mu, y_mu, std_x, std_y, intensity)
    
    full_img = add_arrays(full_img, gaussian_mask)
    
    return full_img


def add_arrays(original_img_arr, mask_arr):
    
    img_arr = np.copy(original_img_arr)
    
    (max_x, max_y, n_channel) = np.shape(img_arr)
    
    for i in range(max_x):
        for j in range(max_y):
            for k in range(n_channel):
                img_arr[i, j, k] = int(min(img_arr[i, j, k] + mask_arr[i, j], 254))
            
    return img_arr


def test_sunspot():
    
    img = np.load('D:/steep_training/ski-race/training_data-3--aug.npy')[89][0]
    
    full_img = cv2.imdecode(img, 1)
    cv2.imshow('Original', full_img)
    
    sun_spot_img = generate_sunspot(full_img)
    
    cv2.imshow('Sunspot', sun_spot_img)
    
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        cv2.destroyAllWindows()