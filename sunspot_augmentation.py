import numpy as np
from scipy.stats import multivariate_normal
import random
import cv2
import matplotlib.pyplot as plt


def generate_2d_gaussian(x_mu, y_mu, std_x, std_y, intensity):
    
    mean = [x_mu, y_mu]
    cov = [[std_x * std_x, 0], [0, std_y * std_y]]
    
    x, y = np.mgrid[0:180:1, 0:320:1]
    pos = np.empty(x.shape + (2,))
    
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    
    rv = multivariate_normal(mean, cov)
    gaussian = rv.pdf(pos)
    
    scaling = intensity / np.max(gaussian)
    
    return (x, y, scaling * gaussian)


def generate_sunspot(full_img):
    
    x_mu = random.randint(20, 160)
    y_mu = random.randint(100, 300)
    
    std_x = random.randint(15, 40)
    std_y = random.randint(15, 40)
    
    intensity = random.randint(120, 200)
    
    (_, _, gaussian_mask) = generate_2d_gaussian(x_mu, y_mu, std_x, std_y, intensity)

    return add_arrays(full_img, gaussian_mask)


def add_arrays(img_arr, mask_arr):
    
    c0 = np.clip(img_arr[:, :, 0] + mask_arr, 0, 255).astype('uint8')
    c1 = np.clip(img_arr[:, :, 1] + mask_arr, 0, 255).astype('uint8')
    c2 = np.clip(img_arr[:, :, 2] + mask_arr, 0, 255).astype('uint8')
    
    return np.dstack((c0, c1, c2)).astype('uint8')


def test_sunspot():
    
    img = np.load('D:/steep_training/ski-race/training_data-3.npy')[89][0]
    
    full_img = cv2.imdecode(img, 1)
    cv2.imshow('Original', full_img)
    
    np.save('D:/steep_training/original.npy', full_img)
    
    sun_spot_img = generate_sunspot(full_img)
    np.save('D:/steep_training/sunspot.npy', sun_spot_img)
    
    cv2.imshow('Sunspot', sun_spot_img)
    
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        
    cv2.imwrite('sun_spot.png', sun_spot_img)
    
#test_sunspot()