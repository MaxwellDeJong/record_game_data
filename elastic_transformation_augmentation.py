# -*- coding: utf-8 -*-
"""
Adapted from code by Nasim Rahaman
Original code: https://gist.github.com/nasimrahaman/8ed04be1088e228c21d51291f47dd1e6
"""

import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# Elastic transform

def elastic_transform(img):
    """`images` is a numpy array of shape (K, M, N) of K images of size M*N."""
    
    images = [img[:, :, i] for i in range(3)]
    
    print(np.shape(images))
    print(images[0].shape)
    
    alpha = 2000
    sigma = 40
    
    rng = np.random.RandomState(42)
    interpolation_order = 1
    
    # Take measurements
    image_shape = images[0].shape
    # Make random fields
    dx = rng.uniform(-1, 1, image_shape) * alpha
    dy = rng.uniform(-1, 1, image_shape) * alpha
    # Smooth dx and dy
    sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
    sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
    # Make meshgrid
    x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
    # Distort meshgrid indices
    distorted_indices = (y + sdy).reshape(-1, 1), \
                            (x + sdx).reshape(-1, 1)

    # Map cooordinates from image to distorted index set
    transformed_images = [map_coordinates(image, distorted_indices, mode='reflect',
                                          order=interpolation_order).reshape(image_shape)
                          for image in images]
    
    return np.dstack((transformed_images[0], transformed_images[1], transformed_images[2]))


def test_transform():
    
    img = np.load('D:/steep_training/ski-race/training_data-3--aug.npy')[89][0]
    
    full_img = cv2.imdecode(img, 1)
    cv2.imshow('Original', full_img)
    
    distorted_img = elastic_transform_2D(2000, 40, full_img)
    
    cv2.imshow('Sunspot', distorted_img)
    
    if (cv2.waitKey(0) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        
test_transform()