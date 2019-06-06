# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:32:11 2019

@author: Max
"""
from process_raw_training_data import process_raw_training_data
from balance_training_data import balance_training_data
from split_validation_data import calc_label_dict
from calculate_normalization_weights import calculate_normalization_coefficients

def process_all_training_data():
    
    # First, process our raw training files to determine relevant categories
    process_raw_training_data()
    
    # Augment and balance training data
    balance_training_data()

    # Split up training data into individual files and calculate normalization weights
    calculate_normalization_weights()

    # Split up validation files
    calc_label_dict()


process_all_training_data()
