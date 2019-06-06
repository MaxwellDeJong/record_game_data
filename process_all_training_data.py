# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:32:11 2019

@author: Max
"""
from process_raw_training_data import process_raw_training_data
from balance_training_data import balance_training_data

def generate_all_training_data():
    
    # First, process our raw training files to determine relevant categories
    process_raw_training_data()
    
    # Augment and balance training data
    balance_training_data()
    
    