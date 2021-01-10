#! "D:\anaconda\python.exe
"""
Created on Sun Oct  4 12:46:42 2020

@author: Max
"""
import time

from getkeys import key_check
from data_collection import get_one_hot, keys_to_output


one_hot = get_one_hot('wing_suit', add_shift=True)
valid_keys = list(one_hot.values())
n_keys = 0

while n_keys < 1000:
    key = key_check()
    output = keys_to_output(key, one_hot)
    if output in valid_keys:
        print('valid output: ', output)
    else:
        print('invalid output: ', output)
    time.sleep(0.01)