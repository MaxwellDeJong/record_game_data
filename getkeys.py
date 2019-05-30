# Citation: Box Of Hats (https://github.com/Box-Of-Hats )
# Modified from github.com/Sentdex/pygta

import win32api as wapi

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ ":
    keyList.append(char)
keyList.append('\t')
keyList.append('\xa0')

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys