# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:01:21 2019

@author: Max
"""

from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_core(filename):
    
    text = pytesseract.image_to_string(Image.open(filename))
    
    return text

print(ocr_core('D:/steep_training/time.png'))