# -*- coding: utf-8 -*-
"""

"""

from PIL import Image, ImageDraw, ImageOps
from random import randint, uniform
#import matplotlib.pyplot as plt
#import numpy as np




def DrawRectangle():    
    x_c = randint(0, 128)
    y_c = randint(0, 128)
    half_height = randint(6, 64)
    half_width = randint(6, 64)
    line_width = randint(1, 5)
    img_rect = Image.new("L", (128, 128), "black")
    draw = ImageDraw.Draw(img_rect)
    draw.rectangle((x_c-half_width, y_c-half_height, x_c+half_width, y_c+half_height), fill="white")
    draw.rectangle((x_c-half_width+line_width, y_c-half_height+line_width, x_c+half_width-line_width, y_c+half_height-line_width), fill="black")
    return img_rect

def CreateImage(idx, set_name):
    image = Image.new("L", (128, 128), "black")
    n_rect=randint(1,5)
    for i in range(0, n_rect):
        image = Image.composite(image, DrawRectangle(), image)
    rotation_angle = uniform(-10, 10)
    image=image.rotate(rotation_angle)
    image=ImageOps.invert(image)
    image.save(set_name+"set/img"+str(idx).zfill(4)+"_"+str(rotation_angle)+".png")
    
#    plt.imshow(np.asarray(image))
    

if __name__ == '__main__':
    trainset_size = 10000
    for i in range(0, trainset_size):
        CreateImage(idx=i, set_name="train")
        
    validset_size = 1000
    for i in range(0, validset_size):
        CreateImage(idx=i, set_name="valid")