import cv2
import numpy as np
import os
import glob
from utils import *

# SET THE PARAMETERS HERE
if3d = False # if false, then per channel
ifHsv = False # if false, then rgb
interval = 16
grid_n = 1 #if 1, then no grid
query_1,query_2,query_3 = True,True,True

peror3d = "3d" if if3d else "per_channel"
hsv_or_rgb = "hsv" if ifHsv else "rgb"


if __name__== '__main__':
    #get the images
    images_support = get_images("support_96/*.jpg")
    image_names_support = get_image_names("support_96/*.jpg")
    images_query1 = get_images("query_1/*.jpg")
    image_names_query1 = get_image_names("query_1/*.jpg")
    images_query2 = get_images("query_2/*.jpg")
    image_names_query2 = get_image_names("query_2/*.jpg")
    images_query3 = get_images("query_3/*.jpg")
    image_names_query3 = get_image_names("query_3/*.jpg")

    string = """
            3D or Per Channel: %s
            HSV or RGB: %s
            Interval: %s
            Grid: %s"""%(peror3d,hsv_or_rgb,interval,grid_n)
    if grid_n == 1:

        if query_1:
            print("----------------------------------------")
            print('Testing Query 1 with the following parameters:',string)
            top1_calculator(images_support,images_query1,image_names_support,image_names_query1,interval,peror3d,ifHsv)
        if query_2:
            print("----------------------------------------")
            print('Testing Query 2 with the following parameters:',string)
            top1_calculator(images_support,images_query2,image_names_support,image_names_query2,interval,peror3d,ifHsv)
        if query_3:
            print("----------------------------------------")
            print('Testing Query 3 with the following parameters:',string)
            top1_calculator(images_support,images_query3,image_names_support,image_names_query3,interval,peror3d,ifHsv)
    else:
        if query_1:
            print("----------------------------------------")
            print('Testing Query 1 with the following parameters:',string)
            top1_calculator_grid(images_support,images_query1,image_names_support,image_names_query1,interval,peror3d,ifHsv,grid_n)
        if query_2:
            print("----------------------------------------")
            print('Testing Query 2 with the following parameters:',string)
            top1_calculator_grid(images_support,images_query2,image_names_support,image_names_query2,interval,peror3d,ifHsv,grid_n)
        if query_3:
            print("----------------------------------------")
            print('Testing Query 3 with the following parameters:',string)
            top1_calculator_grid(images_support,images_query3,image_names_support,image_names_query3,interval,peror3d,ifHsv,grid_n)

