# HISTOGRAM TYPES AND IMPLEMENTATIONS

This repository includes the 3D, per-channel, grid based and non-grid based, RGB, and HSV based histogram implementations coded using only numpy functions, from scratch. A histogram provides information about the intensity or color distribution in an image or signal. The spatial meaning of RGB color histograms is related to the distribution of intensity values in individual color channels R,G and B, while the spatial meaning of HSV color histograms is related to perceptual attributes such as hue, saturation and value. 

METU - CENG483 (Introduction to Computer Vision) Course Take Home Exam - 1 Work

FILEPATH: main.py

"""
This script is used for making experiments with the different configurations given in THE1.

Usage:
    python main.py

Options:
    Go to the main.py and change the parameters given in the top of the code.
    if3d: Controls if 3D color histogram will be used instead of per-channel setting. (True: 3D, False: per-channel) (Defaults to False)
    ifHsv: Controls if the inputs will be converted to HSV space or not. (True: HSV, False: RGB) (Defaults to False)
    interval: Controls the number of quantization interval. (Defaults to 16)
    grid_n: Controls the number of grids (nxn) (Defaults to 1: no grids)

Dependencies:
    opencv-python
    numpy
    os
    glob
    utils
"""
