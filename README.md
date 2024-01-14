# FILEPATH: main.py

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
