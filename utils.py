
import cv2
import numpy as np
import os
import glob

def per_channel_histogram(image,interval):
    histogram_b = np.bincount(np.int_(image[:,:,0].flatten()/interval),minlength=int(256/interval))
    histogram_g = np.bincount(np.int_(image[:,:,1].flatten()/interval),minlength=int(256/interval))
    histogram_r = np.bincount(np.int_(image[:,:,2].flatten()/interval),minlength=int(256/interval))
    #take an array of three histograms
    histogram = np.array([histogram_b, histogram_g, histogram_r])
    return histogram

def color_3d_histogram(image,interval):
    r_combinations, g_combinations, b_combinations = np.meshgrid(np.arange(len(per_channel_histogram(image,interval)[0])), np.arange(len(per_channel_histogram(image,interval)[1])), np.arange(len(per_channel_histogram(image,interval)[2])), indexing='ij')
    comb_arr = np.stack([r_combinations, g_combinations, b_combinations], axis=-1)
    flattened = comb_arr.reshape((-1, 3)).astype("uint32")
    histogram_3d = np.zeros((int(256/interval)**3))
    for flat in flattened: 
        index = int(flat[0] * (256/interval) ** 2 + flat[1] * (256/interval) + flat[2])
        histogram_3d[index] = per_channel_histogram(image,interval)[0][flat[0]] + per_channel_histogram(image,interval)[1][flat[1]] + per_channel_histogram(image,interval)[2][flat[2]]
    return histogram_3d

def split_image(image, grid_size):

    # Get the dimensions of the original image
    height, width, channels = image.shape

    # Calculate the size of each grid cell
    cell_height = height // grid_size
    cell_width = width // grid_size

    # Reshape the image to a 2D array to make slicing easier
    reshaped_image = image.reshape(grid_size, cell_height, grid_size, cell_width, channels)

    # Swap axes to get the final grid
    grid = reshaped_image.transpose(0, 2, 1, 3, 4).reshape(-1, cell_height, cell_width, channels)

    return grid

def rgb_to_hsv(image):
    #convert bgr to rgb with numpy
    image = image[:, :, ::-1]
    height, width, channels = image.shape
    img_exp = np.expand_dims(image, axis=0)
    
    img_exp =  img_exp.reshape(-1, channels) / 255.0
    r,g,b = img_exp[:,2], img_exp[:,1], img_exp[:,0]
    maxc = np.maximum(np.maximum(r,g),b)
    max_index = np.argmax(img_exp,axis = 1)
    minc = np.minimum(np.minimum(r,g),b)
    delta = maxc - minc 

    value = maxc

    # saturation is 0 if maxc = 0, delta/maxc if maxc>0
    saturation = (maxc > 0) * (delta / maxc)

    nonzero_delta = delta != 0
    hue = np.zeros(maxc.shape)
    mask1 = (max_index == 0) & nonzero_delta
    mask2 = (max_index == 1) & nonzero_delta
    mask3 = (max_index == 2) & nonzero_delta


    hue[mask1] = (1/6) * (((img_exp[mask1,1]-img_exp[mask1,2])/delta[mask1]) % 6)
    hue[mask2] = (1/6) * (((img_exp[mask2,2]-img_exp[mask2,1])/delta[mask2]) + 2 )
    hue[mask3] = (1/6) * (((img_exp[mask3,0]-img_exp[mask3,1])/delta[mask3]) + 4 )
    

    '''# map between 0 and 255
    hue = hue * 255
    hue = hue.astype("uint8")
    saturation = saturation * 255
    saturation = saturation.astype("uint8")
    value = value * 255
    value = value.astype("uint8")'''

    hsv_image = np.stack([hue, saturation, value], axis=-1)*255
    hsv_image = hsv_image.astype("uint8")
    hsv_image = hsv_image.reshape(height,width,channels)
    return hsv_image

def l1(hist):
    prob_dist = hist / np.sum(hist)
    return prob_dist

def similarity_check_per_channel(image1,image2,interval):
    histogram_1 = per_channel_histogram(image1,interval)
    histogram_2 = per_channel_histogram(image2,interval)
    #calculate the similarity between those two histograms for each channel and take an average
    sm1 = np.sum(np.minimum(l1(histogram_1[0]),l1(histogram_2[0])))
    sm2 = np.sum(np.minimum(l1(histogram_1[1]),l1(histogram_2[1])))
    sm3 = np.sum(np.minimum(l1(histogram_1[2]),l1(histogram_2[2])))
    similarity = (sm1 + sm2 + sm3)/3
    return similarity

def similarity_check_per_channel_histograms(histogram_1,histogram_2):
    sm1 = np.sum(np.minimum(l1(histogram_1[0]),l1(histogram_2[0])))
    sm2 = np.sum(np.minimum(l1(histogram_1[1]),l1(histogram_2[1])))
    sm3 = np.sum(np.minimum(l1(histogram_1[2]),l1(histogram_2[2])))
    similarity = (sm1 + sm2 + sm3)/3
    return similarity

def similarity_check_3d(image1,image2,interval):
    histogram_1 = color_3d_histogram(image1,interval)
    histogram_2 = color_3d_histogram(image2,interval)
    #calculate the similarity between those two histograms for each channel and take an average
    similarity = np.sum(np.minimum(l1(histogram_1),l1(histogram_2)))
    return similarity

def similarity_check_3d_histograms(histogram_1,histogram_2):
    #calculate the similarity between those two histograms for each channel and take an average
    similarity = np.sum(np.minimum(l1(histogram_1),l1(histogram_2)))
    return similarity

def get_images(path):
    #get all the images in the directory
    images = [cv2.imread(file) for file in glob.glob(path)]
    return images
def get_image_names(path):
    #get all the images in the directory
    images = [file.split(os.sep)[1] for file in glob.glob(path)]
    return images

def top1_calculator(support,query,names_support,names_query,interval,type,hsv_true):
    counter = 0
    if hsv_true:
        supportcop = []
        querycop = []
        for xk in range(len(support)):
            supportcop.append(rgb_to_hsv(support[xk]))
        for yk in range(len(query)):
            querycop.append(rgb_to_hsv(query[yk]))
    else:
        supportcop = support
        querycop = query
            
    if type == '3d':
        histlistsupport = np.array([color_3d_histogram(i, interval) for i in supportcop])
        histlistquery = np.array([color_3d_histogram(i, interval) for i in querycop])
        for i in range(len(histlistquery)):
            similarities = np.array([similarity_check_3d_histograms(histlistquery[i], histlistsupport[j]) for j in range(len(histlistsupport))])
            if names_query[i]==names_support[np.argmax(similarities)]:
                counter+=1
    elif type == 'per_channel':
        histlistsupport = np.array([per_channel_histogram(i, interval) for i in supportcop])
        histlistquery = np.array([per_channel_histogram(i, interval) for i in querycop])
        for i in range(len(histlistquery)):
            similarities = np.array([similarity_check_per_channel_histograms(histlistquery[i], histlistsupport[j]) for j in range(len(histlistsupport))])
            if names_query[i]==names_support[np.argmax(similarities)]:
                counter+=1
    print("Top1 Accuracy: ",counter/len(histlistquery)*100)

    return counter/len(histlistquery)*100


def top1_calculator_grid(support,query,names_support,names_query,interval,type,hsv_true,grid_size):
    counter = 0
    if hsv_true:
        supportcop = []
        querycop = []
        for xk in range(len(support)):
            supportcop.append(rgb_to_hsv(support[xk]))
        for yk in range(len(query)):
            querycop.append(rgb_to_hsv(query[yk]))
    else:
        supportcop = support
        querycop = query
    
    support_grid = np.array([split_image(i, grid_size) for i in supportcop])
    query_grid = np.array([split_image(i, grid_size) for i in querycop])

    if type == '3d':

        histlistsupport = np.array([[color_3d_histogram(j, interval) for j in i] for i in support_grid])
        histlistquery = np.array([[color_3d_histogram(j, interval) for j in i] for i in query_grid])
        
        for i in range(len(histlistquery)):
            similarities = np.array([np.mean([similarity_check_3d_histograms(histlistquery[i][k], histlistsupport[j][k]) for k in range(grid_size * grid_size)]) for j in range(len(histlistsupport))])
            if names_query[i]==names_support[np.argmax(similarities)]:
                counter+=1


    elif type == 'per_channel':

        histlistsupport = np.array([[per_channel_histogram(j, interval) for j in i] for i in support_grid])
        histlistquery = np.array([[per_channel_histogram(j, interval) for j in i] for i in query_grid])
        
        for i in range(len(histlistquery)):
            similarities = np.array([np.mean([similarity_check_per_channel_histograms(histlistquery[i][k], histlistsupport[j][k]) for k in range(grid_size * grid_size)]) for j in range(len(histlistsupport))])
            if names_query[i]==names_support[np.argmax(similarities)]:
                counter+=1
        
    
    print("Top1 Accuracy: ",counter/len(names_query)*100)
    return counter/len(names_query)*100