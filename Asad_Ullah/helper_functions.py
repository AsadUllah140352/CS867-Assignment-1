import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mp_img
import numpy as np
import sys
import math
from scipy.ndimage import gaussian_filter, median_filter, sobel,gaussian_laplace
from skimage.util import random_noise
import scipy.misc



def allImages():
    x=[]
    for image_name in os.listdir("./images"):
        x.append(image_name)
    return x

def displayHistogram(image,title):
    plt.hist(image.ravel(),256,[0,256])
    plt.title("Histogram of " + title)
    plt.show()


def displaySubplots(images,titles,x):
    fig, ax = plt.subplots(1,x,sharey=True,figsize=(15,4)) #figsize=(15,4),sharey=True)#,squeeze=False)
    for i in range(x):
        try:
            ax[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        except:
            ax[i].imshow(images[i],cmap=plt.cm.gray,vmin=0,vmax=255)
        ax[i].set_title(titles[i])
    plt.show()


def displayImage(image, title):
    #Convert to RGB (default for matplotlib), since BGR is default for opencv
    try:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except:
        plt.imshow(image,cmap=plt.cm.gray,vmin=0,vmax=255)
    plt.title(title)
    plt.show()

def rgbExclusion(image, exclusionColor):
    exclusionColor = str(exclusionColor)

    only_blue, only_green,only_red = image.copy(), image.copy(),image.copy()

    only_blue[:,:,1],only_blue[:,:,2] = 0,0
    only_green[:,:,0],only_green[:,:,2]=0,0
    only_red[:,:,0],only_red[:,:,1]=0,0

    i=0
    while i==0:
        if exclusionColor.lower() in ("r","red","blue","b","green","g"):
            i=1
        else:
            i=0
        
        if exclusionColor.lower() in ("r","red"):
            no_red = image.copy()
            no_red = no_red - only_red
            displayImage(only_red,"Only Red Channel")
            displayImage(no_red,"Without Red Channel")


        elif exclusionColor.lower() in ("g","green"):
            no_green = image.copy()
            no_green = no_green - only_green
            displayImage(only_green,"Only Green Channel")
            displayImage(no_green,"Without Green Channel")
        
        elif exclusionColor.lower() in ("b","blue"):
                no_blue = image.copy()
                no_blue = no_blue - only_blue
                displayImage(only_blue,"Only Blue Channel")
                displayImage(no_blue,"Without Blue Channel")
        
        else:
            print("Invalid exclusionColor - Use R,G,B or Red, Blue, Green")
            exclusionColor = input("Please input the color: ")
            


def convolve2D(image, filter_kernel):#, padding=0, strides=1):s
    
    # Cross Correlation 
    filter_kernel = np.flipud(np.fliplr(filter_kernel))

    # Gather Shapes of filter_kernel + Image + Padding
    filter_kernel_rows = filter_kernel.shape[0]
    filter_kernel_cols = filter_kernel.shape[1]
    image_rows = image.shape[0]
    image_cols = image.shape[1]
    
    # Setting Padding and Strides
    padding = 0
    strides = 1

    # Shape of output_image Convolution
 
    output_image_rows = int(((image_rows - filter_kernel_rows + 2 * padding) / strides) + 1)
    output_image_cols = int(((image_cols - filter_kernel_cols + 2 * padding) / strides) + 1)
    output_image = np.zeros((output_image_rows, output_image_cols))
 
    # Iterate through image
    for col in range(image_cols):
        #print(y,image_cols,filter_kernel_cols)
        # Exit Convolution
        if col > image_cols - filter_kernel_cols:
            break
        for row in range(image_rows):
            try:
                output_image[row, col] = (filter_kernel * image[row: row + filter_kernel_rows, col: col + filter_kernel_cols]).sum()
            except:
                break
            

    return output_image
