"""Image Light Source Adjustment Program"""
"""Create by Hung-Hsiu Yen"""
"""newsted5566@gmail.com"""

import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from sys import argv
from scipy.linalg import fractional_matrix_power


#Setting Default Value
theta = 60
phi = 60
theta_slider_max = 360
phi_slider_max = 360
Save_Id = 1
title_window = 'Light Source Adjustment'
trackbar_name = 'Phi: 0-360'
trackbar_name2 = 'Theta: 0-360'
img1_path = argv[1]
img2_path = argv[2]
img3_path = argv[3]


#Function Setting
def on_trackbar(val):
    global phi
    global theta
    phi = val
    s = np.array([[math.cos(math.radians(theta))*math.sin(math.radians(phi))], [math.sin(math.radians(theta))*math.sin(math.radians(phi))], [math.cos(math.radians(phi))]])
    b1 = np.maximum(B.dot(s) ,0)
    small = np.min(b1)
    big = np.max(b1)
    new_img = np.reshape(b1,(row,col))
    new_img-=small
    new_img/=big
    cv.imshow(title_window, new_img)
    return new_img

def on_trackbar2(val):
    global phi
    global theta
    theta = val
    s = np.array([[math.cos(math.radians(theta))*math.sin(math.radians(phi))], [math.sin(math.radians(theta))*math.sin(math.radians(phi))], [math.cos(math.radians(phi))]])
    b1 = np.maximum(B.dot(s) ,0)
    small = np.min(b1)
    big = np.max(b1)
    new_img = np.reshape(b1,(row,col))
    new_img-=small
    new_img/=big
    cv.imshow(title_window, new_img)
    return new_img

def save(img):
    global Save_Id
    file_name = 'Synthesize_%d.jpg' % Save_Id
    cv.imwrite(file_name, img*255)
    Save_Id+=1


#Import 3 images of same object with different light source
img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
img3 = cv.imread(img3_path, cv.IMREAD_GRAYSCALE)


#Change the type of image matrix to double in order to do SVD
row, col =  img1.shape[0], img1.shape[1]
img1 = np.double(img1)
img2 = np.double(img2)
img3 = np.double(img3)


#Combine 3 images into one two T matrix for SVD and do SVD
img1 = np.reshape(img1, (1,-1))
img2 = np.reshape(img2, (1,-1))
img3 = np.reshape(img3, (1,-1))
t_mat = np.append(img1, img2, axis=0)
t_mat = np.append(t_mat, img3, axis = 0)
trans_t_mat = np.transpose(t_mat)
src = t_mat.dot(trans_t_mat)
w, u, vt = cv.SVDecomp(src)
w = np.diagflat(w)# let this matrix become diagonal 
w_neg_half = fractional_matrix_power(w, -0.5)


#Synthesis the radiosity of the light source
B = trans_t_mat.dot(u)
B = B.dot(w_neg_half)


#Create trackbar for adjustment
cv.namedWindow(title_window)
cv.createTrackbar(trackbar_name, title_window , 60, phi_slider_max, on_trackbar)
cv.createTrackbar(trackbar_name2, title_window , 60, theta_slider_max, on_trackbar2)


while(1):
    systhe_image = on_trackbar(phi)
    k = cv.waitKey(33)
    if k==ord('s'):#Press 's' to save picture
        save(systhe_image)
        print('Photo Saved')

    if k==27: #Press 'ESC' for exiting the program
        break
   
