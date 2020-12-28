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
    b_chan = np.maximum(B_b.dot(s) ,0)
    g_chan = np.maximum(B_g.dot(s), 0)
    r_chan = np.maximum(B_r.dot(s), 0)    
    res =  np.zeros((row, col, 3))
    res[:,:,0] = np.reshape(b_chan, (row, col))
    res[:,:,1] = np.reshape(g_chan, (row, col))
    res[:,:,2] = np.reshape(r_chan, (row, col))
    max_b, max_g, max_r = np.max(res[:,:,0]), np.max(res[:,:,1]), np.max(res[:,:,2])
    res[:,:,0]/=max_b
    res[:,:,1]/=max_g
    res[:,:,2]/=max_r
    cv.imshow(title_window, res)
    return res

def on_trackbar2(val):
    global phi
    global theta
    theta = val
    s = np.array([[math.cos(math.radians(theta))*math.sin(math.radians(phi))], [math.sin(math.radians(theta))*math.sin(math.radians(phi))], [math.cos(math.radians(phi))]])
    b_chan = np.maximum(B_b.dot(s) ,0)
    g_chan = np.maximum(B_g.dot(s), 0)
    r_chan = np.maximum(B_r.dot(s), 0)
    res =  np.zeros((row, col, 3))
    res[:,:,0] = np.reshape(b_chan, (row, col))
    res[:,:,1] = np.reshape(g_chan, (row, col))
    res[:,:,2] = np.reshape(r_chan, (row, col))
    max_b, max_g, max_r = np.max(res[:,:,0]), np.max(res[:,:,1]), np.max(res[:,:,2])
    res[:,:,0]/=max_b
    res[:,:,1]/=max_g
    res[:,:,2]/=max_r
    cv.imshow(title_window, res)
    return res

def save(img):
    global Save_Id
    file_name = 'Synthesize_%d.jpg' % Save_Id
    cv.imwrite(file_name, img*255)
    Save_Id+=1


#Import 3 images of same object with different light source
img1 = cv.imread(img1_path)
img2 = cv.imread(img2_path)
img3 = cv.imread(img3_path)


#Change the type of image matrix to double in order to do SVD
row, col =  img1.shape[0], img1.shape[1]
img1 = np.double(img1)
img2 = np.double(img2)
img3 = np.double(img3)



#Decompose image into RGB seperately
img1_b = img1[:,:,0]
img1_g = img1[:,:,1]
img1_r = img1[:,:,2]
img2_b = img2[:,:,0]
img2_g = img2[:,:,1]
img2_r = img2[:,:,2]
img3_b = img3[:,:,0]
img3_g = img3[:,:,1]
img3_r = img3[:,:,2]



#Combine 3 images into one two T matrix for SVD and do SVD
img1＿b = np.reshape(img1_b, (1,-1))
img1＿g = np.reshape(img1_g, (1,-1))
img1＿r = np.reshape(img1_r, (1,-1))

img2＿b = np.reshape(img2_b, (1,-1))
img2＿g = np.reshape(img2_g, (1,-1))
img2＿r = np.reshape(img2_r, (1,-1))

img3＿b = np.reshape(img3_b, (1,-1))
img3＿g = np.reshape(img3_g, (1,-1))
img3＿r = np.reshape(img3_r, (1,-1))

t_mat_b = np.append(img1_b, img2_b, axis=0)
t_mat_b = np.append(t_mat_b, img3_b, axis = 0)

t_mat_g = np.append(img1_g, img2_g, axis=0)
t_mat_g = np.append(t_mat_g, img3_g, axis = 0)

t_mat_r = np.append(img1_r, img2_r, axis=0)
t_mat_r = np.append(t_mat_r, img3_r, axis = 0)

trans_t_mat_b = np.transpose(t_mat_b)
src_b = t_mat_b.dot(trans_t_mat_b)
w_b, u_b, vt_b = cv.SVDecomp(src_b)
w_b = np.diagflat(w_b)# let this matrix become diagonal 
w_neg_half_b = fractional_matrix_power(w_b, -0.5)

trans_t_mat_g = np.transpose(t_mat_g)
src_g = t_mat_g.dot(trans_t_mat_g)
w_g, u_g, vt_g = cv.SVDecomp(src_g)
w_g = np.diagflat(w_g)# let this matrix become diagonal 
w_neg_half_g = fractional_matrix_power(w_g, -0.5)

trans_t_mat_r = np.transpose(t_mat_r)
src_r = t_mat_r.dot(trans_t_mat_r)
w_r, u_r, vt_r = cv.SVDecomp(src_r)
w_r = np.diagflat(w_r)# let this matrix become diagonal 
w_neg_half_r = fractional_matrix_power(w_r, -0.5)



#Synthesis the radiosity of the light source
B_b = trans_t_mat_b.dot(u_b)
B_b = B_b.dot(w_neg_half_b)

B_g = trans_t_mat_g.dot(u_g)
B_g = B_g.dot(w_neg_half_g)

B_r = trans_t_mat_r.dot(u_r)
B_r = B_r.dot(w_neg_half_r)




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
   
