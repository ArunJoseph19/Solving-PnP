# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:55:01 2024

@author: aarun
"""
import cv2
import numpy as np
import json

f = open('Images/Centre Dining Table/Dining Table Book Annotations.json')
coords = json.load(f)

frame = list(coords.keys())[0]
img = cv2.imread(frame)
img = cv2.resize(img, (1280, 720))
size = img.shape

output_points = {}

#in cms
figure_points_3D = np.array(
    [
        (0.0, 0.0, 1),  # Left Top
        (19.7, 0.0, 1),  # Right top
        (19.7, 4.5, 1),  # Right Bottom
        (0, 4.5, 1),  # Left Bottom
    ]
)
distortion_coeffs = np.zeros((4, 1))
#1280 x 720 Matrix
matrix_camera = np.array([[2998.35240915,    0.        ,  619.25831477],
       [   0.        , 2939.12399854,  359.27498341],
       [   0.        ,    0.        ,    1.        ]])
matrix_camera = np.array(matrix_camera)

for i in list(coords.keys()):
    #i = 'Images/Centre Dining Table/2.jpg'
    img = cv2.imread(i)
    img = cv2.resize(img, (1280, 720))
    size = img.shape
    
    image_points_2D = np.array(
        [
            tuple(coords[i][0]),  # Left Top
            tuple(coords[i][1]),  # Right Top
            tuple(coords[i][2]),  # Right Bottom
            tuple(coords[i][3]),  # Left Bottom
        ],
        dtype="double",
    )

    ret, rvec1, tvec1 = cv2.solvePnP(
        figure_points_3D, image_points_2D, matrix_camera, distortion_coeffs, flags=cv2.SOLVEPNP_IPPE
    )
    #print(tvec1)
    R_mtx, jac=cv2.Rodrigues(rvec1)
    Rt=np.column_stack((R_mtx,tvec1))
    #print(Rt.shape)
    P_mtx= matrix_camera.dot(Rt)
    
    W = np.array([0.0, 0.0, 0, 1.0]) #World Point we wanna find the coords for
    #CamMtx*R|t - Projection Matrix * W
    #This below variable will give the point as in image_points_2D
    Image_Point_OG = P_mtx.dot(W)
    Image_Point_OG = Image_Point_OG/Image_Point_OG[2]
    print("Original Value: {}".format(image_points_2D[0]))
    print("Recalculated: {}".format(Image_Point_OG[:2]))

    #R|t - Extrinsic Matrix * W
    Rt = np.append(Rt, np.array([0,0,0,1]))
    Rt = Rt.reshape(4,4)
    result_matrix = np.dot(Rt,W)
    #W = np.array([4.5, 19.7, 0.0])
    #result_matrix = np.dot(R_mtx, W)
    
    output_points[i] = result_matrix
    print('\n')
    
    
#R2 * (-R1.t()*tvec1) + tvec2 
#tvec1_2 = output_points[j][0] * (-output_points[i][0].T * output_points[i][1] ) + output_points[j][1]

print(W)
i = 'Images/Centre Dining Table/2.jpg'
j = 'Images/Centre Dining Table/3.jpg'
k = 'Images/Centre Dining Table/4.jpg'
l = 'Images/Centre Dining Table/5.jpg'
m = 'Images/Centre Dining Table/6.jpg'
print(np.linalg.norm(output_points[i] - output_points[j]))
print(np.linalg.norm(output_points[j] - output_points[k]))
print(np.linalg.norm(output_points[k] - output_points[l]))
print(np.linalg.norm(output_points[l] - output_points[m]))
print(np.linalg.norm(output_points[i] - output_points[m]))    

'''
Individually
ret, rvec1, tvec1 = cv2.solvePnP(
    figure_points_3D, image_points_2D, matrix_camera, distortion_coeffs, flags=0
)

print("pnp rvec1 - Rotation")
print(rvec1)

print("pnp tvec1 - Translation")
print(tvec1)

print("R - rodrigues vecs")
R_mtx, jac=cv2.Rodrigues(rvec1)
print(R_mtx)

print("R|t - Extrinsic Matrix")
Rt=np.column_stack((R_mtx,tvec1))
print(Rt)

print("newCamMtx*R|t - Projection Matrix")
P_mtx= matrix_camera.dot(Rt)
print(P_mtx)
'''
#distance = (24  * 596 / 596 * pixel size (in um))



