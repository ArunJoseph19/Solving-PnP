# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:55:01 2024

@author: aarun
"""
import cv2
import numpy as np
import json

f = open('Images/Indoors Book Annotations.json')
coords = json.load(f)

frame = list(coords.keys())[0]
img = cv2.imread(frame)
img = cv2.resize(img, (1280, 720))
size = img.shape

output_points = {}

#in cms
figure_points_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # Left Top
        (0.0, 19.7, 0.0),  # Right top
        (4.5, 19.7, 0.0),  # Right Bottom
        (4.5, 0.0, 0.0),  # Left Bottom
    ]
)
distortion_coeffs = np.zeros((4, 1))
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
matrix_camera = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype="double",
)
matrix_camera = np.array(matrix_camera)

for i in list(coords.keys()):
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
    print(i)
    ret, rvec1, tvec1 = cv2.solvePnP(
        figure_points_3D, image_points_2D, matrix_camera, distortion_coeffs, flags=0
    )

    R_mtx, jac=cv2.Rodrigues(rvec1)
    Rt=np.column_stack((R_mtx,tvec1))
    P_mtx= matrix_camera.dot(Rt)

    W = np.array([4.5, 19.7, 0.0, 1]) #World Point we wanna find the coords for
    #CamMtx*R|t - Projection Matrix * W
    #This below variable will give the point as in image_points_2D
    Image_Point_OG = P_mtx.dot(W)
    Image_Point_OG = Image_Point_OG/Image_Point_OG[2]

    #R|t - Extrinsic Matrix * W
    Rt = np.append(Rt, np.array([0,0,0,1]))
    Rt = Rt.reshape(4,4)
    result_matrix = np.dot(Rt,W)
    
    output_points[i] = result_matrix

i = 'Images/1_B.jpg'
j = 'Images/2_B.jpg'
k = 'Images/3_B.jpg'
#1_B and 2_B
print(np.linalg.norm(output_points[i] - output_points[j]))
#2_B and 3_B
print(np.linalg.norm(output_points[j] - output_points[k]))
#1_B and 3_B
print(np.linalg.norm(output_points[i] - output_points[k]))    

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


