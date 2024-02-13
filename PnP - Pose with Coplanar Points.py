# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:54:09 2024

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

camera_points = {}

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

output_points = {}

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
    
    #start of Tajmul's code
    homography_matrix, status = cv2.findHomography(image_points_2D, figure_points_3D) #he homography is a 3×3 matrix that maps the points in one point to the corresponding point in another image
    homography_matrix_normalised = homography_matrix/np.linalg.norm(homography_matrix) #Normalised for some reason
    
    rotation_matrix = np.zeros((3, 3))
    r1 = homography_matrix_normalised[:, 0:1]
    r2 = homography_matrix_normalised[:, 1:2]
    r3 = np.transpose(np.cross(np.transpose(r1), np.transpose(r2)))
    
    rotation_matrix[:, 0:1] = r1
    rotation_matrix[:, 1:2] = r2
    rotation_matrix[:, 2:3] = r3
    
    translation_matrix = -(np.matmul(np.linalg.inv(rotation_matrix), homography_matrix_normalised[:, 2:3]))
    #End of Tajmul's code
        
    #Doubt here to be clarified
    H = np.hstack([rotation_matrix, translation_matrix])
    H = np.vstack([H,np.array([[0., 0., 0., 1.]])])
    
    W = np.array([4.5, 19.7, 0.0, 1]) #World Point we wanna find the coords for
    
    result_matrix = np.dot(H,W)
    
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
