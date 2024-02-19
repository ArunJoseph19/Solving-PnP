# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:54:09 2024

@author: aarun
"""
import cv2
import numpy as np
import json
np.set_printoptions(suppress=True)
f = open('Images/Centre Dining Table/Dining Table Book Annotations.json')
coords = json.load(f)

frame = list(coords.keys())[0]
img = cv2.imread(frame)
img = cv2.resize(img, (1280, 720))
size = img.shape

camera_points = {}

#in cms
figure_points_3D = np.array(
    [
        (0.0, 0.0, 1),  # Left Top
        (19.7, 0.0, 1),  # Right top
        (19.7, 4.5, 1),  # Right Bottom
        (0, 4.5, 1),  # Left Bottom
    ]
)

matrix_camera = np.array([[2998.35240915,    0.        ,  619.25831477],
       [   0.        , 2939.12399854,  359.27498341],
       [   0.        ,    0.        ,    1.        ]])


output_points = {}

for i in list(coords.keys()):
    #i = 'Images/Centre Dining Table/4.jpg'
    img = cv2.imread(i)
    img = cv2.resize(img, (1280, 720))
    size = img.shape
    image_points_2D_2 = np.array(
        [
            tuple(coords[i][0]),  # Left Top
            tuple(coords[i][1]),  # Right Top
            tuple(coords[i][2]),  # Right Bottom
            tuple(coords[i][3]),  # Left Bottom
        ],
        dtype="double",
    )
    for j in coords[i]:
        j.append(1)
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
    intrinsic_inv = np.linalg.inv(matrix_camera)
    img_pts_prime = []

    for point in image_points_2D:
        new_point = np.matmul(intrinsic_inv, point)
        img_pts_prime.append(new_point)

    img_pts_prime = np.array(img_pts_prime)
    #start of Tajmul's code
    homography_matrix, status = cv2.findHomography(figure_points_3D, img_pts_prime) #he homography is a 3×3 matrix that maps the points in one point to the corresponding point in another image
    homography_matrix_normalised = homography_matrix/np.linalg.norm(homography_matrix[:,0])
    
    #COPLANAR METHOD
    r_coplanar = np.zeros((3, 3))
    r1 = homography_matrix_normalised[:, 0:1]
    r2 = homography_matrix_normalised[:, 1:2]
    r3 = np.transpose(np.cross(np.transpose(r1), np.transpose(r2)))
    
    r_coplanar[:, 0:1] = r1
    r_coplanar[:, 1:2] = r2
    r_coplanar[:, 2:3] = r3
    
    #t_coplanar = -(np.matmul(np.linalg.inv(r_coplanar), homography_matrix_normalised[:, 2:3]))
    t_coplanar = homography_matrix_normalised[2]
    
    #Sanity Check
    print(np.dot(r1.flatten(),r2.flatten()))
    print(np.linalg.det(r_coplanar))
    print(t_coplanar)
    
    #ORTHONORMALIZATION
    r_ortho = np.zeros((3, 3))
    qr = np.linalg.qr(r_coplanar[:, 0:2])
    #qr = np.linalg.qr(r_coplanar)
    r_ortho[0] = qr[0][:, 0]
    r_ortho[1] = qr[0][:, 1]
    r_ortho[2] = np.transpose(np.cross(np.transpose(r_ortho[0]), np.transpose(r_ortho[1])))
    
    #t_ortho = -(np.matmul(np.linalg.inv(r_ortho), homography_matrix_normalised[:, 2:3]))
    t_ortho = homography_matrix_normalised[2]
    t_ortho = t_ortho.reshape(-1, 1)
    
    #Sanity Check
    print(np.dot(r_ortho[0], r_ortho[1]))
    print(np.linalg.det(r_ortho))
    print(t_ortho)

    #SVD Decomposition    
    #r_SVD = np.zeros((3, 3))
    #SVD_W, SVD_U, SVD_Vt = cv2.SVDecomp(r_coplanar)
    #R = np.dot(SVD_U,SVD_Vt)
    #r_SVD[0] = R[:, 0]
    #r_SVD[1] = R[:, 1]
    #r_SVD[2] = np.transpose(np.cross(np.transpose(r_SVD[0]), np.transpose(r_SVD[1])))
    
    #t_SVD = -(np.matmul(np.linalg.inv(r_SVD), homography_matrix_normalised[:, 2:3]))
    
    #Sanity Check
    #print(np.linalg.det(r_SVD))
    #print(np.dot(r_SVD[0], r_SVD[1]))
    #print(t_SVD)

    W = np.array([4.5, 19.7, 0, 1]) #World Point we wanna find the coords for
    rotation_matrix, translation_matrix = r_coplanar, t_coplanar.reshape(-1, 1)
    
    H = np.hstack([rotation_matrix, translation_matrix])
    H = np.vstack([H,np.array([[0., 0., 0., 1.]])])
    
    result_matrix = np.dot(H, W)
    
    output_points[i] = result_matrix
    #print(result_matrix)

#R2 * (-R1.t()*tvec1) + tvec2 
#tvec1_2 = output_points[j][0] * (-output_points[i][0].T * output_points[i][1] ) + output_points[j][1]

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
   
