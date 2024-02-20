# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 23:19:58 2024

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

matrix_camera = np.array([[2998.35240915/3.2,    0.        ,  619.25831477],
       [   0.        , 2939.12399854/3.2,  359.27498341],
       [   0.        ,    0.        ,    1.        ]])

matrix_camera = matrix_camera
distortion_coeffs = np.zeros((4, 1))

output_points = {}
homographies = []
rs = []
ts = []
hs = []

coords_list = ['Images/Centre Dining Table/2.jpg', 'Images/Centre Dining Table/3.jpg', 'Images/Centre Dining Table/4.jpg', 'Images/Centre Dining Table/5.jpg', 'Images/Centre Dining Table/6.jpg']


#for i in coords_list:
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
    # for j in coords[i]:
    #     j.append(1)
    image_points_2D = np.array(
        [
            tuple(coords[i][0]),  # Left Top
            tuple(coords[i][1]),  # Right Top
            tuple(coords[i][2]),  # Right Bottom
            tuple(coords[i][3]),  # Left Bottom
        ],
        dtype="double",
    )
 
    intrinsic_inv = np.linalg.inv(matrix_camera)
    img_pts_prime = []

    for point in image_points_2D:
        # new_point = np.matmul(intrinsic_inv, point)
        img_pts_prime.append(point)

    img_pts_prime = np.array(img_pts_prime)

    ret, rvec1, tvec1 = cv2.solvePnP(
          figure_points_3D, img_pts_prime, matrix_camera, distortion_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
      )

    R_mtx, jac=cv2.Rodrigues(rvec1)
    rs.append(R_mtx)
    # rs.append(rvec1)
    ts.append(tvec1)
    # Rt = np.vstack((Rt, np.array([0,0,0,1])))

#R2 * (-R1.t()*tvec1) + tvec2 
#tvec1_2 = output_points[j][0] * (-output_points[i][0].T * output_points[i][1] ) + output_points[j][1]

i = 'Images/Centre Dining Table/2.jpg' #0
j = 'Images/Centre Dining Table/3.jpg' #1
k = 'Images/Centre Dining Table/4.jpg' #2
l = 'Images/Centre Dining Table/5.jpg' #3
m = 'Images/Centre Dining Table/6.jpg' #4


for i in range(0,4):
    # print(hs[i])
    print(   np.linalg.inv(rs[i])@(-rs[i+1]@ts[i+1]) + ts[i])
    # print(np.matmul(rs[i], (-np.matmul(rs[i+1], ts[i+1])))) + ts[i]

# index = 2
# corner = 0
# coords[coords_list[index]][corner].append(1)
# coords_vector = np.array(np.matmul(np.linalg.inv(matrix_camera),coords[coords_list[index]][corner]))
# wp = np.matmul(np.linalg.inv(rs[index]), coords_vector.T) + ts[index].T
# print(wp)