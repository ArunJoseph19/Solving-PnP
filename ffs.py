# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:24:33 2024

@author: aarun
"""
import numpy as np
import cv2 as cv
import json
np.set_printoptions(suppress=True)

world_pts = np.array(
    [
        (0.0, 0.0, 1),  # Left Top
        (19.7, 0.0, 1),  # Right top
        (19.7, 4.5, 1),  # Right Bottom
        (0, 4.5, 1),  # Left Bottom
    ]
)

f = open('Images/Indoors Book Annotations.json')
coords = json.load(f)
frame = list(coords.keys())[0]
img = cv.imread(frame)
img = cv.resize(img, (1280, 720))
size = img.shape

matrix_camera = [[2996.398252  ,    0.        , 2015.94881616],
       [   0.        , 2996.77143007, 1137.12998586],
       [   0.        ,    0.        ,    1.        ]]
matrix_camera = np.array(matrix_camera)
intrinsic = np.array(matrix_camera)
intrinsic_inv = np.linalg.inv(intrinsic)
img_pts_prime = []
i = 'Images/3_B.jpg'
img = cv.imread(i)
img = cv.resize(img, (1280, 720))
for j in coords[i]:
    j.append(1)
img_pts = np.array([
    coords[i][0],
    coords[i][1],
    coords[i][2],
    coords[i][3]
])
    
for point in img_pts:
  new_point = np.matmul(intrinsic_inv, point)
  img_pts_prime.append(new_point)

img_pts_prime = np.array(img_pts_prime)
# print(img_pts_prime)
homography_matrix, status = cv.findHomography(world_pts, img_pts)

W = world_pts[2]
result  = np.dot(homography_matrix, W)
print(result)
print(result/result[2])
print(img_pts[2])

'''
solutions = cv.decomposeHomographyMat(homography_matrix, intrinsic)

j = 3
Rt=np.column_stack((solutions[1][j],solutions[2][j]))
W = world_pts[2]
print(np.dot(homography_matrix, W))


print(Rt.shape)
P_mtx= intrinsic.dot(Rt)
#print(P_mtx.shape)

W = np.array([0, 0, 0.0, 1])
Image_Point_OG = P_mtx.dot(W)
Image_Point_OG = Image_Point_OG/Image_Point_OG[2]
print(Image_Point_OG)
print(img_pts[0])
'''
import cv2 as cv
import numpy as np
output_points = {}
#Validation 45 degrees, x = 10, y = 5
img_pts = np.array(
    [
        (10, 5, 1),  # Left Top
        (17.07,12.07 , 1),  # Right top
        (10, 19.14, 1),  # Right Bottom
        (2.93, 12.07, 1),  # Left Bottom
    ]
)

def rotate_translate_square(cx, cy,angle_deg, tx, ty):
    # Convert the angle from degrees to radians
    angle_rad = np.radians(angle_deg)
    
    # Define the vertices of the square
    vertices = np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
    ])
    
    # Rotate the vertices
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    rotated_vertices = np.dot(vertices, rotation_matrix.T)
    
    # Translate the rotated vertices
    translated_vertices = rotated_vertices + np.array([cx, cy])
    
    # Translate the vertices by (tx, ty)
    translated_vertices += np.array([tx, ty])
        
    return translated_vertices

world_pts = np.array(
    [
        (0.0, 0.0, 1),  
        (10, 0, 1),  
        (10, 10, 1),  
        (0, 10, 1),  
    ]
)

cx = 0 
cy = 0  
angle_deg = 30
tx = 0
ty = 20

img_pts = rotate_translate_square(cx, cy, angle_deg, tx, ty).tolist()
for j in img_pts:
    j.append(1)
img_pts = np.array(img_pts)

intrinsic = [[2996.398252  ,    0.        , 2015.94881616],
       [   0.        , 2996.77143007, 1137.12998586],
       [   0.        ,    0.        ,    1.        ]]
intrinsic = np.array(intrinsic)
intrinsic_inv = np.linalg.inv(intrinsic)
img_pts_prime = []

for point in img_pts:
  new_point = np.matmul(intrinsic_inv, point)
  img_pts_prime.append(new_point)
img_pts_prime = np.array(img_pts_prime)

homography_matrix, status = cv.findHomography(world_pts,img_pts_prime)

W = np.array([10, 10, 1])

output_points[str(tx)+str(ty)] = np.dot(homography_matrix,W)

i = '105'
j = '300'
print(np.linalg.norm(output_points[i] - output_points[j]))

for i in list(output_points.keys()):
    for j in list(output_points.keys()):
        print(i, j)
        print(np.linalg.norm(output_points[i] - output_points[j]))

homography_matrix_normalised = homography_matrix/np.linalg.norm(homography_matrix) #Normalised for some reason

rotation_matrix = np.zeros((3, 3))
r1 = homography_matrix_normalised[:, 0:1]
r2 = homography_matrix_normalised[:, 1:2]
r3 = np.transpose(np.cross(np.transpose(r1), np.transpose(r2)))

rotation_matrix[:, 0:1] = r1
rotation_matrix[:, 1:2] = r2
rotation_matrix[:, 2:3] = r3

translation_matrix = -(np.matmul(np.linalg.inv(rotation_matrix), homography_matrix_normalised[:, 2:3]))
print(rotation_matrix, translation_matrix)




