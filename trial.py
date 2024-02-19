# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:34:14 2024

@author: aarun
"""
import numpy as np
import cv2 as cv
import json

def reproject(test_points,actual_img_pts,homo_mat,intrin):
    print('\n')
    for i in range(len(test_points)):
        # new_point=np.matmul(homo_mat,)
        # point_arr=np.array([[test_points[i][0]],[test_points[i][1]],[test_points[i][2]]])
        # print(point_arr)
        point_transformed = np.matmul(homo_mat, test_points[i])
        # point_transformed=point_transformed/point_transformed[2]
        inter_point = np.array([[point_transformed[0]], [point_transformed[1]], [point_transformed[2]]])
        print(inter_point)
        final_point = np.transpose(np.matmul(intrin, inter_point))
        final_point_x = final_point[0][0]/final_point[0][2]
        final_point_y = final_point[0][1]/final_point[0][2]


        accuracy_x = final_point_x / actual_img_pts[i][0] * 100
        accuracy_y = final_point_y / actual_img_pts[i][1] * 100

        print('Reprojected : ', [final_point_x, final_point_y, 1], '       ' + 'Actual :', actual_img_pts[i],
              '         Accuracy = ', (accuracy_x + accuracy_y) / 2 )
        return final_point

world_pts = np.array([
 	[0.0, 0.0, 1],
    [0.0, 19.7, 1],
    [4.5, 19.7, 1],
    [4.5, 0.0, 1]
])

fx=384.352
fy=384.352
cx=321.095
cy=240.635

intrinsic = np.array([
	[fx,0,cx],
	[0,fy,cy],
	[0,0,1]])

f = open('Images/Indoors Book Annotations.json')
coords = json.load(f)
frame = list(coords.keys())[0]
img = cv.imread(frame)
img = cv.resize(img, (1280, 720))
size = img.shape

distortion_coeffs = np.zeros((4, 1))
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
matrix_camera = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype="double",
)
intrinsic = np.array(matrix_camera)

f = open('Images/Indoors Book Annotations.json')
coords = json.load(f)

output_points = {}

intrinsic_inv = np.linalg.inv(intrinsic)

for i in list(coords.keys()):
    img_pts_prime = []
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
    homography_matrix, status = cv.findHomography(img_pts_prime, world_pts)
    homography_matrix_normalised = homography_matrix/np.linalg.norm(homography_matrix)
    
    rotation_matrix = np.zeros((3, 3))
    r1 = homography_matrix_normalised[:, 0:1]
    r2 = homography_matrix_normalised[:, 1:2]
    r3 = np.transpose(np.cross(np.transpose(r1), np.transpose(r2)))
    
    rotation_matrix[:, 0:1] = r1
    rotation_matrix[:, 1:2] = r2
    rotation_matrix[:, 2:3] = r3
    
    translation_matrix = -(np.matmul(np.linalg.inv(rotation_matrix), homography_matrix_normalised[:, 2:3]))
    print("ROTATION MATRIX :")
    print(rotation_matrix)
    print('\n')
    print("TRANSLATION MATRIX :")
    print(translation_matrix)
    print('\n')
    
    W = np.array([4.5, 19.7, 0.0, 1])
    H = np.column_stack([rotation_matrix, translation_matrix])
    H = np.vstack([H,np.array([[0., 0., 0., 1.]])])
    
    result_matrix = np.dot(H,W)
    output_points[i] = result_matrix
    
    homography_matrix_inv = np.linalg.inv(homography_matrix)
    re_pro = reproject(world_pts, img_pts, homography_matrix_inv, intrinsic)
    re_pro = np.dot(intrinsic_inv, np.transpose(re_pro))
    re_pro[0] = re_pro[0]/re_pro[2]
    re_pro[1] = re_pro[1]/re_pro[2]
    re_pro[2] = re_pro[2]/re_pro[2]
    
    output_points[i] = re_pro
    #H = np.hstack([homography_matrix_inv, translation_matrix])
    #H = np.vstack([H,np.array([[0., 0., 0., 1.]])])
    #output_points[i] = np.dot(H, W)

i = 'Images/1_B.jpg'
j = 'Images/2_B.jpg'
k = 'Images/3_B.jpg'
#1_B and 2_B
print(np.linalg.norm(output_points[i] - output_points[j]))
#2_B and 3_B
print(np.linalg.norm(output_points[j] - output_points[k]))
#1_B and 3_B
print(np.linalg.norm(output_points[i] - output_points[k]))    


