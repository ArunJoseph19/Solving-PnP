# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 23:23:17 2024

@author: aarun
"""
import cv2
import numpy as np
import json
from scipy.linalg import sqrtm, inv

np.set_printoptions(precision=2, suppress=False)

#f = open('Testing Annotations.json')
f = open('Underpass IISc/Selected Frames/Underpass Videos Annotations.json')
coords = json.load(f)

def plot_points(key):
    frame = key
    img = cv2.imread(frame)
    img = cv2.resize(img, (1280, 720))
    points = coords[frame]
    
    for point in points:
        cv2.circle(img, point, 2, (0, 255, 0), -1)
    
    # Display the image
    cv2.imshow("Image with Points", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sym(w):
    return w.dot(inv(sqrtm(w.T.dot(w))))

#plot_points('Underpass IISc/Selected Frames/Black Maruti Baleno Up (1).jpg')

#in cms
# figure_points_3D = np.array(
#     [
#         (0.0, 0.0, 1),  # Left Top
#         (30.0, 0.0, 1),  # Right top
#         (30.0, 21.0, 1),  # Right Bottom
#         (0, 21.0, 1),  # Left Bottom
#     ]
# )

figure_points_3D = np.array(
    [
        (0.0, 0.0, 1),  # Left Top
        (50.0, 0.0, 1),  # Right top
        (50.0, 12.0, 1),  # Right Bottom
        (0, 12.0, 1),  # Left Bottom
    ]
)

matrix_camera = np.array([[2998.35240915/3.2,    0.        ,  619.25831477],
       [   0.        , 2939.12399854/3.2,  359.27498341],
       [   0.        ,    0.        ,    1.        ]])

matrix_camera = matrix_camera
distortion_coeffs = np.zeros((4, 1))

output_points = {}
rs = {}
ts = {}

#coords_list = ['Testing/1 Slowish Drive (1).jpg', 'Testing/1 Slowish Drive (2).jpg', 'Testing/2 Fast Boi Drive (1).jpg', 'Testing/2 Fast Boi Drive (2).jpg', 'Testing/3 Mid Drive (1).jpg', 'Testing/3 Mid Drive (2).jpg', 'Testing/4 Slowish Drive (1).jpg', 'Testing/4 Slowish Drive (2).jpg', 'Testing/HSRP (1).jpg', 'Testing/HSRP (2).jpg']
coords_list = list(coords.keys())

for i in coords_list:
    img = cv2.imread(i)
    img = cv2.resize(img, (1280, 720))
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
          figure_points_3D, image_points_2D, matrix_camera, distortion_coeffs#, flags=cv2.SOLVEPNP_ITERATIVE
      )

    R_mtx, jac=cv2.Rodrigues(rvec1)
    #R_mtx = sym(R_mtx)
    rs[i] = R_mtx
    # rs.append(rvec1)
    ts[i] = tvec1
    Rt = np.column_stack((R_mtx,tvec1))   
    Rt = np.append(Rt, np.array([0,0,0,1]))
    Rt = Rt.reshape(4,4)
    output_points[i] = Rt

a,b = 8,9
print(coords_list[a][31:],coords_list[b][31:])
X1_2 = np.linalg.inv(output_points[coords_list[b]])@output_points[coords_list[a]]
print(np.linalg.norm(X1_2[:, 3]))
print(ts[coords_list[a]][2] - ts[coords_list[b]][2])
plot_points(coords_list[a])
plot_points(coords_list[b])
