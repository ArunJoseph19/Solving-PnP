# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 23:19:58 2024

@author: aarun
"""
import cv2
import numpy as np
import json
np.set_printoptions(precision=2, suppress=True)
import time

def calculate_mse(predicted_values, measured_values):
    # Ensure both lists have the same length
    assert len(predicted_values) == len(measured_values), "Lengths of predicted and measured values must be equal."
    
    # Calculate the mean squared error (MSE)
    mse = sum((p - m)**2 for p, m in zip(predicted_values, measured_values)) / len(predicted_values)
    
    return mse

start = time.time()

f = open('Images/Centre Dining Table/Dining Table Book Annotations.json')
coords = json.load(f)

frame = list(coords.keys())[0]
img = cv2.imread(frame)
print(img.shape)
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


for i in coords_list:
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
          figure_points_3D, image_points_2D, matrix_camera, distortion_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
      )

    R_mtx, jac=cv2.Rodrigues(rvec1)
    rs.append(R_mtx)
    # rs.append(rvec1)
    ts.append(tvec1)
    Rt = np.column_stack((R_mtx,tvec1))   
    Rt = np.append(Rt, np.array([0,0,0,1]))
    Rt = Rt.reshape(4,4)
    output_points[i] = Rt

predicted_values = []
measured_values = []

for i in range(len(coords_list)-1):
    X1 = np.array([0, 0, 1.0, 1.0])
    X2 = np.array([19.7, 4.5, 1.0, 1.0])
    #print('Distance between:', coords_list[i],coords_list[i+1])
    #X1_2 = np.linalg.norm(np.linalg.inv(output_points[i])@output_points[j]@X1)
    X1_2 = np.linalg.inv(output_points[coords_list[i]])@output_points[coords_list[i+1]]@X1
    predicted_values.append(X1_2)
    measured_values.append(10)
    #print(X1_2)
    
i = 'Images/Centre Dining Table/2.jpg'
j = 'Images/Centre Dining Table/3.jpg'

figure_points_3D = np.array(
    [
        (0.0, 0.0, 1),  # Left Top
        (19.7, 0.0, 1),  # Right top
        (19.7, 4.5, 1),  # Right Bottom
        (0, 4.5, 1),  # Left Bottom
    ]
)
points = []

W = np.array([0, 4.5, 1, 1])
X = output_points[i]@W
X = matrix_camera@X[:3]
X = X/X[2]
print(X)

points.append(X)


import cv2
# Function to plot points on an image
def plot_points(image, points, color=(0, 255, 0), radius=1):
    for p in points:
        print(tuple(p[:2]))
        cv2.circle(image, tuple(p[:2]), radius, color, 1)

# Create a black image
image = np.zeros((1080, 1920, 3), dtype=np.uint8)

# Plot points on the image
plot_points(image, points)

# Display the image
cv2.imshow('Image with Points', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np
from matplotlib import pyplot as plt

data = np.array([
    [1, 2],
    [2, 3],
    [3, 6],
])
x, y = data.T
plt.scatter(x,y)
plt.show()

P = np.linalg.inv(output_points[i])@output_points[j]

repro_pts = matrix_camera@X1_2[:3]
repro_pts = repro_pts/repro_pts[2]
print(repro_pts)

# p1 = repro_pts[:2]
# p2 = coords[i][0]
# np.linalg.norm(p1 - p2)

#accuracy = 1 - np.mean(np.abs(np.array(predicted_values) - np.array(measured_values)))
#mse = calculate_mse(predicted_values, measured_values)
#print("Accuracy:", accuracy)
#print('It took', time.time()-start, 'seconds.')
