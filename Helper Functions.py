# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 13:13:15 2024

@author: aarun
"""
import cv2
import os
filepath = 'G:/My Drive/Bangalore Traffic Police/BTP Cams Clips Local Copy/325.mp4'
vidcap = cv2.VideoCapture(filepath)

fps = vidcap.get(cv2.CAP_PROP_FPS)
print('frames per second =',fps)

os.mkdir(filepath[:-4])
success,image = vidcap.read()

count = 0
while success:
  cv2.imwrite(filepath[:-4]+"/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1

#Get Pixel coordinates from an Image
import cv2
def click_event(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')      
      #cv2.putText(img, f'({x},{y})',(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      coords.append((x,y))
      cv2.circle(img, (x,y), 3, (0,255,255), -1)

coords_dictionary = {}

coords = []  

filepath = 'Images/Centre Dining Table/6.jpg'
img = cv2.imread(filepath)
cv2.namedWindow('Point Coordinates')
cv2.setMouseCallback('Point Coordinates', click_event)

while True:
   img = cv2.resize(img, (1280, 720))
   cv2.imshow('Point Coordinates',img)
   k = cv2.waitKey(1) & 0xFF
   if k == 27:
      break
cv2.destroyAllWindows()

coords_dictionary[filepath] = coords

for i in list(coords_dictionary.keys()):
    print(coords_dictionary[i][3])

import json
with open("Dining Table Book Annotations.json", "w") as outfile: 
    json.dump(coords_dictionary, outfile)

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
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
i = 'Images/Centre Dining Table/2.jpg'
img = cv2.imread(i)
img = cv2.resize(img, (1280, 720))

while True:
   img = cv2.resize(img, (1280, 720))
   cv2.imshow('Point Coordinates',img)
   k = cv2.waitKey(1) & 0xFF
   if k == 27:
      break
cv2.destroyAllWindows()

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

#World Coordinates
X = np.array([0, 19.7, 19.7, 0])
Y = np.array([0, 0, 4.5, 4.5])

#Image Coordinates
u = np.array([614, 1206, 1197, 616])
v = np.array([355, 351, 476, 487])

plt.scatter(u,v,s=100, facecolors='none', edgecolors='y')
plt.imshow(img)
plt.show()

A = np.zeros((8,9))
for i in range(0,4):
    A[2*i,:] = [0, 0, 0, -X[i], -Y[i], -1, v[i]*X[i], v[i]*Y[i], v[i]]
    A[2*i+1, :] = [X[i], Y[i], 1, 0, 0, 0, -u[i]*X[i], -u[i]*Y[i], -u[i]]

L, V = np.linalg.eig(A.T@A)
h = V[:,-1]
H = np.reshape(h, (3,3))
H = np.linalg.inv(H)

homography_matrix, status = cv2.findHomography(figure_points_3D, image_points_2D_2)

im_warp = cv2.warpPerspective(img, homography_matrix, ((img.shape[1]*2, img.shape[0])))

while True:
   cv2.imshow('Point Coordinates',im_warp)
   k = cv2.waitKey(1) & 0xFF
   if k == 27:
      break
cv2.destroyAllWindows()
cv2.imwrite('warp.jpg', im_warp)
