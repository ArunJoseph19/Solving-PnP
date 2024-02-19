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