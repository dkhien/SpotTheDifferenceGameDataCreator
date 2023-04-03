import numpy as np
import cv2

# Load the image and template
img = cv2.resize(cv2.imread('assets/starrynight.jpg', 1), (0, 0), fx=0.8, fy=0.8)
template = cv2.resize(cv2.imread('assets/startoadd.jpg', 1), (0, 0), fx=0.8, fy=0.8)

# Find the location to place the template in the image
h, w, _ = template.shape
location = (100, 100)  # specify the location to place the template
bottom_right = (location[0] + w, location[1] + h)
cv2.rectangle(img, location, bottom_right, (255, 255, 255), 5)

# Blend the template region with the image
alpha = 0.1
roi = img[location[1]:location[1]+h, location[0]:location[0]+w]
blended_roi = cv2.addWeighted(roi, alpha, template, 1-alpha, 0)
img[location[1]:location[1]+h, location[0]:location[0]+w] = blended_roi

cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
