# Remove object
import numpy as np
import cv2

# Load the image and template
img = cv2.resize(cv2.imread('assets/starrynight.jpg', 1), (0, 0), fx=0.8, fy=0.8)
template = cv2.resize(cv2.imread('assets/house2.jpg', 1), (0, 0), fx=0.8, fy=0.8)

# Find the location of the template in the image
h, w, _ = template.shape
method = cv2.TM_CCORR_NORMED
img2 = img.copy()
result = cv2.matchTemplate(img2, template, method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    location = min_loc
else:
    location = max_loc
bottom_right = (location[0] + w, location[1] + h)    
cv2.rectangle(img2, location, bottom_right, (255, 255, 255), 5)

# Inpaint the template region using surrounding pixels
mask = np.zeros(img.shape[:2], np.uint8)
mask[location[1]:location[1]+h, location[0]:location[0]+w] = 255
dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

cv2.imshow('Original', img)
cv2.imshow('Removed Template', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
