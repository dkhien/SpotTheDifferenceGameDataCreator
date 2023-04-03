import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

# Load the image
img = cv2.imread('assets/avengers.png')
img2 = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

hist = cv2.calcHist([img],[0],None,[256],[0,256]) 


# Find the bin with the maximum value in the histogram
max_bin = np.argmax(hist)
print(max_bin)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)
print(max_val)
while True:
    color = np.array([np.random.randint(0, 256)], dtype=np.uint8)
    if (hist[color] != 0):
        break

# Set a limit on the number of regions to change
max_changes = 5
num_changes = 0

# Loop until the maximum number of changes has been reached
while num_changes < max_changes:
    # Find a random color within a certain range of the selected color
    colorRange = random.randint(20, 40)
    mask = cv2.inRange(img, color - colorRange, color + colorRange)
    
    # If the mask contains any pixels, change their color
    if np.count_nonzero(mask) > 0:
        # Generate a random color to replace the selected color
        new_color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)

        # Replace the selected color with the new color
        img2[mask > 0] = new_color
        num_changes += 1

# Show the modified image
cv2.imshow("Modified", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
