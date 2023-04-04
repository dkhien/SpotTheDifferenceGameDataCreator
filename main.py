import cv2
import numpy as np
import os
from utils import *
from levelBuilder import *

# Ask for image path, check if file exists
imagePath = input("Image path: ")
while not os.path.exists(imagePath):
    imagePath = input()

# Read image and detect saliency
sourceImage = cv2.imread(imagePath)
regionsPath = detectSaliency(imagePath)

# Loop through and display each level
for level in range(3):
    modifiedImage = createLevel(regionsPath, sourceImage, level)
    # Resize the images to the same height
    height = 400
    resizedSourceImage = cv2.resize(sourceImage, (int(
        sourceImage.shape[1] * height / sourceImage.shape[0]), height))
    resizedmodifiedImage = cv2.resize(modifiedImage, (int(
        modifiedImage.shape[1] * height / modifiedImage.shape[0]), height))

    # Create a separation line
    lineThickness = 5
    separationLine = np.zeros((height, lineThickness, 3), dtype=np.uint8)

    # Concatenate the images and the separation line
    combinedImage = np.concatenate(
        (resizedSourceImage, separationLine, resizedmodifiedImage), axis=1)
    cv2.imshow(f"Level {level}", combinedImage)
    cv2.waitKey(0)
    circleDifferences(sourceImage, modifiedImage)

cv2.destroyAllWindows()
