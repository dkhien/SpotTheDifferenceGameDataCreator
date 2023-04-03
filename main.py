import cv2
import imutils
import numpy as np
import random
import os
from utils import *
from levelBuilder import *
imagePath = "assets/avengers.png"

def run():
    # print("Image path: ")
    # imagePath = input()
    # while not os.path.exists(imagePath):
    #     imagePath = input()
    sourceImage = cv2.imread(imagePath)
    regionsPath = detectSaliency(imagePath)
    for level in range(3):
        modifiedImage = createLevel(regionsPath, sourceImage, level)
        # Resize the images to the same height
        height = 400
        resizedSourceImage = cv2.resize(sourceImage, (int(sourceImage.shape[1] * height / sourceImage.shape[0]), height))
        resizedmodifiedImage = cv2.resize(modifiedImage, (int(modifiedImage.shape[1] * height / modifiedImage.shape[0]), height))

        # Create a black separation line
        lineThickness = 5
        separationLine = np.zeros((height, lineThickness, 3), dtype=np.uint8)

        # Concatenate the images and the separation line horizontally
        combinedImage = np.concatenate((resizedSourceImage, separationLine, resizedmodifiedImage), axis=1)
        cv2.imshow(f"Level {level}", combinedImage)
        cv2.waitKey(0)
        circleDifferences(sourceImage, modifiedImage)
        # cv2.waitKey(0)

    cv2.destroyAllWindows()

run()

