import cv2
import numpy as np
# Load the image
img = cv2.imread('assets/avengers.png')

# Initialize the objectness saliency detector
objectness = cv2.ximgproc.createStructuredEdgeDetection('objectness')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform the objectness saliency detection
objectness_map = objectness.detectEdges(np.float32(gray))

# Normalize the objectness saliency map
objectness_map = cv2.normalize(objectness_map, None, 0, 255, cv2.NORM_MINMAX)

# Convert the objectness saliency map to uint8 datatype
objectness_map = np.uint8(objectness_map)

# Display the objectness saliency map
cv2.imshow('Objectness Salience Map', objectness_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
