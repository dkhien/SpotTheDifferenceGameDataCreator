import cv2
import numpy as np
import imutils
import random
import os

def detectSaliency(imagePath):
    # Load image
    image = cv2.imread(imagePath)
    image_name = os.path.splitext(os.path.basename(imagePath))[0]

    # Initialize saliency detector
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    # Compute saliency map
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    # Threshold saliency map
    threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find contours in saliency map
    contours, hierarchy = cv2.findContours(threshMap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create folder
    if not os.path.exists(f"regions/{image_name}"):
        os.makedirs(f"regions/{image_name}")

    regionsPath = []
    for i, contour in enumerate(contours):
        # Create bounding box around the contour and save as image
        x, y, w, h = cv2.boundingRect(contour)
        region = image[y:y+h, x:x+w]
        cv2.imwrite(f"regions/{image_name}/region{i}.png", region)

        # Append to list of regions
        regionsPath.append(f"regions/{image_name}/region{i}.png")

    return regionsPath

def changeRegionColor(img):
    result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgMat = np.zeros(img.shape, dtype="uint8")

    for contour in contours:
        cv2.fillPoly(imgMat, [contour], color=[255, 255, 255])

    mask = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(imgMat, mask, iterations=1)
    dilate = cv2.dilate(erosion, mask, iterations=1)

    objectEdges = cv2.Canny(dilate, 40, 100)
    objectContours, _ = cv2.findContours(objectEdges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in objectContours:
        color_hsv = np.array([random.randint(0, 179), random.randint(60, 255), random.randint(80, 255)])
        color_rgb = cv2.cvtColor(np.uint8([[color_hsv]]), cv2.COLOR_HSV2RGB)[0][0].tolist()
        cv2.fillPoly(result, [contour], color=color_rgb)

    return result

def changeEdgeColor(sourceImage):
    result = sourceImage.copy()
    gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Select 15 random contours to change color
    color_contours = random.sample(contours, 15)

    for contour in color_contours:
        color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
        color_tuple = tuple(map(int, color))
        cv2.drawContours(result, [contour], -1, color_tuple, 2)

    return result

def flipObject(template, sourceImage, location):
    h, w, _ = template.shape
    flipType = random.randint(-1, 1)
    template = cv2.flip(template, flipType)
    sourceImage[location[1]:location[1] + h, location[0]:location[0] + w] = template

def removeObject(template, sourceImage, location):
    h, w, _ = template.shape
    mask = np.zeros(sourceImage.shape[:2], np.uint8)
    mask[location[1]:location[1]+h, location[0]:location[0]+w] = 255
    sourceImage = cv2.inpaint(sourceImage, mask, 3, cv2.INPAINT_TELEA)

def distortObject(template, sourceImage, location):
    h, w, _ = template.shape
    distortion_scale = random.randint(3, 6)
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    sin_map_x = map_x + distortion_scale*np.sin(map_y/20)
    sin_map_y = map_y + distortion_scale*np.sin(map_x/20)
    distorted_template = cv2.remap(template, sin_map_x.astype('float32'), sin_map_y.astype('float32'), cv2.INTER_LINEAR)
    sourceImage[location[1]:location[1] + h, location[0]:location[0] + w] = distorted_template

def templateMatching(template, sourceImage):
    method = cv2.TM_CCORR_NORMED
    result = cv2.matchTemplate(sourceImage, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc
    return location

def circleDifferences(img1, img2, threshold=10, min_area=50):
    height = 400
    img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
    img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))

    # Grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray1 = cv2.GaussianBlur(gray1, (5,5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5,5), 0)

    # Calculate absolute difference between two arrays 
    diff = cv2.absdiff(gray1, gray2)

    # Apply threshold
    thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # Dilation
    kernel = np.ones((5,5), np.uint8) 
    dilate = cv2.dilate(thresh, kernel, iterations=2) 

    # Calculate contours
    contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            # Calculate bounding box around contour
            (x,y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x),int(y))
            radius = int(radius)
            # Draw circle - bounding box on both images
            cv2.circle(img1,center,radius,(0,0,255),2)
            cv2.circle(img2,center,radius,(0,0,255),2)

    # Create a separation line
    lineThickness = 5
    separationLine = np.zeros((height, lineThickness, 3), dtype=np.uint8)

    # Concatenate the images and the separation line
    combinedImage = np.concatenate((img1, separationLine, img2), axis=1)
    cv2.imshow("Differences", combinedImage)
    cv2.waitKey(0)