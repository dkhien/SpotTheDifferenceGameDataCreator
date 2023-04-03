import cv2
import numpy as np
import imutils
import random
import os

def detectSaliency(imagePath):
    # Load image
    image = cv2.imread(imagePath)
    image_name = os.path.splitext(os.path.basename(imagePath))[0]

    # Initialize fine-grained saliency detector
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    # Compute saliency map
    (success, saliencyMap) = saliency.computeSaliency(image)

    # Normalize saliency map for visualization
    saliencyMap = (saliencyMap * 255).astype("uint8")

    # Threshold the saliency map
    threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find contours in the thresholded saliency map
    contours, hierarchy = cv2.findContours(threshMap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list to store the extracted regions
    regions = []

    # Create folder for saving regions
    if not os.path.exists(f"regions/{image_name}"):
        os.makedirs(f"regions/{image_name}")

    regionsPath = []
    # Iterate over the contours
    for i, contour in enumerate(contours):
        # Create a bounding box around the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract the region of interest from the image and save it as an individual image
        region = image[y:y+h, x:x+w]
        cv2.imwrite(f"regions/{image_name}/region{i}.png", region)
        
        # Append the region to the list of regions
        regions.append(region)
        regionsPath.append(f"regions/{image_name}/region{i}.png")

    # Display results
    # cv2.imshow("Image", image)
    # cv2.imshow("Saliency Map", saliencyMap)
    # cv2.imshow("Thresh Map", threshMap)
    # cv2.waitKey(0)
    return regionsPath


def changeColor(template, sourceImage, location):
    h, w, _ = template.shape
    # Get a random shift for hue, saturation, and value
    hue_shift = np.random.randint(0, 180)
    sat_shift = np.random.uniform(0.5, 2.0)
    val_shift = np.random.uniform(0.5, 1.0)

    # Convert the template to HSV color space
    template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

    # Shift the hue, saturation, and value
    template_hsv[:, :, 0] = (template_hsv[:, :, 0] + hue_shift) % 180
    template_hsv[:, :, 1] = np.clip(template_hsv[:, :, 1] * sat_shift, 0, 255)
    # template_hsv[:, :, 2] = np.clip(template_hsv[:, :, 2] * val_shift, 0, 255)

    # Convert the template back to BGR color space and replace the original template with the modified one
    template_colored = cv2.cvtColor(template_hsv, cv2.COLOR_HSV2BGR)
    sourceImage[location[1]:location[1] + h, location[0]:location[0] + w] = template_colored

def changePixelColor(template, sourceImage, location):
    # Load the image
    img2 = sourceImage.copy()
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

def flipObject(template, sourceImage, location):
    h, w, _ = template.shape
    flipType = random.randint(-1, 1)
    template = cv2.flip(template, flipType)
    sourceImage[location[1]:location[1] + h, location[0]:location[0] + w] = template

def removeObject(template, sourceImage, location):
    h, w, _ = template.shape
    # Inpaint the template regions using surrounding pixels
    mask = np.zeros(sourceImage.shape[:2], np.uint8)
    mask[location[1]:location[1]+h, location[0]:location[0]+w] = 255
    sourceImage = cv2.inpaint(sourceImage, mask, 3, cv2.INPAINT_TELEA)

def duplicateObject(template, sourceImage, location):
    h, w, _ = template.shape

    # Create a duplicate of the template
    template_copy = template.copy()

    # Choose a random region to place the duplicate template
    x_offset = random.randint(0, sourceImage.shape[1] - w)
    y_offset = random.randint(0, sourceImage.shape[0] - h)

    # Place the duplicate template in the random region
    # roi = sourceImage[y_offset:y_offset+h, x_offset:x_offset+w]

    # Slightly blur and blend the edges of the template with the background
    # sigma = 25
    # ksize = int(2 * round(2 * sigma) + 1)
    # blurred_template = cv2.GaussianBlur(template_copy, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    # alpha = 0.5
    # blended = cv2.addWeighted(blurred_template, alpha, roi, 1-alpha, 0, roi)

    sourceImage[y_offset:y_offset+h, x_offset:x_offset+w] = template_copy

def distortObject(template, sourceImage, location):
    h, w, _ = template.shape
    distortion_scale = random.randint(3, 10)
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

def circleDifferences(img1, img2, threshold=10, min_area=100):
    height = 400
    img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
    img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))

    # Grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray1 = cv2.GaussianBlur(gray1, (5,5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5,5), 0)

    # Apply Canny edge detection
    # edges1 = cv2.Canny(gray1, 100, 200)
    # edges2 = cv2.Canny(gray2, 100, 200)

    # Find the difference between the two images
    # Calculate absolute difference between two arrays 
    diff = cv2.absdiff(gray1, gray2)
    # cv2.imshow("Diff", diff)
    # Apply threshold
    thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow("Thresh", thresh)
    # Dilation
    kernel = np.ones((5,5), np.uint8) 
    dilate = cv2.dilate(thresh, kernel, iterations=2) 
    # cv2.imshow("Dilate", dilate)
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

    # Create a black separation line
    lineThickness = 5
    separationLine = np.zeros((height, lineThickness, 3), dtype=np.uint8)

    # Concatenate the images and the separation line horizontally
    combinedImage = np.concatenate((img1, separationLine, img2), axis=1)
    cv2.imshow("Differences", combinedImage)
    cv2.waitKey(0)