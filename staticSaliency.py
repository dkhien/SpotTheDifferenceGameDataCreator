import cv2
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