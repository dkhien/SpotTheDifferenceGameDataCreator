import cv2
from utils import *

def createLevel(templates, sourceImage, level):
    if level == 0:
        img = changeEdgeColor(sourceImage)
        return img
    elif level == 1:
        img = changeRegionColor(sourceImage)
        return img
    else:
        level3Operations = [flipObject, removeObject, distortObject]
        img = sourceImage.copy()
        templates_copy = templates.copy()
        imgH, imgW, _ = img.shape
    
        for it in templates_copy:
            if not templates:
                break

            # Choose a random template from the list
            while True:
                templatePath = random.choice(templates_copy)
                template = cv2.imread(templatePath)
                h, w, _ = template.shape

                # Excluding templates that are too small or too large
                if (h > imgH/15 or w > imgW/15) and (h < imgH/2 or w < imgW/2):
                    break
            templates_copy.remove(templatePath)
            
            # Template matching to find location of template 
            location = templateMatching(template, img)
            
            # Modify the object at that location using a random method
            operation = random.choice(level3Operations)
            operation(template, img, location)
        return img






