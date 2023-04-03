import cv2
from utils import *

difficulties = [5, 7, 10]
level1Operations = [changeColor]
level2Operations = [changeColor, flipObject, removeObject]
# level3Operations = [changeColor, flipObject, removeObject, duplicateObject, distortObject]
level3Operations = [duplicateObject]
operations = [level1Operations, level2Operations, level3Operations]

def createLevel(templates, sourceImage, level):
    img = sourceImage.copy()
    templates_copy = templates.copy()
    imgH, imgW, _ = img.shape
    for i in range(difficulties[level]):
        if not templates:
            break
        # Choose a random template from the list
        templatePath = random.choice(templates_copy)
        template = cv2.imread(templatePath)
        h, w, _ = template.shape
        # while h < imgH / 10 and w < imgW / 10:
        templatePath = random.choice(templates_copy)
        template = cv2.imread(templatePath)
        h, w, _ = template.shape
        templates_copy.remove(templatePath)
        location = templateMatching(template, img)
        operation = random.choice(operations[level])
        operation(template, img, location)
    return img






