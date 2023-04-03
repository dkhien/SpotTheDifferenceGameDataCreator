import numpy as np
import cv2
def changeColor(templates, img):
    img = cv2.imread('assets/starrynight.jpg', 1)

    for templatePath in templates:
        template = cv2.imread(templatePath, 1)
        h, w, _ = template.shape

        method = cv2.TM_CCORR_NORMED

        img2 = img.copy()

        result = cv2.matchTemplate(img2, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            location = min_loc
        else:
            location = max_loc

        # bottom_right = (location[0] + w, location[1] + h)    
        # cv2.rectangle(img2, location, bottom_right, (255, 255, 255), 5)

        # Get a random shift for hue, saturation, and value
        hue_shift = np.random.randint(0, 100)
        sat_shift = np.random.uniform(0.5, 2.0)
        val_shift = np.random.uniform(0.5, 1.0)

        # Convert the template to HSV color space
        template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

        # Shift the hue, saturation, and value
        template_hsv[:, :, 0] = (template_hsv[:, :, 0] + hue_shift) % 180
        template_hsv[:, :, 1] = np.clip(template_hsv[:, :, 1] * sat_shift, 0, 255)
        template_hsv[:, :, 2] = np.clip(template_hsv[:, :, 2], 0, 255)

        # Convert the template back to BGR color space and replace the original template with the modified one
        template_colored = cv2.cvtColor(template_hsv, cv2.COLOR_HSV2BGR)
        img2[location[1]:location[1]+h, location[0]:location[0]+w] = template_colored
    return img2

# cv2.imshow('Match', img2)
# cv2.waitKey(0)

# cv2.destroyAllWindows()
