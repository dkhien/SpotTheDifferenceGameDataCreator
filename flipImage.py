import cv2

image = cv2.imread("assets/avengers.png")
image = cv2.flip(image, 0)
cv2.imshow("Flipped", image)
cv2.waitKey(0)