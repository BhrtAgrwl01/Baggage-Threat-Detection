import cv2
import numpy as np
from func import auto_canny, thresh_optima

image1 = cv2.imread(r"Blob_Centroid\d.jpeg")

image = cv2.bitwise_not(image1)

height, width, channels = image1.shape
imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(imgray, (3, 3), 0)
edged = cv2.Canny(blurred, *thresh_optima(blurred))
# edged = auto_canny(blurred)

thresh = edged #cv2.adaptiveThreshold(edged,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,2)
# thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)[1]

# thresh = cv2.erode(thresh, None, iterations=2)  # ignore
# thresh = cv2.dilate(thresh, None, iterations=2)  # ignore

# Finds all the closed contours in the image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for cnt in contours:

    area = cv2.contourArea(cnt)
    if area > 10:
        M = cv2.moments(cnt) # Moments of the contour

        if M["m00"] == 0:
            continue

        cx = int(M['m10']/M['m00']) # X-coordinate of the centroid
        cy = int(M['m01']/M['m00']) # Y-coordinate of the centroid

        # Coordinates strings
        coordinates2 = '('+str(cx)+','+str(cy)+')'

        cv2.circle(image1, (cx, cy), 1, (255, 255, 255), 2) # Marker

        # Height and Width strings
        heightstr = 'Height : ' + str(height)
        widthstr = 'Width : ' + str(width)

        # Text insertion commands
        cv2.putText(image1, coordinates2, (cx-100, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)	
        # cv2.putText(image1,coordinates1,(cx-120,cy-40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),1)
        cv2.putText(image1,heightstr, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(image1,widthstr, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.drawContours(image1, contours, -1, (0, 255, 0), 3)
# cv2.imshow('image threshold',image1)
# cv2.imshow('image',contours)

cv2.imshow('Canny',edged)

cv2.imshow("Image",image1)

cv2.imshow("Gaussian Blur", blurred)

cv2.waitKey(0)
cv2.destroyAllWindows()