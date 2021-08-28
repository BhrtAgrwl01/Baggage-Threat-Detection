import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	# return the edged image
	return cv2.Canny(image, lower, upper)

def thresh_optima(img):
    high_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    low_thresh = 0.5*high_thresh
    return low_thresh, high_thresh