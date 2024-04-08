import os
import cv2
import numpy as np

# Load the image
img = cv2.imread('edge-detection/assets/chessboard.png')

# Transform the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# Find contours in the image
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
chessboard_cnt = None

for cnt in contours:
    # Calculate the perimeter of the contour
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # If the contour has 4 corners, it is the chessboard
    if len(approx) == 4:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            chessboard_cnt = approx

# Draw the contours of the chessboard
if chessboard_cnt is not None:
    # Draw the contours of the chessboard
    cv2.drawContours(img, [chessboard_cnt], -1, (0, 255, 0), 3)

    # Draw circles on the corners of the chessboard
    for point in chessboard_cnt:
        cv2.circle(img, tuple(point[0]), 5, (0, 0, 255), -1)

# Show the image
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create results directory if it does not exist
if not os.path.exists('edge-detection/results'):
    os.makedirs('edge-detection/results')

# Save the image with the detected edges
cv2.imwrite('edge-detection/results/chessboard_edges.png', img)
