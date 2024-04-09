import math
import os
from statistics import mean
import cv2
import numpy as np

# Load the image
name = "prueba"
img = cv2.imread("edge-detection/assets/" + name + ".png")

# Transform the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 125, 100, 10)
lines = np.reshape(lines, (-1, 2))

# Separate the lines into vertical and horizontal lines
h_lines, v_lines = [], []
for rho, theta in lines:
    if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
        v_lines.append([rho, theta])
    else:
        h_lines.append([rho, theta])

# Find and cluster the intersecting
points = []
for r_h, t_h in h_lines:
    for r_v, t_v in v_lines:
        a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
        b = np.array([r_h, r_v])
        inter_point = np.linalg.solve(a, b)
        points.append(inter_point)

points_shape = list(np.shape(points))
points = np.reshape(points, (points_shape[0], points_shape[1]))

# Draw the points
for point in points:
    x, y = point
    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

cv2.imshow("Chessboard", img)

# Wait for key press to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create results directory if it does not exist
if not os.path.exists("edge-detection/results"):
    os.makedirs("edge-detection/results")

# Save the image with the detected edges
cv2.imwrite("edge-detection/results/" + name + "_edges.png", img)
