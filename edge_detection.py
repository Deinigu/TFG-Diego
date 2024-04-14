from collections import defaultdict
import math
import operator
import os
from statistics import mean
import cv2
import numpy as np
from scipy import cluster, spatial


def show_image(img, name="Debug"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Canny edge detection
def canny_edge(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper)
    return edges

# Hough Transform
def hough_line(edges, min_line_length=100, max_line_gap=15):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 125, min_line_length, max_line_gap)
    lines = np.reshape(lines, (-1, 2))
    return lines

# Separate line into horizontal and vertical
def h_v_lines(lines):
    h_lines, v_lines = [], []
    for rho, theta in lines:
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])
    return h_lines, v_lines

# Find the intersections of the lines
def line_intersections(h_lines, v_lines):
    points = []
    for rho1, theta1 in h_lines:
        for rho2, theta2 in v_lines:
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            point = np.linalg.solve(A, b)
            point = int(np.round(point[0])), int(np.round(point[1]))
            points.append(point)
    return np.array(points)

# Hierarchical cluster (by euclidean distance) intersection points
def cluster_points(points, max_dist=20):
    Y = spatial.distance.pdist(points)
    Z = cluster.hierarchy.single(Y)
    T = cluster.hierarchy.fcluster(Z, max_dist, 'distance')
    clusters = defaultdict(list)
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    clusters = clusters.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), clusters)

    result = []
    for point in clusters:
        result.append([point[0], point[1]])
    return result

# Debug
debug = False

# Load the image
name = "prueba_sin_fondo"
img = cv2.imread("edge-detection/assets/" + name + ".png")

# Transform the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if debug:
    show_image(gray, "Gray")

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)
if debug:
    show_image(blur, "Blur")

# Apply Canny edge detection
edges = canny_edge(blur)
if debug:
    show_image(edges, "Edges")

# Hough Transform
lines = hough_line(edges)

# Separate the lines into vertical and horizontal lines        
h_lines, v_lines = h_v_lines(lines)

if debug:
    img_houg = img.copy()
    if len(h_lines) < 9 or len(v_lines) < 9:
        print("There are not enough horizontal and vertical lines in this image. Try it anyway!")
    # Draw the lines on the image
    for rho, theta in h_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_houg, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for rho, theta in v_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_houg, (x1, y1), (x2, y2), (0, 0, 255), 2)
    show_image(img_houg, "Hough Transform")
       
# Find and cluster the intersecting        
intersection_points = line_intersections(h_lines, v_lines)
points = cluster_points(intersection_points)

# Draw the points on the image
for point in points:
    cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

# Show the image with the detected edges
show_image(img, "Result")

# Create results directory if it does not exist
if not os.path.exists("edge-detection/results"):
    os.makedirs("edge-detection/results")

# Save the image with the detected edges
cv2.imwrite("edge-detection/results/" + name + "_edges.png", img)
