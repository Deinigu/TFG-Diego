from collections import defaultdict
import glob
import math
import operator
import os
from statistics import mean
import cv2
import numpy as np
from scipy import cluster, spatial
from sympy import Polygon
import chess_pieces_FEN_definition as FEN

# Show an image and wait for a key press
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
            point = int(np.round(point[0][0])), int(np.round(point[1][0]))
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

# Get the chess_piece based on the number
def get_piece_name(number):
    return FEN.chess_pieces.get(number, "Invalid number")
