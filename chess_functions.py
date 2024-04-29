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
from ultralytics import YOLO

# Predict the chess pieces in the image
def predict_chess_pieces(image_path, model_path, save_path_folder):
    # Load the model
    model = YOLO(model=model_path, task="detect")

    # Predict on the image
    results = model.predict(source=image_path, imgsz=640, save_txt=True, save=True, project=save_path_folder)
    return results

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

# Fen notation generator
def generate_fen_notation(chessboard):
    fen = ""
    for i in range(8):
        empty = 0
        for j in range(8):
            piece = chessboard[i][j]
            if piece is None:
                empty += 1
            else:
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                fen += piece.piece_type
        if empty > 0:
            fen += str(empty)
        if i < 7:
            fen += "/"
    return fen

# Returns a array of 64 cells with the coordinates of the corners of each cell
def calculate_cells(points, debug=False, img_cells=None):
    # Order the points by y coordinate (top to bottom)
    coordinates = []
    for point in points:
        coordinates.append([point[0], point[1]])
    coordinates = sorted(coordinates, key=lambda x: (x[1]))

    # Knowing that the points are ordered from top to bottom, we can divide them into cells
    cells = []
    for i in range(0, len(coordinates)-9,1):
        if ((i+1) % 9 > 0 or i == 0):
            cell = np.array([coordinates[i],coordinates[i+1],coordinates[i+9],coordinates[i+10]])
            if(debug):
                for point in cell:
                    cv2.circle(img_cells, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
                    # Show the image with the detected edges
                show_image(img_cells, "Casillas")
            cells.append(cell)
    return cells

def is_piece_in_cell(piece_coords, cell_coords):
    l, r, t, b = piece_coords  # Coordenadas de la pieza
    cell_tl, cell_tr, cell_bl, cell_br = cell_coords  # Coordenadas de la celda

    cell_l = min(cell_tl[0], cell_tr[0], cell_bl[0], cell_br[0])
    cell_r = max(cell_tl[0], cell_tr[0], cell_bl[0], cell_br[0])
    cell_t = min(cell_tl[1], cell_tr[1], cell_bl[1], cell_br[1])
    cell_b = max(cell_tl[1], cell_tr[1], cell_bl[1], cell_br[1])

    if (l < cell_r and r > cell_l) and (t < cell_b and b > cell_t):
        return True
    else:
        return False

# Return the cell which is downer
def get_cell_downer(cells):
    downer = cells[0]
    for cell in cells:
        if cell[0][1] > downer[0][1]:
            downer = cell
    return downer
