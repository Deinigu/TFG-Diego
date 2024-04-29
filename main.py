import chess_functions as cf
from collections import defaultdict
import glob
import math
import operator
import os
from statistics import mean
import cv2
import numpy as np
from scipy import cluster, spatial
from sympy import Point, Polygon
import chess_pieces_FEN_definition as FEN
from datetime import datetime


class ChessPiece:
    def __init__(self, cell_coords, piece_coords, piece_type):
        self.cell_coords = cell_coords
        self.piece_coords = piece_coords
        self.piece_type = piece_type

# Debug
debug = False

# Load the image
name = "test"
img = cv2.imread("workspace/assets/" + name + ".png")

# Create a folder to save the results
save_path_folder = "workspace/results/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + name + "/"
if not os.path.exists(save_path_folder):
    os.makedirs(save_path_folder)

# Transform the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if debug:
    cf.show_image(gray, "Gray")

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)
if debug:
    cf.show_image(blur, "Blur")

# Apply Canny edge detection
edges = cf.canny_edge(blur)
if debug:
    cf.show_image(edges, "Edges")

# Hough Transform
lines = cf.hough_line(edges)

# Separate the lines into vertical and horizontal lines        
h_lines, v_lines = cf.h_v_lines(lines)

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
    cf.show_image(img_houg, "Hough Transform")
       
# Find and cluster the intersecting        
intersection_points = cf.line_intersections(h_lines, v_lines)
points = cf.cluster_points(intersection_points)

# Draw the points on the image
for point in points:
    cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    # Show the image with the detected edges
if debug:
    cf.show_image(img, "Result")

# Save the image with the detected edges
cv2.imwrite(save_path_folder + name + "_edges.png", img)
    
# Clean the image
img = cv2.imread("workspace/assets/" + name + ".png")

cells = cf.calculate_cells(points,debug,img.copy())
# print(cells)

# Read lines of the labels file
lines = []
with open("workspace/assets/labels/" + name + ".txt") as f:
    lines = f.readlines()

# Get the name of the chess pieces
chess_pieces = []

dh, dw, _ = img.shape
for line in lines:
    # Split the numbers in the line
    chesss_piece_number, x, y, w, h = map(float, line.split(' '))
    chesss_piece_number = int(chesss_piece_number)
    
    # Get the name of the chess piece in FEN notation
    chess_piece = cf.get_piece_name(chesss_piece_number)
    # print(chess_piece)
    
    l = float((x - w / 2) * dw)
    r = float((x + w / 2) * dw)
    t = float((y - h / 2) * dh)
    b = float((y + h / 2) * dh)
    
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1
    piece_coords = (l, r, t, b)
    
    # Get the cell where the piece is
    possible_cells = []
    for cell in cells:
        if cf.is_piece_in_cell(piece_coords, cell):
            possible_cells.append(cell)
    
    result = cf.get_cell_downer(possible_cells)
    
    chess_pieces.append(ChessPiece(cell_coords=result,piece_coords=(l,t,r,b), piece_type=chess_piece))

# Order the pieces to chess board order
ordered_chess_pieces = []
i = 0
j = 0
chessboard = [[None for i in range(8)] for j in range(8)]
for cell in cells:
    if i == 8:
        i = 0
        j += 1
    for piece in chess_pieces:
        if np.array_equal(piece.cell_coords, cell):
            # print(piece.piece_type, i, j)  
            chessboard[i][j] = piece
    i += 1

# Invert the rows (for FEN notation)
chessboard = chessboard[::-1]

    # Draw the pieces on the image
for i in range(8):
    for j in range(8):
        piece = chessboard[i][j]
        if piece is not None:
            l, t, r, b = piece.piece_coords
            cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.circle(img, (int(piece.cell_coords[0][0]), int(piece.cell_coords[0][1])), 5, (0, 0, 255), -1)
            cv2.circle(img, (int(piece.cell_coords[1][0]), int(piece.cell_coords[1][1])), 5, (0, 0, 255), -1)
            cv2.circle(img, (int(piece.cell_coords[2][0]), int(piece.cell_coords[2][1])), 5, (0, 0, 255), -1)
            cv2.circle(img, (int(piece.cell_coords[3][0]), int(piece.cell_coords[3][1])), 5, (0, 0, 255), -1)
            if debug:
                cf.show_image(img,"Chess pieces")

cv2.imwrite(save_path_folder + name + "_result.png", img)

# Generate the FEN notation
fen = cf.generate_fen_notation(chessboard)
print("https://lichess.org/editor/" + fen)

# Save the FEN notation in a file
with open(save_path_folder + name + "_fen.txt", "w") as f:
    f.write(fen)