import chess_functions as cf
import glob
import os
import cv2
import numpy as np
from datetime import datetime
import argparse_functions as af


# Chess piece class
class ChessPiece:
    def __init__(self, cell_coords, piece_coords, piece_type):
        self.cell_coords = cell_coords
        self.piece_coords = piece_coords
        self.piece_type = piece_type


# Get the arguments
args = af.argparse_main()

# Debug
debug = args.debug

# Unicode output
unicode_output = not args.utf8

# Workspaces paths
workspace_path = "workspace/"
images_path = args.image
results_path = workspace_path + "results/"
model_folder = "workspace/model/"

# Create results folder if it doesn't exist
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Model path
if args.model is not None:
    model_path = model_folder + args.model + ".pt"

    if not os.path.exists(model_path):
        print(
            "The model could not be loaded. Check that the model exists in the folder: "
            + model_folder
            + "."
        )
        # Finish the program
        exit()
else:
    model_path = (
        glob.glob(model_folder + "*.pt")[0]
        if len(glob.glob(model_folder + "*.pt")) > 0
        else None
    )

# Check if the model was found
if model_path is None:
    print(
        "The model could not be loaded. Check that there is a model in the folder: "
        + model_folder
        + "."
    )
    # Finish the program
    exit()
else:
    print("Model loaded: " + model_path)

# Load the image
supported_image_formats = [".png", ".jpg", ".jpeg"]
img = None

# Check if the image exists and it's in one of the supported formats
if os.path.isfile(images_path):
    name, format = os.path.splitext(os.path.basename(images_path))
    if format not in supported_image_formats:
        print(
            "The image format is not supported. Check that the format is correct: "
            + str(supported_image_formats)
            + "."
        )
        # Finish the program
        exit()
    img = cf.initialize_image(images_path)
else:
    print("The image could not be loaded. Check the image path. ")
    # Finish the program
    exit()

# Create a folder to save the results
save_path_folder = (
    results_path + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + name + "/"
)
if not os.path.exists(save_path_folder):
    os.makedirs(save_path_folder)

# Predict the chess pieces in the image
results = cf.predict_chess_pieces(images_path, model_path, save_path_folder)

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
    cf.show_image(edges, "Canny Edge Detection")

# Hough Transform
lines = cf.hough_line(edges)

# Separate the lines into vertical and horizontal lines
h_lines, v_lines = cf.h_v_lines(lines)

if debug:
    img_houg = cf.create_img(img)
    if len(h_lines) < 9 or len(v_lines) < 9:
        print(
            "There are not enough horizontal and vertical lines in this image. Try it anyway!"
        )
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
    cf.show_image(img, "Edges")

# Save the image with the detected edges
cv2.imwrite(save_path_folder + name + "_edges.png", img)

# Clean the image
img = cf.initialize_image(images_path)

# Calculate the cells
img_cells = cf.create_img(img)
cells = cf.calculate_cells(points, debug, img_cells)
# print(cells)

# Save the image with the cells
cv2.imwrite(save_path_folder + name + "_cells.png", img_cells)

# If there are not chess pieces detected, finish the program
if not os.path.exists(save_path_folder + "predict/labels/" + name + ".txt"):
    print("The model didn't detect any chess pieces. Check the image and the model.")
    # Finish the program
    exit()

# Read lines of the labels file
lines = []
with open(save_path_folder + "predict/labels/" + name + ".txt") as f:
    lines = f.readlines()

# Get the name of the chess pieces
chess_pieces = []

dh, dw, _ = img.shape
for line in lines:
    # Split the numbers in the line
    chess_piece_number, x, y, w, h = map(float, line.split(" "))
    chess_piece_number = int(chess_piece_number)

    # Get the name of the chess piece in FEN notation
    chess_piece = cf.get_piece_name(chess_piece_number)
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
    piece_coords = (l, t, r, b)

    # Get the cell where the piece is
    possible_cells = []
    for cell in cells:
        if cf.is_piece_in_cell(piece_coords, cell):
            possible_cells.append(cell)

    result = cf.get_nearest_cell(piece_coords, possible_cells)

    chess_pieces.append(
        ChessPiece(
            cell_coords=result, piece_coords=(l, t, r, b), piece_type=chess_piece
        )
    )


# Order the pieces to chess board order
i = 0
j = 0
chessboard = [[None for i in range(8)] for j in range(8)]
rest_of_cells = []
cell_index = 0

for cell in cells:
    if i == 8:
        i = 0
        j += 1
    if j == 8:
        rest_of_cells = cells[cell_index:]
        break
    for piece in chess_pieces:
        if np.array_equal(piece.cell_coords, cell):
            # print(piece.piece_type, i, j)
            chessboard[i][j] = piece
            break
    i += 1
    cell_index += 1

if debug:
    print("Chessboard (no extra column process):")
    cf.print_chessboard(
        chessboard[::-1], unicode=unicode_output
    )  # Invert the rows (for FEN notation)

# Check if there are any pieces left
extra_column_needed = False
for piece in chess_pieces:
    if piece not in [piece for row in chessboard for piece in row]:
        extra_column_needed = True
        break

# Add extra columns if needed
while extra_column_needed and len(rest_of_cells) > 0:
    # Move all the pieces to the left
    for i in range(8):
        for j in range(7):
            if chessboard[i][j] is None:
                chessboard[i][j] = chessboard[i][j + 1]
                chessboard[i][j + 1] = None
    # Add the pieces that were left to the last column
    for cell in rest_of_cells:
        for piece in chess_pieces:
            if np.array_equal(piece.cell_coords, cell):
                for i in range(8):
                    if chessboard[i][7] is None:
                        chessboard[i][7] = piece
                        break
        cell_index += 1

    # Update the rest of the cells
    rest_of_cells = cells[cell_index:]

    # Check if there are any pieces left
    extra_column_needed = False
    for piece in chess_pieces:
        if piece not in [piece for row in chessboard for piece in row]:
            extra_column_needed = True
            break
    if debug:
        print("Chessboard (extra column process):")
        cf.print_chessboard(
            chessboard[::-1], unicode=unicode_output
        )  # Invert the rows (for FEN notation)

# Invert the rows (for FEN notation)
chessboard = chessboard[::-1]

# Print the chessboard rotated to the right (as you see it in the image)
chessboard_rotated = list(map(list, zip(*chessboard[::-1])))
print("Chessboard from the perspective of the image:")
cf.print_chessboard(chessboard_rotated, unicode=unicode_output)

# Print the chessboard
print("Final Result:")
cf.print_chessboard(chessboard, unicode=unicode_output)


# Draw the pieces on the image
for i in range(8):
    for j in range(8):
        piece = chessboard[i][j]
        if piece is not None:
            l, t, r, b = piece.piece_coords
            cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.circle(
                img,
                (int(piece.cell_coords[0][0]), int(piece.cell_coords[0][1])),
                5,
                (0, 0, 255),
                -1,
            )
            cv2.circle(
                img,
                (int(piece.cell_coords[1][0]), int(piece.cell_coords[1][1])),
                5,
                (0, 0, 255),
                -1,
            )
            cv2.circle(
                img,
                (int(piece.cell_coords[2][0]), int(piece.cell_coords[2][1])),
                5,
                (0, 0, 255),
                -1,
            )
            cv2.circle(
                img,
                (int(piece.cell_coords[3][0]), int(piece.cell_coords[3][1])),
                5,
                (0, 0, 255),
                -1,
            )
            cv2.polylines(img, [piece.cell_coords.astype(int)], True, (0, 0, 255), 1)
            if debug:
                cf.show_image(img, "Chess pieces")

cv2.imwrite(save_path_folder + name + "_result.png", img)

# Generate the FEN notation
fen = cf.generate_fen_notation(chessboard)
print("https://lichess.org/editor/" + fen)

# Save the FEN notation in a file
with open(save_path_folder + name + "_fen.txt", "w") as f:
    f.write(fen)
