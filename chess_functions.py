from collections import defaultdict
import math
from statistics import mean
import cv2
import numpy as np
from scipy import cluster, spatial
import chess_pieces_FEN as FEN
from ultralytics import YOLO


# Predict the chess pieces in the image
def predict_chess_pieces(image_path, model_path, save_path_folder):
    # Load the model
    model = YOLO(model=model_path, task="detect")

    # Predict on the image
    results = model.predict(
        source=image_path, imgsz=640, save_txt=True, save=True, project=save_path_folder
    )
    return results


# Initialize the image
def initialize_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    # Resize the image to half if it is too big
    if img.shape[0] > 1000 or img.shape[1] > 1000:
        print("Your image is too big! It will be resized.")
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    return img


# Create an image
def create_img(img):
    result = img.copy()
    return result


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
            A = np.array(
                [[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]]
            )
            b = np.array([[rho1], [rho2]])
            point = np.linalg.solve(A, b)
            point = int(np.round(point[0][0])), int(np.round(point[1][0]))
            points.append(point)
    return np.array(points)


# Hierarchical cluster (by euclidean distance) intersection points
def cluster_points(points, max_dist=20):
    Y = spatial.distance.pdist(points)
    Z = cluster.hierarchy.single(Y)
    T = cluster.hierarchy.fcluster(Z, max_dist, "distance")
    clusters = defaultdict(list)
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    clusters = clusters.values()
    clusters = map(
        lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])),
        clusters,
    )

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


# Print the chessboard
def print_chessboard(chessboard, unicode=True):
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] is not None:
                (
                    print(
                        FEN.chess_pieces_unicode[chessboard[i][j].piece_type], end=" "
                    )
                    if unicode
                    else print(chessboard[i][j].piece_type, end=" ")
                )
            else:
                print("▢", end=" ") if unicode else print("X", end=" ")
        print()
    print()


# Returns a array of cells with the coordinates of the corners of each cell
def calculate_cells(points, debug=False, img_cells=None):

    # Filter the points to remove negative values
    points = [point for point in points if point[0] > 0 and point[1] > 0]

    # Create a list of virtual points where the y is the mean of the y of the points in the same row
    points = sorted(points, key=lambda x: x[1])
    virtual_points = []

    # Divide the points into rows based on the y coordinate
    rows = []
    current_row = []
    current_y = points[0][1]

    # Tolerance
    tolerance = 20

    for point in points:
        if abs(point[1] - current_y) <= tolerance:
            current_row.append(point)
        else:
            rows.append(current_row)
            current_row = [point]
            current_y = point[1]

    # Add the last row
    rows.append(current_row)

    if debug:
        img_rows = create_img(img_cells)
        # Draw the points
        for row in rows:
            for point in row:
                cv2.circle(img_rows, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
            show_image(img_rows, "Rows")

    # Calculate the mean y of each row
    for row in rows:
        y = mean([point[1] for point in row])
        row_with_virtual_y = []

        # Create a list of points with the mean y
        for point in row:
            row_with_virtual_y.append([point[0], y])

        # Add the row to the virtual points
        virtual_points.extend(row_with_virtual_y)

    # Get order indexes of the virtual points
    order = sorted(
        range(len(virtual_points)),
        key=lambda k: (virtual_points[k][1], virtual_points[k][0]),
    )

    # Order the real points
    coordinates = []
    for i in order:
        coordinates.append(points[i])

    # Delete the rows with less or equal to the umbral of points
    points_umbral = 4
    for row in rows:
        if len(row) <= points_umbral:
            if debug:
                print("Deleting row this row: ", row)
            for point in row:
                coordinates.remove(point)
            rows.remove(row)

    # Knowing that the points are ordered from top to bottom, we can divide them into cells
    cells = []

    # Save the size of each row
    row_sizes = [len(row) for row in rows]

    # Choose the cells based on the row sizes
    size_index = 0
    row_size = row_sizes[size_index]
    iterator = 0
    # print(row_sizes)
    for i in range(0, len(coordinates) - row_sizes[len(row_sizes) - 1], 1):
        row_size = row_sizes[size_index]
        if ((iterator + 1) % row_size > 0 or iterator == 0) and i + row_size + 1 < len(
            coordinates
        ):
            cell = np.array(
                [
                    coordinates[i],
                    coordinates[i + 1],
                    coordinates[i + row_size],
                    coordinates[i + row_size + 1],
                ]
            )
            iterator += 1
            # Draw the points of the cell
            for point in cell:
                cv2.circle(
                    img_cells, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1
                )
            # Draw the lines of the cell
            cv2.polylines(img_cells, [cell.astype(int)], True, (0, 255, 0), 2)
            if debug:
                # Show the image with the detected edges
                show_image(img_cells, "Cells")

            cells.append(cell)
        else:
            size_index += 1 if size_index + 1 < len(row_sizes) - 1 else 0
            iterator = 0
    return cells


def mean_point(point1, point2):
    return (point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def is_piece_in_cell(piece_coords, cell_coords):
    l, t, r, b = piece_coords  # Piece coordinates
    cell_tl, cell_tr, cell_bl, cell_br = cell_coords  # Cell coordinates

    cell_l = min(cell_tl[0], cell_tr[0], cell_bl[0], cell_br[0])
    cell_r = max(cell_tl[0], cell_tr[0], cell_bl[0], cell_br[0])
    cell_t = min(cell_tl[1], cell_tr[1], cell_bl[1], cell_br[1])
    cell_b = max(cell_tl[1], cell_tr[1], cell_bl[1], cell_br[1])

    if (l < cell_r and r > cell_l) and (t < cell_b and b > cell_t):
        return True
    else:
        return False


# Return the cell where the piece is located
def get_nearest_cell(piece_coords, cells):
    l, t, r, b = piece_coords  # Piece coordinates

    # Get the mean point of the two lower points of the piece
    piece_mean_point = mean_point((l, b), (r, b))

    cell_distance = 1000000  # Big number
    nearest_cell = None

    # Get the cell which its mean point in the lowest line is nearest to the piece's mean point
    for cell in cells:

        cell_distance_temp = euclidean_distance(
            piece_mean_point, mean_point(cell[2], cell[3])
        )
        # Update the nearest cell distance
        if cell_distance_temp < cell_distance:
            cell_distance = cell_distance_temp
            nearest_cell = cell

    return nearest_cell
