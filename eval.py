import pandas as pd
import glob
import ultralytics
from ultralytics import YOLO
import os
from PIL import Image
import cv2
from IPython.display import Video
import glob
import matplotlib.pyplot as plt

# Load a model
model = YOLO(model='runs/detect/train/weights/best.pt', task='detect')

# Perform predictions for each photo in data/test/images
image_folder = 'data/test/images'
image_paths = glob.glob(os.path.join(image_folder, '*.png'))

predictions = []
for image_path in image_paths:
    prediction = model(source=image_path, imgsz=640, save=True, save_txt=True)
    predictions.append(prediction)

# Display the predictions
for prediction in predictions:
    print(prediction)

### MEAN AND STANDARD DEVIATION CALCULATIONS
file_path_pattern = 'runs/detect/**/results.csv'

# Get a list of all file paths matching the pattern
file_paths = glob.glob(file_path_pattern, recursive=True)

results = []

# Read each CSV file and append the data to the results array
for file_path in file_paths:
    data = pd.read_csv(file_path)
    results.append(data)

# Concatenate all the results into a single DataFrame
results_df = pd.concat(results) 

# Remove all spaces from columns
results_df.columns = results_df.columns.str.replace(' ', '')

# Group by the first column and calculate the mean and standard deviation
mean_df = results_df.groupby(results_df.columns[0]).mean()
std_df = results_df.groupby(results_df.columns[0]).std()

# Save mean_df to a CSV file
mean_df.to_csv('eval/mean_results.csv')

# Save std_df to a CSV file
std_df.to_csv('eval/std_results.csv')


