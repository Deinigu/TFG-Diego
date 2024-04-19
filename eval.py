import pandas as pd
import glob
from ultralytics import YOLO
import os

### VALIDATION WITH TEST DATASET ###
# Dataset
test_dataset = "data/test.yml"

# Get all weights in runs/detect/ folder
weights = glob.glob("runs/detect/**/best.pt", recursive=True)

# Iterate through all weights
for weight in weights:
    # Load the model
    model = YOLO(model=weight, task="detect")

    # Test the model with val test folder
    model.val(data=test_dataset, save=True)

### PREDICT ON TEST IMAGES ###
# Get all test images
test_images = glob.glob("data/test/images/*.png")
# Iterate through all weights
for weight in weights:
    # Load the model
    model = YOLO(model=weight, task="detect")

    # Predict on test images
    model.predict(source=test_images, imgsz=640, save_txt=True, save=True)


### MEAN AND STANDARD DEVIATION CALCULATIONS ###
file_path_pattern = "runs/detect/**/results.csv"

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
results_df.columns = results_df.columns.str.replace(" ", "")

# Group by the first column and calculate the mean and standard deviation
mean_df = results_df.groupby(results_df.columns[0]).mean()
std_df = results_df.groupby(results_df.columns[0]).std()

# Create directory eval if it does not exist
if not os.path.exists("eval"):
    os.makedirs("eval")

# Save mean_df to a CSV file
mean_df.to_csv("eval/mean_results.csv")

# Save std_df to a CSV file
std_df.to_csv("eval/std_results.csv")
