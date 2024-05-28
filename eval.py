import pandas as pd
import glob
from ultralytics import YOLO
import os
from datetime import datetime
import eval_functions as ef

# Get the current date and time
now = datetime.now()

# Directories to save the evaluation results
root_dir = "eval/"
eval_dir = root_dir + now.strftime("%Y-%m-%d_%H-%M-%S") + "/"
val_dir = eval_dir + "validation/"
pred_dir = eval_dir + "predictions/"

# Create directory eval if it does not exist
os.makedirs(root_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)

### VALIDATION WITH TEST DATASET ###
# Dataset
test_dataset = "data/test.yml"

# Get all weights in runs/detect/ folder
weights = glob.glob("runs/**/best.pt", recursive=True)

# Iterate through all weights
for weight in weights:
    # Load the model
    model = YOLO(model=weight, task="detect")

    # Test the model with val test folder
    model.val(data=test_dataset, save=True, project=val_dir)

### PREDICT ON TEST IMAGES WITH DIFFERENT BRIGHTNESS CONFIGURATION ###
# i=1 is the original brightness
# i=3 is less brightness
# i=9 is the least brightness
# Get all test images
brightness_values = [1,3,9]
for i in brightness_values:
    # Get all test images
    test_images = glob.glob("data/test/images/*.png")

    # Adjust the brightness of the test images with the value of i
    test_images = ef.adjust_brightness(test_images, i)

    # Iterate through all weights
    for weight in weights:
        # Load the model
        model = YOLO(model=weight, task="detect")

        # Predict on test images
        model.predict(
            source=test_images,
            imgsz=640,
            save_conf=True,
            save_txt=True,
            save=True,
            project=pred_dir + "brightness_" + str(i) + "/",
        )


### MEAN AND STANDARD DEVIATION CALCULATIONS ###
file_path_pattern = "runs/**/results.csv"

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

# Save mean_df to a CSV file
if mean_df.to_csv(eval_dir + "mean_results.csv"):
    print("Saved mean_results.csv successfully to" + eval_dir + "mean_results.csv")

# Save std_df to a CSV file
if std_df.to_csv(eval_dir + "std_results.csv"):
    print("Saved std_results.csv successfully to" + eval_dir + "std_results.csv")
