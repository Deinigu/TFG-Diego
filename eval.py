import pandas as pd
import glob
from ultralytics import YOLO
import os
from datetime import datetime
import eval_functions as ef
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import argparse_functions as af

# Get the current date and time
now = datetime.now()

# Get the arguments
args = af.argparse_eval()

# Get all weights in args.directory
weights = glob.glob(args.directory + "**/*.pt", recursive=True)

# Print the weights
print("Weights found:")
for weight in weights:
    print(weight)

# Get tests to run
all_tests = args.all
validation_test = args.validation
prediction_test = args.prediction
mean_test = args.mean
std_test = args.std

# If no tests are selected, run all tests
if not any([all_tests, validation_test, prediction_test, mean_test, std_test]):
    all_tests = True

# Dataset root directory
root_path = "../TFG-Diego/"

# Directories to save the evaluation results
root_eval_dir = "workspace/eval/"
eval_dir = root_eval_dir + now.strftime("%Y-%m-%d_%H-%M-%S") + "/"
val_dir = eval_dir + "validation/"
pred_dir = eval_dir + "predictions/"
data_dir = "data/"
brightness_datasets_dir = data_dir + "brightness_variations/"

# Create directory eval if it does not exist
os.makedirs(root_eval_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(brightness_datasets_dir, exist_ok=True)

### CREATE DATASETS WITH DIFFERENT BRIGHTNESS CONFIGURATIONS ###
# i=1 is the original brightness
# i=9 is the least brightness
if all_tests or validation_test or prediction_test:
    brightness_values = [1, 3, 5, 7, 9]
    brigtness_datasets_list = []  # List to store the brightness datasets paths

    # Add the original dataset to the list
    brigtness_datasets_list.append(data_dir + "test.yml")

    # Create the datasets with different brightness configurations
    for i in brightness_values[1:]:  # Skip the original brightness

        # If the dataset already exists, skip the creation
        if os.path.exists(brightness_datasets_dir + "brightness_" + str(i) + "/"):
            print("Dataset with brightness " + str(i) + " already exists.")
            # save the dataset to the list
            brigtness_datasets_list.append(
                brightness_datasets_dir + "brightness_" + str(i) + "/test.yml"
            )
            continue

        print("Creating dataset with brightness: " + str(i))
        # Create the directories for the test images and labels
        brightness_images_dir = (
            brightness_datasets_dir + "brightness_" + str(i) + "/test/images/"
        )
        brightness_labels_dir = (
            brightness_datasets_dir + "brightness_" + str(i) + "/test/labels/"
        )

        # Create the directories for the train images and labels (not using them, but needed for the dataset creation)
        brightness_train_images_dir = (
            brightness_datasets_dir + "brightness_" + str(i) + "/train/images/"
        )
        brightness_train_labels_dir = (
            brightness_datasets_dir + "brightness_" + str(i) + "/train/labels/"
        )

        # Create folders
        os.makedirs(brightness_datasets_dir + "brightness_" + str(i) + "/", exist_ok=True)
        os.makedirs(brightness_images_dir, exist_ok=True)
        os.makedirs(brightness_labels_dir, exist_ok=True)

        os.makedirs(brightness_train_images_dir, exist_ok=True)
        os.makedirs(brightness_train_labels_dir, exist_ok=True)

        # Get all test images and adjust the brightness
        test_images = glob.glob("data/test/images/*.png")

        # Save in a list the names of the images
        test_images_names = [os.path.basename(image) for image in test_images]

        # Adjust the brightness of the test images with the value of i
        test_images = ef.adjust_brightness(test_images, i)

        # Save the images in the new dataset
        ef.save_images(test_images, brightness_images_dir, test_images_names)

        # Get the labels
        test_labels = glob.glob("data/test/labels/*.txt")

        # Copy the labels to the new dataset
        for label in test_labels:
            shutil.copy(label, brightness_labels_dir)

        # Copy the test.yml file and change its name
        shutil.copy(
            data_dir + "test.yml",
            brightness_datasets_dir + "brightness_" + str(i) + "/test.yml",
        )

        # Change lines of the test.yml file
        with open(
            brightness_datasets_dir + "brightness_" + str(i) + "/test.yml", "r"
        ) as file:
            data = file.readlines()
        data[0] = (
            "path: " + root_path + brightness_datasets_dir + "brightness_" + str(i) + "\n"
        )

        # Write the new data to the file
        with open(
            brightness_datasets_dir + "brightness_" + str(i) + "/test.yml", "w"
        ) as file:
            file.writelines(data)

        # Save the dataset to the list
        brigtness_datasets_list.append(
            brightness_datasets_dir + "brightness_" + str(i) + "/test.yml"
        )
        
        # Print the success message
        print("\nDataset with brightness " + str(i) + " created in " + brightness_datasets_dir + "brightness_" + str(i) + "/\n")

### VALIDATION WITH DIFFERENT BRIGHTNESS CONFIGURATIONS ###
if all_tests or validation_test:
    # Index to iterate through the weights
    i = 1

    # Iterate through all weights
    for weight in weights:
        # Index to iterate through the brightness values
        brigthness_index = 0

        # Directory to save the validation results
        weight_eval_dir = eval_dir + "validation/" + os.path.basename(weight) + str(i) + "/"

        # Load the model
        model = YOLO(model=weight, task="detect")

        # List of results
        results = []

        # Iterate through all brightness datasets
        for dataset in brigtness_datasets_list:
            # Directory to save the validation results with the brightness configuration
            weight_brightness_eval_dir = (
                weight_eval_dir + "brightness_" + str(brightness_values[brigthness_index]) + "/"
            )

            print("\nValidating weight: " + weight + " with dataset: " + dataset + "\n")

            # Validate the model with the dataset
            result = model.val(
                data=dataset, save=True, save_json=True, project=weight_brightness_eval_dir
            )

            # Convert the results_dict to a DataFrame
            result_df = pd.DataFrame([result.results_dict])

            # Add the brightness column
            result_df["brightness"] = brightness_values[brigthness_index]

            # Add the DataFrame to the list of results
            results.append(result_df)

            # Increase the brightness index
            brigthness_index += 1

        # After all validations, concatenate all results into a single DataFrame
        results_df = pd.concat(results)

        # Save the result to a CSV file
        results_df.to_csv(weight_eval_dir + "result.csv")
        print("\nSaved result.csv successfully to " + weight_eval_dir + "result.csv")
            
        # Load the result from the CSV file
        df = pd.read_csv(weight_eval_dir + "result.csv")

        ## CALCULATE THE F1 SCORE VS BRIGHTNESS ##
        # Calcule the F1 score
        df["f1_score"] = (
            2
            * (df["metrics/precision(B)"] * df["metrics/recall(B)"])
            / (df["metrics/precision(B)"] + df["metrics/recall(B)"])
        )

        # Get the unique brightness values
        brightness_values = df["brightness"].unique()

        # Calculate the mean F1 score for each brightness value
        mean_f1_scores = df.groupby("brightness")["f1_score"].mean()

        # Create a plot with the F1 score vs brightness
        plt.figure(figsize=(10, 6))
        plt.plot(brightness_values, mean_f1_scores, marker="o")
        plt.xlabel("Brightness")
        plt.ylabel("F-Score")
        plt.title("F-Score as a function of Brightness")
        plt.grid(True)
        
        # Save the plot
        plt.savefig(weight_eval_dir + "f1_score_vs_brightness.png")
        print("\nSaved f1_score_vs_brightness.png successfully to " + weight_eval_dir + "f1_score_vs_brightness.png")
        
        # Save f-score vs brightness to a CSV file
        mean_f1_scores.to_csv(weight_eval_dir + "f1_score_vs_brightness.csv")
        print("\nSaved f1_score_vs_brightness.csv successfully to " + weight_eval_dir + "f1_score_vs_brightness.csv")
        
        ## CALCULATE THE PERCENTAGE OF DETECTED PIECES VS BRIGHTNESS ##
        # Calculate the mean precision for each brightness value
        mean_precision = df.groupby("brightness")["metrics/mAP50(B)"].mean()

        # Create a plot with the precision vs brightness
        plt.figure(figsize=(10, 6))
        plt.plot(brightness_values, mean_precision, marker="o")
        plt.xlabel("Brightness")
        plt.ylabel("% of correctly positioned pieces")
        plt.title("Precision as a function of Brightness")
        plt.grid(True)

        # Save the plot
        plt.savefig(weight_eval_dir + "precision_vs_brightness.png")
        print("\Saved precision_vs_brightness.png successfully to " + weight_eval_dir + "precision_vs_brightness.png")
        
        # Save precision vs brightness to a CSV file
        mean_precision.to_csv(weight_eval_dir + "precision_vs_brightness.csv")
        print("\nSaved precision_vs_brightness.csv successfully to " + weight_eval_dir + "precision_vs_brightness.csv")

        # Increase the index
        i += 1

### PREDICT ON TEST IMAGES WITH DIFFERENT BRIGHTNESS CONFIGURATION ###
# i=1 is the original brightness
# i=9 is the least brightness
if all_tests or prediction_test:
    # Get all test images
    for i in brightness_values:
        # Get all test images
        test_images = glob.glob("data/test/images/*.png")

        # Adjust the brightness of the test images with the value of i
        test_images = ef.adjust_brightness(test_images, i)

        # Iterate through all weights
        for weight in weights:
            # Load the model
            model = YOLO(model=weight, task="detect")
            
            print("\nPredicting with weight: " + weight + " with brightness: " + str(i) + "\n")

            # Predict on test images
            results = model.predict(
                source=test_images,
                imgsz=640,
                save_conf=True,
                save_txt=True,
                save=True,
                project=pred_dir + "brightness_" + str(i) + "/",
            )


### MEAN AND STANDARD DEVIATION CALCULATIONS ###
if all_tests or mean_test or std_test:
    # Get all the files 'results.csv' in the directory
    file_paths = glob.glob(args.directory + "**/results.csv", recursive=True)
    
    # Print the file paths
    if len(file_paths) == 0:
        print("No csv files found in " + args.directory + ". Exiting...")
        exit()
    else:
        print("Files 'results.csv' found:")
        for file_path in file_paths:
            print(file_path)


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
if all_tests or mean_test:
    mean_df = results_df.groupby(results_df.columns[0]).mean()

# Save mean_df to a CSV file
if all_tests or mean_test:
    mean_df.to_csv(eval_dir + "mean_results.csv")
    print("Saved mean_results.csv successfully to" + eval_dir + "mean_results.csv")

if all_tests or std_test:
    std_df = results_df.groupby(results_df.columns[0]).std()

# Save std_df to a CSV file
if all_tests or std_test:
    std_df.to_csv(eval_dir + "std_results.csv")
    print("Saved std_results.csv successfully to" + eval_dir + "std_results.csv")
