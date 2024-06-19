import datetime
import shutil
from pathlib import Path
from collections import Counter

import gc

import yaml
import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import KFold
import argparse_functions as af

# Get the arguments
args = af.argparse_crossvalidation()

dataset = args.dataset
model_path = args.model
k_folds = int(args.folds)
epochs = int(args.epochs)
batch = int(args.batch)
lr0 = float(args.learning0)
lrf = float(args.learningf)
plots = args.plots

# Load the dataset
dataset_path = Path(dataset)
dataset_path = dataset_path.parent.absolute() # Get the data directory
# Get the labels
labels = []
labels.extend(sorted((dataset_path / "train" / "labels").rglob("*.txt")))
labels.extend(sorted((dataset_path / "val" / "labels").rglob("*.txt")))

# Get the classes
yaml_file = dataset
with open(yaml_file, "r", encoding="utf8") as y:
    classes = yaml.safe_load(y)["names"]
cls_idx = sorted(classes.keys())

# Create a DataFrame with the labels
indx = [l.stem for l in labels]
labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

# Count the number of labels for each class
for label in labels:
    lbl_counter = Counter()

    with open(label, "r") as lf:
        lines = lf.readlines()

    for l in lines:
        # classes for YOLO label uses integer at first position of each line
        lbl_counter[int(l.split(" ")[0])] += 1

    labels_df.loc[label.stem] = lbl_counter

labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`

# Divide the dataset into k splits
ksplit = k_folds
kf = KFold(
    n_splits=ksplit, shuffle=True, random_state=20
)  # setting random_state for repeatable results

kfolds = list(kf.split(labels_df))

# Create a DataFrame with the folds
folds = [f"split_{n}" for n in range(1, ksplit + 1)]
folds_df = pd.DataFrame(index=indx, columns=folds)

for idx, (train, val) in enumerate(kfolds, start=1):
    folds_df[f"split_{idx}"].loc[labels_df.iloc[train].index] = "train"
    folds_df[f"split_{idx}"].loc[labels_df.iloc[val].index] = "val"

# Calculate the distribution of the labels in each fold
fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()

    # To avoid division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals / (train_totals + 1e-7)
    fold_lbl_distrb.loc[f"split_{n}"] = ratio

supported_extensions = [".jpg", ".jpeg", ".png"]

# Initialize an empty list to store image file paths
images = []

# Loop through supported extensions and gather image files
for ext in supported_extensions:
    images.extend(sorted((dataset_path / "train" / "images").rglob(f"*{ext}")))
    images.extend(sorted((dataset_path / "val" / "images").rglob(f"*{ext}")))
    images.sort()

# Create the necessary directories and dataset YAML files
save_path = Path(
    dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val"
)
save_path.mkdir(parents=True, exist_ok=True)
ds_yamls = []

for split in folds_df.columns:
    # Create directories
    split_dir = save_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # Create dataset YAML files
    dataset_yaml = split_dir / f"{split}_dataset.yaml"
    ds_yamls.append(dataset_yaml)

    with open(dataset_yaml, "w") as ds_y:
        yaml.safe_dump(
            {
                "path": split_dir.as_posix(),
                "train": "train/images",
                "val": "val/images",
                "names": classes,
            },
            ds_y,
        )

# Copy the images and labels to the new directories
for image, label in zip(images, labels):
    for split, k_split in folds_df.loc[image.stem].items():
        # Destination directory
        img_to_path = save_path / split / k_split / "images"
        lbl_to_path = save_path / split / k_split / "labels"

        # Copy image and label files to new directory (SamefileError if file already exists)
        shutil.copy(image, img_to_path / image.name)
        shutil.copy(label, lbl_to_path / label.name)

# Save the dataframes to a CSV file
folds_df.to_csv(save_path / "kfold_datasplit.csv")
fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")

# Training
if model_path is None:
    model = YOLO("yolov8s.pt", task="detect") # Load a pretrained model
else:
    model = YOLO(model_path, task="detect") # Load a custom model

results = {}

# Train the model for each split
for k in range(ksplit):
    gc.collect()
    dataset_yaml = ds_yamls[k]
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch,
        lr0=lr0,
        lrf=lrf,
        plots=plots,
    )
    results[k] = model.metrics  # save output metrics for further analysis
