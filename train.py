import ultralytics
from ultralytics import YOLO
import argparse_functions as af

# Warnings are not shown
import warnings

warnings.filterwarnings("ignore")

# Return a human-readable YOLO software and hardware summary.
ultralytics.checks()

# Get arguments
args = af.argparse_train()

dataset_path = args.dataset
model_path = args.model
epochs = int(args.epochs)
batch = int(args.batch)
lr0 = float(args.learning0)
lrf = float(args.learningf)
plots = args.plots

# Load the model
if model_path is None:
    model = YOLO("yolov8s.pt")  # Load a pretrained model (recommended for training)
else:
    model = YOLO(model_path)  # Load a custom model

# Trains the model
results = model.train(
    data=dataset_path, epochs=epochs, batch=batch, lr0=lr0, lrf=lrf, plots=plots
)

# Print that the training has finished
print("Training finished")
