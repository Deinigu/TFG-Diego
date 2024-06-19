import argparse
from textwrap import dedent


# Argument parser function for the main program
def argparse_main():
    # Argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Define the arguments
    # Image path
    parser.add_argument(
        "-i",
        "--image",
        required=True,
        metavar="path_to_image",
        help=dedent(
            """\
            [REQUIRED] Path to the image to process.
            """
        ),
    )

    # Model path
    parser.add_argument(
        "-m",
        "--model",
        metavar="model_name",
        required=False,
        default=None,
        help=dedent(
            """\
            Name of the model in directory workspace/models/.
            """
        ),
    )

    # Debug
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Show debug images and information.",
    )

    # Unicode output
    parser.add_argument(
        "-utf8",
        "--utf8",
        action="store_true",
        default=False,
        help="Output the results in utf-8 instead of unicode.",
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


# Argument parser function for the evaluation program
def argparse_eval():
    # Argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Define the arguments
    # Path of the directory with all the models and results.csv
    parser.add_argument(
        "-d",
        "--directory",
        metavar="path_to_directory",
        required=True,
        default=None,
        help=dedent(
            """\
            [REQUIRED] Path to the directory with all the models to test and all the results.csv files.
            The program will search for all the files in the directory and subdirectories.
            It's important to have each model and results.csv file in a different subdirectory.
            """
        ),
    )

    # Execute all the tests
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=False,
        help="[DEFAULT] Run all the tests.",
    )

    # Execute brightness validation test
    parser.add_argument(
        "-v",
        "--validation",
        action="store_true",
        default=False,
        help="Run the brightness validation.",
    )

    # Execute brightness prediction test
    parser.add_argument(
        "-p",
        "--prediction",
        action="store_true",
        default=False,
        help="Run the brightness prediction.",
    )

    # Execute mean of results test
    parser.add_argument(
        "-m",
        "--mean",
        action="store_true",
        default=False,
        help="Run the mean of results test.",
    )

    # Execute standard deviation of results test
    parser.add_argument(
        "-s",
        "--std",
        action="store_true",
        default=False,
        help="Run the standard deviation of results test.",
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


# Argument parser function for the training program
def argparse_train():
    # Argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Define the arguments
    # Path of the directory with the dataset.yml
    parser.add_argument(
        "-d",
        "--dataset",
        metavar="path_to_file",
        required=True,
        default=None,
        help=dedent(
            """\
            [REQUIRED] Path to the YAML file of the dataset.
            """
        ),
    )

    # Path of the model to train
    parser.add_argument(
        "-m",
        "--model",
        metavar="path_to_model",
        required=False,
        default=None,
        help=dedent(
            """\
            Path to the model to train. If you don't have a model, it will create a new one.
            """
        ),
    )

    # Epochs
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="number_of_epochs",
        required=False,
        default=30,
        help=dedent(
            """\
            Number of epochs to train the model.
            """
        ),
    )

    # Batch size
    parser.add_argument(
        "-b",
        "--batch",
        metavar="batch_size",
        required=False,
        default=-1,
        help=dedent(
            """\
            Batch size for the training.
            """
        ),
    )

    # Initial learning rate
    parser.add_argument(
        "-l0",
        "--learning0",
        metavar="initial_learning_rate",
        required=False,
        default=0.01,
        help=dedent(
            """\
            Initial learning rate for the training.
            """
        ),
    )

    # Final learning rate
    parser.add_argument(
        "-lf",
        "--learningf",
        metavar="final_learning_rate",
        required=False,
        default=0.01,
        help=dedent(
            """\
            Final learning rate for the training.
            """
        ),
    )

    # Save plots
    parser.add_argument(
        "-p",
        "--plots",
        action="store_true",
        default=False,
        help="Saves plots of the training.",
    )

    # Parse the arguments
    args = parser.parse_args()

    return args

# Argument parser function for the cross-validation program
def argparse_crossvalidation():
    # Argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Define the arguments
    # Path of the directory with the dataset.yml
    parser.add_argument(
        "-d",
        "--dataset",
        metavar="path_to_file",
        required=True,
        default=None,
        help=dedent(
            """\
            [REQUIRED] Path to the YAML file of the dataset.
            """
        ),
    )

    # Uses k-folds
    parser.add_argument(
        "-k",
        "--folds",
        metavar="number_of_folds",
        required=True,
        default=0,
        help=dedent(
            """\
            [REQUIRED] Number of k-folds to create and/or use.
            """
        ),
    )

    # Path of the model to train
    parser.add_argument(
        "-m",
        "--model",
        metavar="path_to_model",
        required=False,
        default=None,
        help=dedent(
            """\
            Path to the model to train. If you don't have a model, it will create a new one.
            """
        ),
    )

    # Epochs
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="number_of_epochs",
        required=False,
        default=30,
        help=dedent(
            """\
            Number of epochs to train the model.
            """
        ),
    )

    # Batch size
    parser.add_argument(
        "-b",
        "--batch",
        metavar="batch_size",
        required=False,
        default=-1,
        help=dedent(
            """\
            Batch size for the training.
            """
        ),
    )

    # Initial learning rate
    parser.add_argument(
        "-l0",
        "--learning0",
        metavar="initial_learning_rate",
        required=False,
        default=0.01,
        help=dedent(
            """\
            Initial learning rate for the training.
            """
        ),
    )

    # Final learning rate
    parser.add_argument(
        "-lf",
        "--learningf",
        metavar="final_learning_rate",
        required=False,
        default=0.01,
        help=dedent(
            """\
            Final learning rate for the training.
            """
        ),
    )

    # Save plots
    parser.add_argument(
        "-p",
        "--plots",
        action="store_true",
        default=False,
        help="Saves plots of the training.",
    )

    # Parse the arguments
    args = parser.parse_args()

    return args
