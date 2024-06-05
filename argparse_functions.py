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