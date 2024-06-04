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
