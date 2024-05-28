from PIL import Image
import numpy as np

# Brigthness adjustment function
def adjust_brightness(images, b):
    result = []
    for image_path in images:
        # Load the image
        image = Image.open(image_path)

        imgArray = np.array(image)
        imgArray = imgArray.astype(dtype=float)

        imgArray = imgArray / b 

        image = Image.fromarray(np.uint8(imgArray))
        result.append(image)
    return result