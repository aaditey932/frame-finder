import cv2
import numpy as np
from PIL import Image

def crop_largest_rect_from_pil(pil_image, area_threshold=20000):
    """
    Crop the largest rectangular object (e.g., a painting) from a PIL image.

    Args:
        pil_image (PIL.Image): Input PIL image.
        area_threshold (int): Minimum contour area to be considered as the painting.

    Returns:
        PIL.Image: Cropped painting image.
    """
    # Convert PIL to OpenCV format (RGB → BGR)
    image = np.array(pil_image)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    original = image_bgr.copy()

    # Grayscale and preprocess
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (13, 13), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Dilation to connect parts of the painting
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    dilate = cv2.dilate(thresh, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
    dilate = cv2.dilate(dilate, vertical_kernel, iterations=2)

    # Find contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Find the largest valid contour
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area > area_threshold:
            x, y, w, h = cv2.boundingRect(c)
            cropped = original[y:y+h, x:x+w]
            # Convert back to PIL
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            return cropped_pil

    raise Exception("No suitable rectangular region found.")
