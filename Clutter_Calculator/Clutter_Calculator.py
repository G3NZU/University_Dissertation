import cv2
import numpy as np

# Load clutter mask (white = clutter, black = background)
mask = cv2.imread('Clutter_mask3.png', cv2.IMREAD_GRAYSCALE)

# Count white pixels (clutter)
clutter_pixels = cv2.countNonZero(mask)
total_pixels = mask.shape[0] * mask.shape[1]

clutter_percentage = (clutter_pixels / total_pixels) * 100
print(f"Clutter covers {clutter_percentage:.2f}% of the image.")

#   Clutter Level Classification:
#   Level 1 of clutter should cover ~25% (-+10%)
#   Level 2 of clutter should cover ~50% (-+10%)
#   Level 3 of clutter should cover ~75% (-+10%)