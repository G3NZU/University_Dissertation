#   Calculating clutter levels

##  Objective
Determine the **level of clutter** in the shelf by calculating the **percentage of pixels** covered in the image with clutter.

## Levels of clutter
Different levels of clutter based by pixel percentage covered with clutter items:
-   **Level 0**: There is not clutter.
-   **Level 1**: ~25% (-+10%) of clutter in the shelf.
-   **Level 2**: ~50% (-+10%) of clutter in the shelf.
-   **Level 3**: ~75% (-+10%) of clutter in the shelf.

---

## Process

### Step 1: create clutter mask in GIMP

*First of all, I had to take multiple images with different amount of clutter in the shelf*

1.  Open the image in GIMP.
2.  Use selection tool (Free select in GIMP) to manually isolate clutter items.
3.  With the isolated item selected, create transparent layer (only create one layer for all isolated items).
4.  Fill the isolated clutter item with white (Fill tool in GIMP) in the new layer.
5.  Fill rest of the image with black (Fill tool in GIMP).
6.  Export final image as PNG.

    -   **White pixels = clutter.**
    -   **Black pixels = background.**

### Step 2: python code to calculate clutter coverage

**My Python Script**:
```python
import cv2
import numpy as np

# Load clutter mask (white = clutter, black = background)
mask = cv2.imread('Clutter_mask1.png', cv2.IMREAD_GRAYSCALE)

# Count white pixels (clutter)
clutter_pixels = cv2.countNonZero(mask)
total_pixels = mask.shape[0] * mask.shape[1]

clutter_percentage = (clutter_pixels / total_pixels) * 100
print(f"Clutter covers {clutter_percentage:.2f}% of the image.")

#   Clutter Level Classification:
#   Level 1 of clutter should cover ~25% (-+10%)
#   Level 2 of clutter should cover ~50% (-+10%)
#   Level 3 of clutter should cover ~75% (-+10%)
```