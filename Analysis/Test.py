from ultralytics import YOLO
import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import re

#   Function to sort files
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# Paths
model_paths = {
    'YOLOv8': 'c:/Users/vagfl/Python Uni/Dissertation/Analysis/YOLOv8.pt',
    'YOLOv11': 'c:/Users/vagfl/Python Uni/Dissertation/Analysis/YOLOv11.pt'
}

test_images = 'c:/Users/vagfl/Python Uni/Dissertation/Analysis/test_images'
ground_truth_labels = 'c:/Users/vagfl/Python Uni/Dissertation/Analysis/ground_truth'
output_csv = 'results.csv'
output_images_v8 = 'output_images_v8'
output_images_v11 = 'output_images_v11'

# Create output directories if they don't exist
os.makedirs(output_images_v8, exist_ok=True)
os.makedirs(output_images_v11, exist_ok=True)

# Load models
models = {name: YOLO(path) for name, path in model_paths.items()}

# Function to calculate IoU
def calculate_iou(pred_box, gt_box):
    x_min_pred, y_min_pred, x_max_pred, y_max_pred = pred_box
    x_min_gt, y_min_gt, x_max_gt, y_max_gt = gt_box

    # Calculate intersection
    x_min_inter = max(x_min_pred, x_min_gt)
    y_min_inter = max(y_min_pred, y_min_gt)
    x_max_inter = min(x_max_pred, x_max_gt)
    y_max_inter = min(y_max_pred, y_max_gt)

    intersection_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # Calculate union
    area_pred = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
    area_gt = (x_max_gt - x_min_gt) * (y_max_gt - y_min_gt)
    union_area = area_pred + area_gt - intersection_area

    # IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

# Function to convert YOLO format to [x_min, y_min, x_max, y_max]
def convert_yolo_to_xyxy(yolo_box, img_width, img_height):
    if len(yolo_box) !=5:
        print(f"Skipping invalid box: {yolo_box}")  #   Log invalid box
        return None
    class_id, x_center, y_center, width, height = yolo_box
    # Convert normalized to absolute values
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)
    return [x_min, y_min, x_max, y_max]

# Output data list
data = []

# Iterate over folders
for clutter_level in sorted(os.listdir(test_images), key=natural_sort_key):
    level_path = os.path.join(test_images, clutter_level)
    gt_level_path = os.path.join(ground_truth_labels, clutter_level)
    if not os.path.isdir(level_path):
        continue

    # Iterate over each image
    for img_file in sorted(os.listdir(level_path), key=natural_sort_key):
        if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):  # Check for valid image files
            continue
        img_path = os.path.join(level_path, img_file)
        gt_file_name = Path(img_file).stem + '.txt'
        gt_path = os.path.join(gt_level_path, gt_file_name)

        # Read the image for drawing bounding boxes
        original_img = cv2.imread(img_path)
        img_height, img_width = original_img.shape[:2]

        # Load ground truth
        with open(gt_path, 'r') as f:
            gt_boxes = [list(map(float, line.strip().split())) for line in f.readlines()]

        # Convert ground truth boxes to [x_min, y_min, x_max, y_max]
        gt_boxes_converted = []
        for gt_box in gt_boxes:
            converted_box = convert_yolo_to_xyxy(gt_box, img_width, img_height)
            if converted_box:
                gt_boxes_converted.append(converted_box)

        ious = []

        for model_name, model in models.items():

            # Reset counters for each model
            tp, fp, fn = 0, 0, 0

            # Run inference
            results = model(img_path)[0]
            detections = results.boxes

            # Confidence scores and bounding boxes
            confidences = detections.conf if detections is not None else []
            num_detections = len(confidences)
            avg_confidence = float(confidences.mean()) if num_detections > 0 else 0.0

            # Process the predictions and calculate metrics
            pred_boxes = detections.xyxy.tolist()  # List of [x_min, y_min, x_max, y_max, confidence]
            iou_scores = []
            matched_gt_boxes = [False] * len(gt_boxes_converted)

            # Copy of original image to avoid overlapping boxes
            img = original_img.copy()

            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1

                # Compare prediction with all ground truth boxes
                for i, gt_box in enumerate(gt_boxes_converted):
                    if not matched_gt_boxes[i]:
                        iou = calculate_iou(pred_box[:4], gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i

                if best_iou > 0.5:
                    tp += 1
                    matched_gt_boxes[best_gt_idx] = True
                    iou_scores.append(best_iou)
                    color = (0, 255, 0)  # Green for TP
                else:
                    fp += 1
                    color = (0, 0, 255)  # Red for FP

            # Draw the predicted box
            x_min, y_min, x_max, y_max = map(int, pred_box[:4])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(img, f'{best_iou:.2f}', (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Count false negatives (unmatched ground truth boxes)
            fn = matched_gt_boxes.count(False)

            # Calculate Precision, Recall, F1 score
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            # Save the image with bounding boxes
            if model_name == 'YOLOv8':
                output_img_path = os.path.join(output_images_v8, f"{img_file}")
            elif model_name == 'YOLOv11':
                output_img_path = os.path.join(output_images_v11, f"{img_file}")

            cv2.imwrite(output_img_path, img)

            # Save record in data
            data.append({
                'Model': model_name,
                'Clutter_Level': clutter_level,
                'Image_Name': img_file,
                'Num_Detections': num_detections,
                'Avg_Confidence': avg_confidence,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1_score,
                'Avg_IOU': np.mean(iou_scores) if iou_scores else 0.0
            })

# Save the results to a CSV file
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"Saved results to {output_csv}")
