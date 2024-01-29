# model_comparisons.py --- Create NN Comparison Figures and Metrics
#
# Filename: model_comparisons.py
# Author: Zach Maas and Clair Huffine
# Created: Fri Jan 26 2024
#

# Commentary:
#
# This file contains functions to generate comparison plots and standard 
# metrics for cell segmentation neural network.
# Specifically it contains functions to generate confusion matrices and 
# ROC curves and the metrics precision, recall, and IoU.

# Code:

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, jaccard_score
import seaborn as sns
import numpy as np
from tifffile import imread
import os
import glob

class ClassificationComparer: 
    """
    This class takes in the tensor from the trained classifier model and the ground truth 
    tensor and outputs the confusion matrix and ROC curve.
    """
    def __init__(self, ground_truth_tensor, predicted_tensor):
        """
        Initializes the class with the ground truth and predicted tensors.
        """
        self.ground_truth = ground_truth_tensor.flatten()
        self.predicted = predicted_tensor.flatten()

    def calculate_metrics(self):
        """
        Calculates precision, recall, and IoU metrics.
        """
        precision = precision_score(self.ground_truth, self.predicted)
        recall = recall_score(self.ground_truth, self.predicted)
        iou = jaccard_score(self.ground_truth, self.predicted)

        return precision, recall, iou

    def create_confusion_matrix(self):
        """
        Generates a confusion matrix plot for the given ground truth and predictions.
        """
        # Calculate confusion matrix
        cm = confusion_matrix(self.ground_truth, self.predicted)

        # Normalize confusion matrix
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Plot confusion matrix
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=0.5, square=True, cmap="Blues_r")
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        plt.title("Confusion Matrix", size=15)
        plt.show()

        #save confusion matrix as png
        plt.savefig(os.path.join(f"{self.predicted_tensor}_confusion_matrix.png"))

        # Metrics code taken from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html


# Assuming you have ground_truth_tensor and predicted_tensor
ground_truth_tensor = []#TODO ADD tensor code 
predicted_tensor = []#TODO Add tensor code 

# Create an instance of the ClassificationComparer class
comparer = ClassificationComparer(ground_truth_tensor, predicted_tensor)

# Calculate metrics
precision, recall, iou = comparer.calculate_metrics()
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"IoU: {iou}")

# Generate confusion matrix
comparer.create_confusion_matrix()

class SegmentationComparer:
    """
    This class takes in the TIFF files from the segmentation model and the ground truth 
    and outputs the precision, recall, IoU metrics, and a ROC plot.
    """
    def __init__(self, ground_truth_file, predicted_file):
        """
        Initializes the class with the ground truth and predicted TIFF files.
        """
        self.ground_truth = imread(ground_truth_file).flatten()
        self.predicted = imread(predicted_file).flatten()

    def calculate_metrics(self):
        """
        Calculates precision, recall, and IoU metrics.
        """
        precision = precision_score(self.ground_truth, self.predicted)
        recall = recall_score(self.ground_truth, self.predicted)
        iou = jaccard_score(self.ground_truth, self.predicted)

        return precision, recall, iou

    def create_ROC_Plot(self):
        """
        Generates a ROC plot for the given ground truth and predictions.
        """
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(self.ground_truth, self.predicted)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(
            fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()

        #save ROC curve as png
        plt.savefig(os.path.join(f"{self.predicted_file}_ROC.png"))

        # ROC curve code taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html


# Directory containing the ground truth and predicted TIFF files
ground_truth_dir = "path_to_ground_truth_dir"
predicted_dir = "path_to_predicted_dir"

# Get list of TIFF files
ground_truth_files = sorted(glob.glob(os.path.join(ground_truth_dir, '*.tif')))
predicted_files = sorted(glob.glob(os.path.join(predicted_dir, '*.tif')))

# Iterate over TIFF files
for gt_file, pred_file in zip(ground_truth_files, predicted_files):
    # Full path to the files
    gt_file_path = os.path.join(ground_truth_dir, gt_file)
    pred_file_path = os.path.join(predicted_dir, pred_file)

    # Create an instance of the SegmentationComparer class
    comparer = SegmentationComparer(gt_file_path, pred_file_path)

    # Calculate metrics
    precision, recall, iou = comparer.calculate_metrics()
    print(f"File: {gt_file}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"IoU: {iou}")

    # Save metrics to csv file
    with open("metrics.csv", "a") as f:
        f.write(f"{pred_file},{precision},{recall},{iou}\n")

    # Generate ROC plot
    comparer.create_ROC_Plot()


# model_comparisons.py ends here
