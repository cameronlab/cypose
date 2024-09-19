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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, jaccard_score
import numpy as np
from skimage import io
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns


class SegmentationComparer:
    """
    This class takes in the TIFF files from the segmentation model and the ground truth
    and outputs the precision, recall, IoU metrics, and a ROC plot.
    """

    def __init__(self, data_dir):
        """
        Initializes the class with the ground truth and predicted TIFF files.
        """
        # Load the ground truth and predicted TIFF files
        self.loadSegData(data_dir)

        # Calculate metrics
        precision, recall, iou = self.calculateMetrics()
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"IoU: {iou}")

        # Create a DataFrame
        df = pd.DataFrame({"Precision": [precision], "Recall": [recall], "IoU": [iou]})

        # Save DataFrame to a .csv file
        df.to_csv(os.path.join(data_dir, "metrics.csv"), index=False)

        # Save metrics to csv file
        with open("metrics.csv", "a") as f:
            f.write(f"{precision},{recall},{iou}\n")

        # Generate ROC plot
        self.createConfusionMatrix(data_dir)

    def loadSegData(self, data_dir):
        """
        Loads the ground truth and predicted TIFF files.
        """
        # Get TIFF files
        ground_truth_image = io.imread(os.path.join(data_dir, "*predicted.tif"))
        predicted_image = io.imread(os.path.join(data_dir, "*ground_truth.tif"))

        # Convert images to numpy arrays
        ground_truth_array = np.array(ground_truth_image)
        predicted_array = np.array(predicted_image)

        # Flatten arrays to 1D
        self.ground_truth = ground_truth_array.flatten()
        self.predicted = predicted_array.flatten()

        # Convert to binary
        self.ground_truth = np.where(self.ground_truth > 0, 1, 0)
        self.predicted = np.where(self.predicted > 0, 1, 0)

    def calculateMetrics(self):
        """
        Calculates precision, recall, and IoU metrics.
        """
        print("Calculating metrics...")
        precision = precision_score(self.ground_truth, self.predicted)
        recall = recall_score(self.ground_truth, self.predicted)
        iou = jaccard_score(self.ground_truth, self.predicted)

        return precision, recall, iou

    def createConfusionMatrix(self, data_dir):
        """
        Generates a confusion matrix plot for the given ground truth and predictions.
        """
        print("Creating confusion matrix...")
        # Calculate confusion matrix
        cm = confusion_matrix(self.ground_truth, self.predicted)

        # Normalize confusion matrix
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Define your new labels
        class_labels = ["Background", "Cell"]

        # Plot confusion matrix
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            square=True,
            cmap="Blues_r",
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        plt.title("Confusion Matrix", size=15)
        plt.show()

        # save confusion matrix as pdf
        plt.savefig(os.path.join(data_dir, "confusion_matrix.pdf"))

    def createROCPlot(self, data_dir):
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
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")

        # save ROC curve as png
        plt.savefig(os.path.join(data_dir, "ROC.png"))
        plt.show()

        # ROC curve code taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html


# Directory containing the ground truth and predicted TIFF files
data_dir = "E:\ZachML\seg_test"

# Create an instance of the SegmentationComparer class
comparer = SegmentationComparer(data_dir)


# model_comparisons.py ends here
