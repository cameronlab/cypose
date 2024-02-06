# classification_comparisons.py --- Create NN Comparison Figures and Metrics
#
# Filename: classification_comparisons.py
# Author: Zach Maas and Clair Huffine
# Created: Fri Jan 26 2024
#

# Commentary:
#
# This file contains functions to generate comparison plots and standard
# metrics for cell classification.
# Specifically it contains functions to generate confusion matrices and
# the metrics precision, recall, and IoU.

# Code:

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, jaccard_score
import seaborn as sns
import numpy as np
import os
import glob
import pandas as pd


class ClassificationComparer:
    """
    This class takes in the tensor from the trained classifier model and the ground truth
    tensor and outputs the confusion matrix and ROC curve.
    """

    def __init__(self, data_dir):
        """
        Initializes the class with the ground truth and predicted csv and runs the comparison.
        """
        # Load the ground truth and predicted csv files
        self.loadClassData(data_dir)

        # Convert string classes to numeric values for later comparison
        self.ground_truth = self.ground_truth.replace("cyto", 1)
        self.ground_truth = self.ground_truth.replace("csome", 2)
        self.ground_truth = self.ground_truth.replace("WT", 3)
        self.ground_truth = self.ground_truth.replace("pcsome", 4)

        self.predicted = self.predicted.replace("cyto", 1)
        self.predicted = self.predicted.replace("csome", 2)
        self.predicted = self.predicted.replace("WT", 3)
        self.predicted = self.predicted.replace("pcsome", 4)

        # Select the 'class' column
        self.ground_class = self.ground_truth["class"]
        self.predicted_class = self.predicted["class"]

        # Calculate metrics
        precision, recall, iou = self.calculateMetrics()
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"IoU: {iou}")

        # Create a DataFrame
        df = pd.DataFrame({
            'Precision': [precision],
            'Recall': [recall],
            'IoU': [iou]
        })

        # Save DataFrame to a .csv file
        df.to_csv(os.path.join(data_dir, 'metrics.csv'), index=False)

        # Generate confusion matrix
        self.createConfusionMatrix(data_dir)

    def loadClassData(self, data_dir):
        """
        Loads the data from the given directory.
        """
        # Load csv files from the given directory
        # Find all CSV files that have 'ground_truth.csv' in their name
        csv_files = glob.glob(os.path.join(data_dir, "*ground_truth.csv"))

        # Read the first CSV file found
        if csv_files:
            self.ground_truth = pd.read_csv(csv_files[0])
        else:
            print("No *ground_truth.csv file found")

        # Find all the CSV files that have 'predicted.csv' in their name
        csv_files = glob.glob(os.path.join(data_dir, "*predicted.csv"))

        # Read the first CSV file found
        if csv_files:
            self.predicted = pd.read_csv(csv_files[0])
        else:
            print("No *predicted.csv file found")

    def calculateMetrics(self):
        """
        Calculates precision, recall, and IoU metrics.
        """
        precision = precision_score(
            self.ground_class, self.predicted_class, average="micro"
        )
        recall = recall_score(self.ground_class, self.predicted_class, average="micro")
        iou = jaccard_score(self.ground_class, self.predicted_class, average="micro")

        return precision, recall, iou

    def createConfusionMatrix(self, data_dir):
        """
        Generates a confusion matrix plot for the given ground truth and predictions.
        """
        # Calculate confusion matrix
        cm = confusion_matrix(self.ground_class, self.predicted_class)

        # Normalize confusion matrix
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Define your new labels
        class_labels = ['Cytoplasm', 'Carboxysome', 'WT', 'Procarboxysome']

        # Plot confusion matrix
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            cm, annot=True, fmt=".3f", linewidths=0.5, square=True, cmap="Blues_r",
            xticklabels=class_labels, yticklabels=class_labels
        )
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        plt.title("Confusion Matrix", size=15)
        plt.show()

        # save confusion matrix as png
        plt.savefig(os.path.join(data_dir, "confusion_matrix.png"))

        # Metrics code taken from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html


# Path to classification model ground truth and predicted csv files
data_dir = "E:\ZachML\class_test"

# Create an instance of the ClassificationComparer class
comparer = ClassificationComparer(data_dir)

# classification_comparisons.py ends here
