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
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, jaccard_score

class ClassificationComparer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.class_labels = ['Cytoplasm', 'Carboxysome', 'WT', 'Procarboxysome']
        self.metrics_df = pd.DataFrame(columns=['File', 'Precision', 'Recall', 'IoU'])
        self.process_files()

    def process_files(self):
        # Load ground truth CSV file
        ground_truth_file = glob.glob(os.path.join(self.data_dir, "*ground_truth.csv"))
        if not ground_truth_file:
            print("No *ground_truth.csv file found")
            return
        self.ground_truth = pd.read_csv(ground_truth_file[0], names=["frame", "x", "y", "class"], header=0)
        self.ground_truth.dropna(subset=['class'], inplace=True)

        # Crop the ground truth to include only rows up to "frame 70"
        self.ground_truth = self.ground_truth[self.ground_truth['frame'] <= 70]

        # Load predicted CSV files
        predicted_files = glob.glob(os.path.join(self.data_dir, "*predicted.csv"))
        if not predicted_files:
            print("No *predicted.csv files found")
            return

        for predicted_file in predicted_files:
            self.predicted = pd.read_csv(predicted_file, names=["frame", "x", "y", "class"], header=0)
            self.predicted.dropna(inplace=True)
            self.calculate_and_save_metrics(predicted_file)

    def calculate_and_save_metrics(self, predicted_file):
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

        # Generate confusion matrix
        self.createConfusionMatrix(predicted_file, data_dir)

        # Calculate precision, recall, and IoU
        precision = precision_score(self.ground_class, self.predicted_class, average='macro', zero_division=0)
        recall = recall_score(self.ground_class, self.predicted_class, average='macro', zero_division=0)
        iou = jaccard_score(self.ground_class, self.predicted_class, average='macro')


        # Append metrics to DataFrame
        self.metrics_df = self.metrics_df.append({
            'File': os.path.basename(predicted_file),
            'Precision': precision,
            'Recall': recall,
            'IoU': iou
        }, ignore_index=True)


    def createConfusionMatrix(self, predicted_file, data_dir):
        """
        Generates a confusion matrix plot for the given ground truth and predictions.
        """

        # Calculate confusion matrix
        cm = confusion_matrix(self.ground_class, self.predicted_class)

        # Normalize confusion matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm[np.isnan(cm)] = 0  # Set NaNs to 0

        # Define your new labels
        class_labels = ['Cytoplasm', 'Carboxysome', 'WT', 'Procarboxysome']

        # Plot confusion matrix
        print(cm)
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            cm, annot=True, fmt=".3f", linewidths=0.5, square=True, cmap="Blues_r",
            xticklabels=class_labels, yticklabels=class_labels
        )
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        plt.title("Confusion Matrix", size=15)

        # Save confusion matrix plot
        confusion_matrix_filename = os.path.splitext(os.path.basename(predicted_file))[0] + '_confusion_matrix.png'
        plt.savefig(os.path.join(self.data_dir, confusion_matrix_filename))
        plt.close()


    def save_metrics(self):
        # Save metrics to a single CSV file
        metrics_filename = 'all_metrics.csv'
        self.metrics_df.to_csv(os.path.join(self.data_dir, metrics_filename), index=False)

# Example usage
data_dir = 'F:/Widefield/20230428_gssu-roGFP_rbcL-gssu-roGFP_dccmOrbclgssuroGFP_WT_LHLCO2/Cell_Seg3/'
comparer = ClassificationComparer(data_dir)

# Save all metrics to a single CSV file
comparer.save_metrics()