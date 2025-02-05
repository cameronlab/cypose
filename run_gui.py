import sys
import os
import subprocess

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog,
    QDoubleSpinBox, QTextEdit, QCheckBox, QSpinBox
)


def run_script(train_folder, n_epochs, learning_rate=None, weight_decay=None, use_gpu=None, pretrained_model=None,
               chan0=None, chan1=None, verbose=None, mask_filter=None, img_filter=None, mean_diameter=None,
               batch_size=None):
    """
    Running the script.
    """
    if not train_folder or not str(n_epochs).isdigit():
        return "Error: Please fill in all required fields."

    command = ["python3", "-m", "cellpose", "--dir", train_folder, "--n_epochs", str(n_epochs)]
    if learning_rate:
        command.extend(["--learning_rate", str(learning_rate)])
    if weight_decay:
        command.extend(["--weight_decay", str(weight_decay)])
    if use_gpu:
        command.append("--gpu")
    if pretrained_model:
        command.extend(["--pretrained_model", pretrained_model])
    if chan0 is not None:
        command.extend(["--chan0", str(chan0)])
    if chan1 is not None:
        command.extend(["--chan1", str(chan1)])
    if verbose:
        command.append("--verbose")
    if mask_filter:
        command.extend(["--mask_filter", mask_filter])
    if img_filter:
        command.extend(["--img_filter", img_filter])
    if mean_diameter is not None:
        command.extend(["--mean_diameter", str(mean_diameter)])
    if batch_size is not None:
        command.extend(["--batch_size", str(batch_size)])

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"


class TrainingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Train directory
        self.train_folder_label = QLabel("Select Train Image Directory:")
        self.train_folder_button = QPushButton("Browse")
        self.train_folder_button.clicked.connect(self.select_train_folder)
        self.train_folder = QLineEdit()

        # Number of Epochs Input
        self.epochs_label = QLabel("Number of Epochs:")
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 5000)
        self.epochs_input.setValue(100)

        # Model File Selection
        self.model_label = QLabel("Select Pretrained Model File:")
        self.model_button = QPushButton("Browse")
        self.model_button.clicked.connect(self.select_model_file)
        self.model_file = QLineEdit()

        # Learning Rate Input
        self.lr_label = QLabel("Learning Rate:")
        self.lr_input = QDoubleSpinBox()
        self.lr_input.setDecimals(6)
        self.lr_input.setRange(0.000001, 1.0)
        self.lr_input.setSingleStep(0.000001)
        self.lr_input.setValue(0.001)

        # Weight Decay Input
        self.wd_label = QLabel("Weight Decay:")
        self.wd_input = QDoubleSpinBox()
        self.wd_input.setDecimals(6)
        self.wd_input.setRange(0.0001, 1.0)
        self.wd_input.setSingleStep(0.0001)
        self.wd_input.setValue(0.0001)

        # GPU Checkbox
        self.gpu_checkbox = QCheckBox("Use GPU")

        # Channels
        self.chan0_label = QLabel("Channel 0")
        self.chan0_input = QSpinBox()
        self.chan0_input.setRange(0, 5)
        self.chan0_input.setValue(0)

        self.chan1_label = QLabel("Channel 1")
        self.chan1_input = QSpinBox()
        self.chan1_input.setRange(0, 5)
        self.chan1_input.setValue(0)

        # Verbose Checkbox
        self.verbose_checkbox = QCheckBox("Verbose")

        # Mask & Image Filters
        self.mask_filter_label = QLabel("Mask Filter:")
        self.mask_filter = QLineEdit()
        self.img_filter_label = QLabel("Image Filter:")
        self.img_filter = QLineEdit()

        # Mean Diameter
        self.mean_diameter_label = QLabel("Mean Diameter:")
        self.mean_diameter = QSpinBox()
        self.mean_diameter.setRange(1, 100)
        self.mean_diameter.setValue(30)

        # Batch Size
        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 100)
        self.batch_size.setValue(8)

        # Run Button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_process)

        # Output Display
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)

        # Adding widgets to layout
        widgets = [
            self.train_folder_label, self.train_folder, self.train_folder_button,
            self.epochs_label, self.epochs_input,
            self.model_label, self.model_file, self.model_button,
            self.lr_label, self.lr_input,
            self.wd_label, self.wd_input,
            self.chan0_label, self.chan0_input,
            self.chan1_label, self.chan1_input,
            self.mean_diameter_label, self.mean_diameter,
            self.batch_size_label, self.batch_size,
            self.gpu_checkbox, self.verbose_checkbox,
            self.mask_filter_label, self.mask_filter,
            self.img_filter_label, self.img_filter,
            self.run_button, self.output_display
        ]
        for widget in widgets:
            layout.addWidget(widget)

        self.setLayout(layout)
        self.setWindowTitle("Training Script GUI")

    def select_train_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Train Folder")
        if folder:
            self.train_folder.setText(folder)

    def select_model_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Model File")
        if file:
            self.model_file.setText(file)

    def run_process(self):
        result = run_script(
            self.train_folder.text(), self.epochs_input.value(), self.lr_input.value(), self.wd_input.value(),
            self.gpu_checkbox.isChecked(), self.model_file.text(), self.chan0_input.value(), self.chan1_input.value(),
            self.verbose_checkbox.isChecked(), self.mask_filter.text(), self.img_filter.text(),
            self.mean_diameter.value(), self.batch_size.value()
        )
        self.output_display.setText(result)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = TrainingGUI()
    gui.show()
    sys.exit(app.exec())