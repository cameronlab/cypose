import sys
import os
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog,
    QDoubleSpinBox, QTextEdit, QCheckBox, QSpinBox
)
from PyQt6.QtCore import QThread, pyqtSignal

"""
TODO:
1. Add docstrings for all classes and functions.
2. Improve layout structure for better organization.
3. Add a custom field to allow users to input additional training parameters.
"""

class ScriptRunner(QThread):
    output_signal = pyqtSignal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command
        self.process = None

    def run(self):
        command_str = ' '.join(self.command)
        print(f"Training using command: {command_str}")
        self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        bufsize=1, universal_newlines=True)
        for line in self.process.stdout:
            self.output_signal.emit(line.strip())
        self.process.stdout.close()
        self.process.wait()

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.output_signal.emit("Process terminated.")


class TrainingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.script_thread = None
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

        # GPU and Verbose Checkboxes in a Horizontal Layout
        self.gpu_checkbox = QCheckBox("Use GPU")
        self.verbose_checkbox = QCheckBox("Verbose")
        self.verbose_checkbox.setChecked(False)  # Ensure checkbox is visible and starts unchecked
        gpu_verbose_layout = QHBoxLayout()
        gpu_verbose_layout.addWidget(self.gpu_checkbox)
        gpu_verbose_layout.addWidget(self.verbose_checkbox)

        # Channel Inputs
        self.chan_label = QLabel("Channel 0:")
        self.chan_input = QSpinBox()
        self.chan_input.setRange(0, 5)
        self.chan_input.setValue(0)

        self.chan2_label = QLabel("Channel 1:")
        self.chan2_input = QSpinBox()
        self.chan2_input.setRange(0, 5)
        self.chan2_input.setValue(0)

        # Mask & Image Filters
        self.mask_filter_label = QLabel("Mask Filter:")
        self.mask_filter = QLineEdit()
        self.img_filter_label = QLabel("Image Filter:")
        self.img_filter = QLineEdit()

        # Mean Diameter
        self.diam_mean_label = QLabel("Mean Diameter:")
        self.diam_mean = QSpinBox()
        self.diam_mean.setRange(1, 100)
        self.diam_mean.setValue(30)

        # Batch Size
        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 100)
        self.batch_size.setValue(8)

        # Run Button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_process)

        # Stop Button
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_process)
        self.stop_button.setEnabled(False)

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
            self.chan_label, self.chan_input,
            self.chan2_label, self.chan2_input,
            self.diam_mean_label, self.diam_mean,
            self.batch_size_label, self.batch_size,
            self.mask_filter_label, self.mask_filter,
            self.img_filter_label, self.img_filter,
            self.run_button, self.stop_button, self.output_display
        ]
        for widget in widgets:
            layout.addWidget(widget)

        layout.addLayout(gpu_verbose_layout)  # Add GPU and Verbose checkboxes horizontally

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
        command = ["python", "-m", "cellpose", '--train', "--dir", self.train_folder.text(), "--n_epochs",
                   str(self.epochs_input.value())]
        if self.lr_input.value():
            command.extend(["--learning_rate", str(self.lr_input.value())])
        if self.wd_input.value():
            command.extend(["--weight_decay", str(self.wd_input.value())])
        if self.gpu_checkbox.isChecked():
            command.append("--gpu")
        if self.verbose_checkbox.isChecked():
            command.append("--verbose")
        if self.model_file.text():
            command.extend(["--pretrained_model", self.model_file.text()])
        if self.chan_input.value() or self.chan_input.value() == 0:
            command.extend(["--chan", str(self.chan_input.value())])
        if self.chan2_input.value() or self.chan2_input.value() == 0:
            command.extend(["--chan2", str(self.chan2_input.value())])
        if self.mask_filter.text():
            command.extend(["--mask_filter", self.mask_filter.text()])
        if self.img_filter.text():
            command.extend(["--img_filter", self.img_filter.text()])
        if self.diam_mean.value():
            command.extend(["--diam_mean", str(self.diam_mean.value())])
        if self.batch_size.value():
            command.extend(["--batch_size", str(self.batch_size.value())])
        if self.gpu_checkbox.isChecked():
            command.append("--gpu")
        if self.verbose_checkbox.isChecked():
            command.append("--verbose")

        self.script_thread = ScriptRunner(command)
        self.script_thread.output_signal.connect(self.update_output)
        self.script_thread.start()
        self.stop_button.setEnabled(True)

    def stop_process(self):
        if self.script_thread:
            self.script_thread.stop()
            self.stop_button.setEnabled(False)

    def update_output(self, text):
        self.output_display.append(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = TrainingGUI()
    gui.show()
    sys.exit(app.exec())
