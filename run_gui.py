import sys
import subprocess
from PyQt6.QtWidgets import (
    QTabWidget,
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog,
    QDoubleSpinBox, QTextEdit, QCheckBox, QSpinBox, QGridLayout
)
from PyQt6.QtCore import QThread, pyqtSignal

"""
TODO:
1. Add docstrings for all classes and functions.
2. Fill segmenting tab with correct widgets
3. Fix display, make sure they are not duplicated. (alternatively keep one display and have different tabs for the task)
"""


class ScriptRunner(QThread):
    output_signal = pyqtSignal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command
        self.process = None

    def run(self):
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


class CyposeGUI(QTabWidget):
    def __init__(self):
        super().__init__()
        self.script_thread = None
        self.initTabs()

    def initTabs(self):
        self.setWindowTitle("Cypose GUI")
        self.setGeometry(100, 100, 1000, 600)

        # Create tabs
        self.tabs = QTabWidget()
        self.training_tab = QWidget()
        self.segmenting_tab = QWidget()

        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.segmenting_tab, "Segmenting")

        # Initialize layouts for both tabs
        self.initTrainingTab()
        self.initSegmentingTab()

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def initTrainingTab(self):
        """Initialize the GUI layout and widgets."""
        layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        params_layout = QGridLayout()

        # Custom parameter input
        self.custom_params_label = QLabel("Additional Training Parameters:")
        self.custom_params_input = QLineEdit()

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

        # GPU and Verbose Checkboxes
        self.gpu_checkbox = QCheckBox("Use GPU")
        self.verbose_checkbox = QCheckBox("Verbose")
        self.gpu_checkbox.setChecked(False)  # Ensure checkbox is visible and starts unchecked
        self.verbose_checkbox.setChecked(True)  # Ensure checkbox is visible and starts checked

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

        # Stop Button
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_process)
        self.stop_button.setEnabled(False)

        # Output Display
        self.output_display = QTextEdit()
        #self.output_display.setMinimumWidth(600)
        #self.output_display.setMinimumHeight(
       #     400)  # Increase terminal window size  # Set minimum width for the output display
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
            self.mean_diameter_label, self.mean_diameter,
            self.batch_size_label, self.batch_size,
            self.mask_filter_label, self.mask_filter,
            self.img_filter_label, self.img_filter,
            self.custom_params_label, self.custom_params_input, self.run_button, self.stop_button, self.output_display
        ]

        grid_items = [
            (self.train_folder_label, 0, 0),
            (self.train_folder, 1, 0), (self.train_folder_button, 1, 1),
            (self.model_label, 2, 0),
            (self.model_file, 3, 0), (self.model_button, 3, 1),
            (self.epochs_label, 4, 0), (self.epochs_input, 4, 1),
            (self.lr_label, 5, 0), (self.lr_input, 5, 1),
            (self.wd_label, 6, 0), (self.wd_input, 6, 1),
            (self.chan_label, 7, 0), (self.chan_input, 7, 1),
            (self.chan2_label, 8, 0), (self.chan2_input, 8, 1),
            (self.mean_diameter_label, 9, 0), (self.mean_diameter, 9, 1),
            (self.batch_size_label, 10, 0), (self.batch_size, 10, 1),
            (self.mask_filter_label, 11, 0), (self.mask_filter, 11, 1),
            (self.img_filter_label, 12, 0), (self.img_filter, 12, 1),
            (self.custom_params_label, 13, 0), (self.custom_params_input, 13, 1),
            (self.gpu_checkbox, 14, 0), (self.verbose_checkbox, 14, 1),
            (self.run_button, 15, 0), (self.stop_button, 15, 1)
        ]

        for widget, row, col in grid_items:
            params_layout.addWidget(widget, row, col)

        output_layout = QVBoxLayout()
        output_layout.addWidget(self.output_display)

        left_layout.addLayout(params_layout)
        layout.addLayout(left_layout, stretch=1)
        output_layout.addWidget(self.output_display)
        layout.addLayout(output_layout, stretch=1)

        self.training_tab.setLayout(layout)
        self.setWindowTitle("Cypose GUI")

    def initSegmentingTab(self):
        """Initialize the GUI layout and widgets."""
        segmenting_layout = QHBoxLayout()
        segmenting_left_layout = QVBoxLayout()
        segmenting_params_layout = QGridLayout()

        # Custom parameter input
        self.custom_params_label = QLabel("Additional Training Parameters:")
        self.custom_params_input = QLineEdit()

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

        # GPU and Verbose Checkboxes
        self.gpu_checkbox = QCheckBox("Use GPU")
        self.verbose_checkbox = QCheckBox("Verbose")
        self.gpu_checkbox.setChecked(False)  # Ensure checkbox is visible and starts unchecked
        self.verbose_checkbox.setChecked(True)  # Ensure checkbox is visible and starts checked

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

        # Stop Button
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_process)
        self.stop_button.setEnabled(False)

        # Output Display
        self.segmenting_output_display = QTextEdit()
        # self.segmenting_output_display.setMinimumWidth(600)
        # self.segmenting_output_display.setMinimumHeight(400)
        self.segmenting_output_display.setReadOnly(True)

        # Adding widgets to layout
        widgets = [
            self.train_folder_label, self.train_folder, self.train_folder_button,
            self.epochs_label, self.epochs_input,
            self.model_label, self.model_file, self.model_button,
            self.lr_label, self.lr_input,
            self.wd_label, self.wd_input,
            self.chan_label, self.chan_input,
            self.chan2_label, self.chan2_input,
            self.mean_diameter_label, self.mean_diameter,
            self.batch_size_label, self.batch_size,
            self.mask_filter_label, self.mask_filter,
            self.img_filter_label, self.img_filter,
            self.custom_params_label, self.custom_params_input, self.run_button, self.stop_button, self.segmenting_output_display
        ]

        grid_items = [
            (self.train_folder_label, 0, 0),
            (self.train_folder, 1, 0), (self.train_folder_button, 1, 1),
            (self.model_label, 2, 0),
            (self.model_file, 3, 0), (self.model_button, 3, 1),
            (self.epochs_label, 4, 0), (self.epochs_input, 4, 1),
            (self.lr_label, 5, 0), (self.lr_input, 5, 1),
            (self.wd_label, 6, 0), (self.wd_input, 6, 1),
            (self.chan_label, 7, 0), (self.chan_input, 7, 1),
            (self.chan2_label, 8, 0), (self.chan2_input, 8, 1),
            (self.mean_diameter_label, 9, 0), (self.mean_diameter, 9, 1),
            (self.batch_size_label, 10, 0), (self.batch_size, 10, 1),
            (self.mask_filter_label, 11, 0), (self.mask_filter, 11, 1),
            (self.img_filter_label, 12, 0), (self.img_filter, 12, 1),
            (self.custom_params_label, 13, 0), (self.custom_params_input, 13, 1),
            (self.gpu_checkbox, 14, 0), (self.verbose_checkbox, 14, 1),
            (self.run_button, 15, 0), (self.stop_button, 15, 1)
        ]

        for widget, row, col in grid_items:
            segmenting_params_layout.addWidget(widget, row, col)

        segmenting_output_layout = QVBoxLayout()
        segmenting_output_layout.addWidget(self.segmenting_output_display)

        segmenting_left_layout.addLayout(segmenting_params_layout)
        segmenting_layout.addLayout(segmenting_left_layout, stretch=1)
        segmenting_output_layout.addWidget(self.segmenting_output_display)
        segmenting_layout.addLayout(segmenting_output_layout, stretch=1)

        self.segmenting_tab.setLayout(segmenting_layout)
        self.setWindowTitle("Cypose GUI")



    def select_train_folder(self):
        """Open a dialog to select the training image directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Train Folder")
        if folder:
            self.train_folder.setText(folder)

    def select_model_file(self):
        """Open a dialog to select the pretrained model file."""
        file, _ = QFileDialog.getOpenFileName(self, "Select Model File")
        if file:
            self.model_file.setText(file)

    def run_process(self):
        """Construct and execute the training command based on user inputs."""

        command = ["python", "-m", "cellpose", "--train", "--dir", self.train_folder.text(), "--n_epochs",
                   str(self.epochs_input.value())]

        if self.lr_input.value():
            command.extend(["--learning_rate", str(self.lr_input.value())])
        if self.wd_input.value():
            command.extend(["--weight_decay", str(self.wd_input.value())])
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
        if self.mean_diameter.value():
            command.extend(["--diam_mean", str(self.mean_diameter.value())])
        if self.batch_size.value():
            command.extend(["--batch_size", str(self.batch_size.value())])
        if self.custom_params_input.text():
            command.extend(self.custom_params_input.text().split())
        if self.batch_size.value():
            command.extend(["--batch_size", str(self.batch_size.value())])
        if self.gpu_checkbox.isChecked():
            command.append("--use_gpu")
        if self.verbose_checkbox.isChecked():
            command.append("--verbose")
        command_str = ' '.join(map(str, command))
        print(f"Training using command: {command_str}")

        self.script_thread = ScriptRunner(command)
        self.script_thread.output_signal.connect(self.update_output)
        self.script_thread.start()
        self.stop_button.setEnabled(True)

    def stop_process(self):
        """Stop the currently running training process."""
        if self.script_thread:
            self.script_thread.stop()
            self.stop_button.setEnabled(False)

    def update_output(self, text):
        """Update the output display with real-time logs from the subprocess."""
        self.output_display.append(text)
        self.segmenting_output_display.append(text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = CyposeGUI()
    gui.show()
    sys.exit(app.exec())
