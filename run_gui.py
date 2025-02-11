import os
import sys
import subprocess
import datetime
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
3. Plot loss curve
4. Check if the loss not decreasing and terminate the process. 
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

        # Create training log folders if they don't exist.
        self.t_log_dir = "training_logs"
        if not os.path.exists(self.t_log_dir):
            os.makedirs(self.t_log_dir)
        self.s_log_dir = "segmentation_logs"
        if not os.path.exists(self.s_log_dir):
            os.makedirs(self.s_log_dir)


    def initTabs(self):
        self.setWindowTitle("Cypose GUI")
        self.setGeometry(100, 100, 1000, 600)

        # Create tabs
        self.tabs = QTabWidget()
        self.training_tab = QWidget()
        self.segmenting_tab = QWidget()

        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.segmenting_tab, "Segmentation")

        # Initialize layouts for both tabs
        self.initTrainingTab()
        self.initSegmentingTab()

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def initTrainingTab(self):
        """Create GUI Layouts"""
        train_H_layout = QHBoxLayout() # Horizontal Layout
        train_v_layout = QVBoxLayout()    # Vertical Layout
        params_layout = QGridLayout()   # Grid Layout

        # Train directory
        self.train_folder_label = QLabel("Select Train Image Directory:")
        self.train_folder_button = QPushButton("Browse")
        self.train_folder_button.clicked.connect(self.select_train_folder)
        self.train_folder = QLineEdit()

        # Number of Epochs Input
        self.t_epochs_label = QLabel("Number of Epochs:")
        self.t_epochs_input = QSpinBox()
        self.t_epochs_input.setRange(1, 5000)
        self.t_epochs_input.setValue(100)

        # Pretrained Model File Selection
        self.t_model_label = QLabel("Select Pretrained Model File:")
        self.t_model_button = QPushButton("Browse")
        self.t_model_button.clicked.connect(self.select_t_model_file)
        self.t_model_file = QLineEdit()
        self.t_model_file.setText("None")

        # Learning Rate Input
        self.t_lr_label = QLabel("Learning Rate:")
        self.t_lr_input = QDoubleSpinBox()
        self.t_lr_input.setDecimals(6)
        self.t_lr_input.setRange(0.000001, 1.0)
        self.t_lr_input.setSingleStep(0.000001)
        self.t_lr_input.setValue(0.001)

        # Weight Decay Input
        self.t_wd_label = QLabel("Weight Decay:")
        self.t_wd_input = QDoubleSpinBox()
        self.t_wd_input.setDecimals(6)
        self.t_wd_input.setRange(0.0001, 1.0)
        self.t_wd_input.setSingleStep(0.0001)
        self.t_wd_input.setValue(0.0001)

        # GPU and Verbose Checkboxes
        self.t_gpu_checkbox = QCheckBox("Use GPU")
        self.t_verbose_checkbox = QCheckBox("Verbose")
        self.t_gpu_checkbox.setChecked(False)  # Ensure checkbox is visible and starts unchecked
        self.t_verbose_checkbox.setChecked(True)  # Ensure checkbox is visible and starts checked

        # Channel Inputs
        self.t_chan_label = QLabel("Channel 0:")
        self.t_chan_input = QSpinBox()
        self.t_chan_input.setRange(0, 5)
        self.t_chan_input.setValue(0)

        self.t_chan2_label = QLabel("Channel 1:")
        self.t_chan2_input = QSpinBox()
        self.t_chan2_input.setRange(0, 5)
        self.t_chan2_input.setValue(0)

        # Mask & Image Filters
        self.t_mask_filter_label = QLabel("Mask Filter:")
        self.t_mask_filter = QLineEdit()
        self.t_mask_filter.setText("_labeled")
        self.t_img_filter_label = QLabel("Image Filter:")
        self.t_img_filter = QLineEdit()

        # Mean Diameter
        self.t_mean_diameter_label = QLabel("Mean Diameter:")
        self.t_mean_diameter = QSpinBox()
        self.t_mean_diameter.setRange(1, 100)
        self.t_mean_diameter.setValue(30)

        # Batch Size
        self.t_batch_size_label = QLabel("Batch Size:")
        self.t_batch_size = QSpinBox()
        self.t_batch_size.setRange(1, 100)
        self.t_batch_size.setValue(8)

        # Custom parameter input
        self.t_custom_params_label = QLabel("Additional Training Parameters:")
        self.t_custom_params_input = QLineEdit()

        # Run Button
        self.t_run_button = QPushButton("Run")
        self.t_run_button.clicked.connect(self.t_run_process)
        self.t_run_button.setEnabled(False)

        # Stop Button
        self.t_stop_button = QPushButton("Stop")
        self.t_stop_button.clicked.connect(self.t_stop_process)
        self.t_stop_button.setEnabled(False)

        # Output Display
        self.t_output_display = QTextEdit()
        self.t_output_display.setReadOnly(True)

        # Adding widgets to layout
        widgets = [
            self.train_folder_label, self.train_folder, self.train_folder_button,
            self.t_epochs_label, self.t_epochs_input,
            self.t_model_label, self.t_model_file, self.t_model_button,
            self.t_lr_label, self.t_lr_input,
            self.t_wd_label, self.t_wd_input,
            self.t_chan_label, self.t_chan_input,
            self.t_chan2_label, self.t_chan2_input,
            self.t_mean_diameter_label, self.t_mean_diameter,
            self.t_batch_size_label, self.t_batch_size,
            self.t_mask_filter_label, self.t_mask_filter,
            self.t_img_filter_label, self.t_img_filter,
            self.t_custom_params_label, self.t_custom_params_input, self.t_run_button, self.t_stop_button
        ]

        grid_items = [
            (self.train_folder_label, 0, 0),
            (self.train_folder, 1, 0), (self.train_folder_button, 1, 1),
            (self.t_model_label, 2, 0),
            (self.t_model_file, 3, 0), (self.t_model_button, 3, 1),
            (self.t_epochs_label, 4, 0), (self.t_epochs_input, 4, 1),
            (self.t_lr_label, 5, 0), (self.t_lr_input, 5, 1),
            (self.t_wd_label, 6, 0), (self.t_wd_input, 6, 1),
            (self.t_chan_label, 7, 0), (self.t_chan_input, 7, 1),
            (self.t_chan2_label, 8, 0), (self.t_chan2_input, 8, 1),
            (self.t_mean_diameter_label, 9, 0), (self.t_mean_diameter, 9, 1),
            (self.t_batch_size_label, 10, 0), (self.t_batch_size, 10, 1),
            (self.t_mask_filter_label, 11, 0), (self.t_mask_filter, 11, 1),
            (self.t_img_filter_label, 12, 0), (self.t_img_filter, 12, 1),
            (self.t_custom_params_label, 13, 0), (self.t_custom_params_input, 13, 1),
            (self.t_gpu_checkbox, 14, 0), (self.t_verbose_checkbox, 14, 1),
            (self.t_run_button, 15, 0), (self.t_stop_button, 15, 1)
        ]

        for widget, row, col in grid_items:
            params_layout.addWidget(widget, row, col)
        
        # All params layout
        train_v_layout.addLayout(params_layout)
        train_H_layout.addLayout(train_v_layout, stretch=1)
        
        # Output display layout
        output_layout = QVBoxLayout()
        output_layout.addWidget(self.t_output_display)
        train_H_layout.addLayout(output_layout, stretch=1)

        self.training_tab.setLayout(train_H_layout)
        self.setWindowTitle("Cypose GUI")

    def initSegmentingTab(self):
        """Create GUI Layots for segmenting tab."""
        seg_h_layout = QHBoxLayout()
        segmenting_layout = QVBoxLayout()
        segmenting_params_layout = QGridLayout()

        # Flags to activate Run button
        self.output_flag = False
        self.image_flag = False
        self.model_flag = False

        # Image file
        self.s_image_file_label = QLabel("Select Image To Segment:")
        self.s_image_file_button = QPushButton("Browse")
        self.s_image_file_button.clicked.connect(self.select_s_image_file)
        self.s_image_file = QLineEdit()

        # Output location
        self.s_output_dir_label = QLabel("Select Output Directory:")
        self.s_output_dir_button = QPushButton("Browse")
        self.s_output_dir_button.clicked.connect(self.select_output_folder)
        self.s_output_dir = QLineEdit()

        # Model File Selection
        self.s_model_label = QLabel("Select Model File:")
        self.s_model_button = QPushButton("Browse")
        self.s_model_button.clicked.connect(self.select_s_model_file)
        self.s_model_file = QLineEdit()

        # Flow_threshold
        self.s_flow_th_label = QLabel("Flow Threshold:")
        self.s_flow_th_input = QLineEdit()
        self.s_flow_th_input.setText(None)

        # Num iterations
        self.niter_label = QLabel("Iteration:")
        self.niter_input = QLineEdit()
        self.niter_input.setText(None)

        # Size
        self.size_label = QLabel("Cell Size:")
        self.size_input = QLineEdit()
        self.size_input.setText(None)

        # Start Frame
        self.start_frame_label = QLabel("Start Frame:")
        self.start_frame_input = QSpinBox()
        self.start_frame_input.setRange(0, 10000)
        self.start_frame_input.setValue(0)

        # End Frame
        self.end_frame_label = QLabel("End Frame:")
        self.end_frame_input = QSpinBox()
        self.end_frame_input.setRange(-1, 10000)
        self.end_frame_input.setValue(-1)

        # GPU, Debug and Denoise Checkboxes
        self.s_gpu_checkbox = QCheckBox("Use GPU")
        self.s_debug_checkbox = QCheckBox("Debug")
        self.s_denoise_checkbox = QCheckBox("Denoise")

        self.s_gpu_checkbox.setChecked(False)  # Ensure checkbox is visible and starts unchecked
        self.s_debug_checkbox.setChecked(False)  # Ensure checkbox is visible and starts checked
        self.s_denoise_checkbox.setChecked(False)  # Ensure checkbox is visible and starts checked

        # Custom parameter input
        self.s_custom_params_label = QLabel("Additional Training Parameters:")
        self.s_custom_params_input = QLineEdit()

        # Run Button
        self.s_run_button = QPushButton("Run")
        self.s_run_button.clicked.connect(self.s_run_process)
        self.s_run_button.setEnabled(True)

        # Stop Button
        self.s_stop_button = QPushButton("Stop")
        self.s_stop_button.clicked.connect(self.s_stop_process)
        self.s_stop_button.setEnabled(False) # Disabled by default

        # Output Display
        self.s_output_display = QTextEdit()
        self.s_output_display.setReadOnly(True)

        # Adding widgets to layout
        widgets = [
            self.s_image_file_label, self.s_image_file, self.s_image_file_button,
            self.s_model_label, self.s_model_file, self.s_model_button,
            self.s_custom_params_label, self.s_custom_params_input, self.s_run_button, self.s_stop_button, self.s_output_display
        ]

        grid_items = [
            (self.s_image_file_label, 0, 0),
            (self.s_image_file, 1, 0), (self.s_image_file_button, 1, 1),
            (self.s_model_label, 2, 0),
            (self.s_model_file, 3, 0), (self.s_model_button, 3, 1),
            (self.s_output_dir_label, 4, 0),
            (self.s_output_dir, 5, 0), (self.s_output_dir_button, 5, 1),

            (self.s_flow_th_label, 6, 0), (self.s_flow_th_input, 6, 1),
            (self.niter_label, 7, 0), (self.niter_input, 7, 1),
            (self.size_label, 8, 0), (self.size_input, 8, 1),
            (self.start_frame_label, 9, 0), (self.start_frame_input, 9, 1),
            (self.end_frame_label, 10, 0), (self.end_frame_input, 10, 1),

            (self.s_custom_params_label, 11, 0), (self.s_custom_params_input, 11, 1),
            (self.s_gpu_checkbox, 12, 0), (self.s_denoise_checkbox, 12, 1),
            (self.s_debug_checkbox, 13, 0),
            (self.s_run_button, 14, 0), (self.s_stop_button, 14, 1)
        ]
        for widget, row, col in grid_items:
            segmenting_params_layout.addWidget(widget, row, col)

        # All params layout

        segmenting_layout.addLayout(segmenting_params_layout)
        seg_h_layout.addLayout(segmenting_layout, stretch=1)

        # Output display layout
        segmenting_output_layout = QVBoxLayout()
        segmenting_output_layout.addWidget(self.s_output_display)
        seg_h_layout.addLayout(segmenting_output_layout, stretch=1)

        self.segmenting_tab.setLayout(seg_h_layout)
        self.setWindowTitle("Cypose GUI")


    def select_train_folder(self):
        """Open a dialog to select the training image directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Train Folder")
        if folder:
            self.train_folder.setText(folder)
            self.t_run_button.setEnabled(True)

    def select_output_folder(self):
        """Open a dialog to select the output directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.s_output_dir.setText(folder)
            self.output_flag = True

    def select_s_image_file(self):
        """Open a dialog to select the pretrained model file."""
        file, _ = QFileDialog.getOpenFileName(self, "Select Image File")
        if file:
            self.s_image_file.setText(file)
            self.image_flag = True

    def select_t_model_file(self):
        """Open a dialog to select the pretrained model file."""
        file, _ = QFileDialog.getOpenFileName(self, "Select Model File")
        if file:
            self.t_model_file.setText(file)

    def select_s_model_file(self):
        """Open a dialog to select the segmenting model file."""
        file, _ = QFileDialog.getOpenFileName(self, "Select Model File")
        if file:
            self.s_model_file.setText(file)
            self.model_flag = True

    def t_run_process(self):
        """Construct and execute the training command based on user input"""

        command = ["python", "-m", "cellpose", "--train", "--dir", self.train_folder.text(), "--n_epochs",
                   str(self.t_epochs_input.value())]

        if self.t_lr_input.value():
            command.extend(["--learning_rate", str(self.t_lr_input.value())])
        if self.t_wd_input.value():
            command.extend(["--weight_decay", str(self.t_wd_input.value())])
        if self.t_model_file.text():
            command.extend(["--pretrained_model", self.t_model_file.text()])
        if self.t_chan_input or self.t_chan_input.value() == 0:
            command.extend(["--chan", str(self.t_chan_input.value())])
        if self.t_chan2_input.value() or self.t_chan2_input.value() == 0:
            command.extend(["--chan2", str(self.t_chan2_input.value())])
        if self.t_mask_filter.text():
            command.extend(["--mask_filter", self.t_mask_filter.text()])
        if self.t_img_filter.text():
            command.extend(["--img_filter", self.t_img_filter.text()])
        if self.t_mean_diameter.value():
            command.extend(["--diam_mean", str(self.t_mean_diameter.value())])
        if self.t_batch_size.value():
            command.extend(["--batch_size", str(self.t_batch_size.value())])
        if self.t_custom_params_input.text():
            command.extend(self.t_custom_params_input.text().split())
        if self.t_gpu_checkbox.isChecked():
            command.append("--use_gpu")
        if self.t_verbose_checkbox.isChecked():
            command.append("--verbose")
        command_str = ' '.join(map(str, command))

        # Create log file and timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.t_log_file = f"./{self.t_log_dir}/training_log_{timestamp}.txt"

        # Print the command
        self.t_update_output(f"Training using command: {command_str}")

        self.script_thread = ScriptRunner(command)
        self.script_thread.output_signal.connect(self.t_update_output)
        self.script_thread.start()
        self.t_stop_button.setEnabled(True)

    def s_run_process(self):
        """Construct and execute the training command based on user inputs."""

        # Create log file and timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.s_log_file = f"./{self.s_log_dir}/training_log_{timestamp}.txt"

        command = ["python", "run_segmentation.py", "--model", self.s_model_file.text(),
                   "--input_file", self.s_image_file.text(), "--output_file",
                   f"{self.s_output_dir.text()}/{os.path.splitext(os.path.basename(self.s_image_file.text()))[0]}"
                   f"_mask_{timestamp}.tif"]

        if self.s_flow_th_input.text():
            command.extend(["--flow_threshold", self.s_flow_th_input.text()])
        if self.niter_input.text():
            command.extend(["--niter", self.niter_input.text()])
        if self.size_input.text():
            command.extend(["--size", self.size_input.text()])
        if self.start_frame_input.text():
            command.extend(["--start_frame", self.start_frame_input.text()])
        if self.end_frame_input.text():
            command.extend(["--end_frame", self.end_frame_input.text()])
        if self.s_custom_params_input.text():
            command.extend(self.s_custom_params_input.text().split())
        if self.s_gpu_checkbox.isChecked():
            command.append("--gpu")
        if self.s_debug_checkbox.isChecked():
            command.append("--debug")
        if self.s_denoise_checkbox.isChecked():
            command.append("--denoise")


        if all([self.output_flag, self.image_flag, self.model_flag]):
            # Print the command
            command_str = ' '.join(map(str, command))
            self.s_update_output(f""
                                 f"Segmenting using command: {command_str}")
            # Run the script
            self.script_thread = ScriptRunner(command)
            self.script_thread.output_signal.connect(self.s_update_output)
            self.script_thread.start()
            self.s_stop_button.setEnabled(True)
        else:
            self.s_update_output("Make sure to select:\nImage to segment\nModel\nOutput Directory\n")

    def t_stop_process(self):
        """Stop the currently running training process."""
        if self.script_thread:
            self.script_thread.stop()
            self.t_stop_button.setEnabled(False)

    def s_stop_process(self):
        """Stop the currently running training process."""
        if self.script_thread:
            self.script_thread.stop()
            self.s_stop_button.setEnabled(False)

    def t_update_output(self, text):
        """Update the output display with real-time logs from the subprocess."""


        # Append to QTextEdit
        #self.t_output_display.ensureCursorVisible()  # Auto-scroll to the latest log
        self.t_output_display.append(text)

        with open(self.t_log_file, "a", encoding='utf-8') as f:
            f.write(text + "\n")
            
    def s_update_output(self, text):
        """Update the output display with real-time logs from the subprocess."""


        # Append to QTextEdit
        #self.t_output_display.ensureCursorVisible()  # Auto-scroll to the latest log
        self.s_output_display.append(text)

        with open(self.s_log_file, "a", encoding='utf-8') as f:
            f.write(text + "\n")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = CyposeGUI()
    gui.show()
    sys.exit(app.exec())
