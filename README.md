# Object Detection and mAP Evaluation Toolkit

This toolkit provides Python scripts for performing object detection using an ONNX model and calculating the mean Average Precision (mAP) for the detection results.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [File Structure](#file-structure)
- [License](#license)

## Overview

This toolkit consists of two main parts:

1. **Object Detection using ONNX Model (`onnx_detector` module):**
   - Utilizes the `OnnxDetector` class to perform object detection using an ONNX model.
   - Generates detection results and annotation files.
   
2. **mAP Evaluation (`map_evaluator` module):**
   - Utilizes the `MAPFinder` class to calculate the mean Average Precision (mAP) for the detection results.
   - Evaluates the detection performance and outputs class-wise APs and mAP.

## Prerequisites

- Python 3.x
- OpenCV (`cv2` module)
- ONNX Runtime (`onnxruntime` package)
- `reusable_code` module (Provided separately)

## Getting Started

### Installation

1. Clone this repository to your local machine:
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
2. Install the required packages using `pip`:
    pip install -r requirement.txt

### Usage

1. Prepare your ONNX model, test dataset, ground-truth annotations, and detection results.

2. Update the necessary paths and parameters in `main.py`:
- `model_onnx`: Path to your ONNX model file.
- `test_dataset`: Path to the test dataset.
- `Classes`: List of class names.
- `gt_path`: Path to the ground-truth annotations.
- `dt_path`: Path to the detection results.
- `output_json_path`: Path to the output JSON file.

3. Run the script:
    python main.py

4. Review the output:
- The script will print the calculated mAP and class-wise APs.
- Also it will save the result in JSON file

## File Structure

The file structure is organized as follows:

    ├── main.py # Main execution script
    ├── input
    │ ├── onnx_detector.py # ONNX detection module
    │ └── map_evaluator.py # mAP evaluation module
    │ └── reusable_code.py # Reusable code module (provided separately)
    ├── input
    │ ├── detection-results # Detection result files
    │ └── ground-truth # Ground-truth annotation files
    ├── model # Directory containing ONNX model
    ├── Test dataset # Directory containing test images
    ├── requirements.txt # List of required packages
    └── README.md # Project documentation

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE](LICENSE) file for details.

