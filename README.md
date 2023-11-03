# YOLOv8-License-Plate-Detection

This repository provides an implementation of license plate recognition using the YOLOv8 object detection algorithm, showcasing the capabilities of the YOLO architecture for real-world applications such as vehicle identification and traffic monitoring.

## Setup

To use this project, you'll need to install several dependencies and set up your environment. Here are the steps to get started:

### Prerequisites

- Python 3.8 or higher
- PyTorch (with CUDA support for GPU acceleration)
- Ultralytics YOLOv8 library
- OpenCV
- PyTesseract
- Tesseract-OCR Engine

### Installation

1. Clone the repository to your local machine:

    ```sh
    git clone https://github.com/yihong1120/YOLOv8-License-Plate-Detection.git
    ```

2. Navigate to the cloned directory:

    ```sh
    cd YOLOv8-License-Plate-Detection
    ```

3. Install the required Python libraries:

    ```sh
    pip install -r requirements.txt
    ```

4. Install Tesseract-OCR. Please refer to the Tesseract documentation for installation instructions on your operating system.

5. Install PyTesseract using pip:

    ```sh
    pip install pytesseract
    ```

6. Make sure to add the path to the Tesseract executable to your system's PATH, or set it in your Python scripts:

    ```python
    pytesseract.pytesseract.tesseract_cmd = r'path_to_your_tesseract_executable'
    ```

## Usage

To use this model for license plate detection and recognition, you need to have images of cars on which you want to perform the detection.

1. Modify the `weights_path` in `CarLicensePlateDetector.py` to point to your YOLOv8 weights file:

    ```python
    weights_path = './path_to_your_weights/best.pt'
    ```

2. Run the license plate detection script:

    ```sh
    python CarLicensePlateDetector.py
    ```

3. Results will be displayed and can be saved as images with the recognized license plates highlighted.

## Custom Training

For Linux of Mac user, you can simply execution the command below complete the holistic process.
    ```sh
    bash demo.sh
    ```

If you wish to train your own YOLOv8 model on custom license plate data, you'll need to follow these additional steps:

1. Prepare your dataset according to the YOLOv8 requirements (annotate your images in the proper format).
2. Update `data.yaml` with the paths to your training and validation datasets, and the number of classes.
3. Run the training script:

    ```sh
    python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov8n.pt
    ```

4. After training, you can find the best performing weights in the `runs/train/exp*/weights` directory.

## Contributions

Contributions to this project are welcome. Please submit a pull request or create an issue if you have ideas for improvements or have found a bug.

## License

This project is open-sourced under the MIT License. See the [LICENSE](./LICENSE) file for more information.

## Acknowledgements

- The YOLOv8 model and implementation are from Ultralytics. Visit their [GitHub repository](https://github.com/ultralytics/yolov8) for more information.
- Tesseract-OCR is developed by Google. Visit the [Tesseract GitHub repository](https://github.com/tesseract-ocr/tesseract) for more details.