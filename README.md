# YOLOv8-License-Plate-Detection

This repository presents an implementation of license plate recognition using the YOLOv8 object detection algorithm, demonstrating the capabilities of the YOLO architecture for real-world applications such as vehicle identification and traffic monitoring.

The project has been enhanced with the integration of Google Cloud Platform (GCP) Vision AI for performing Optical Character Recognition (OCR) to identify license plate numbers. This modification aligns the project theme and content with a focus on British English writing style.

## Setup

To use this project, you'll need to install several dependencies and set up your environment. Here are the steps to get started:

### Prerequisites

- Python 3.8 or higher
- PyTorch (with MPS or CUDA support for GPU acceleration)
- Ultralytics YOLOv8 library
- OpenCV

### Installation

1. Clone the repository to your local machine:

    ```sh
    git clone https://github.com/yihong1120/YOLOv8-License-Plate-Detection.git
    ```

2. Download the [car license plate dataset](https://1drv.ms/u/s!AiltJg0lR4P-ylzt6zyr3s3tEpij?e=r5E9ja), the  original dataset link is [here](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download).

3. Navigate to the cloned directory:

    ```sh
    cd YOLOv8-License-Plate-Detection
    ```

4. Install the required Python libraries:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

To use this model for license plate detection and recognition, you need to have images of cars on which you want to perform the detection.

1. Modify the `weights_path` in `CarLicensePlateDetector.py` to point to your YOLOv8 weights file:

    ```python
    weights_path = './path_to_your_weights/best.pt'
    ```

2. Run the license plate detection script:

    ```sh
    python inference.py
    ```

3. Results will be displayed and can be saved as images with the recognized license plates highlighted, the following is the message returned after execution.
    ```
    0: 480x640 1 license, 177.7ms
    Speed: 4.2ms preprocess, 177.7ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)
    License: 189-16
    ```

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

## Experiment

we set up an test with the parameters:
- num_augmentations = 5
- epochs = 100
- model_name = yolov8n.pt

![train_output](./images/train_output.png)

1. `Confusion Matrix Normalized`: This chart shows the normalized results of the model's predictions. The strong diagonal entries (0.95 for 'license' and 1.00 for 'background') indicate that the model is performing very well, with high true positive rates and low false negatives and false positives.

2. `Precision-Recall Curve`: This curve is used to evaluate the performance of the object detection model at different thresholds. The high area under the curve (AUC) of 0.928 is excellent, suggesting that the model is able to differentiate between the 'license' class and all other classes with high precision and recall.

3. `F1-Confidence Curve`: This curve represents the F1 score (a harmonic mean of precision and recall) across various confidence thresholds. The peak F1 score of 0.90 at a confidence threshold of 0.408 indicates that at this threshold, the model achieves a good balance between precision and recall.

## Contributions

Contributions to this project are welcome. Please submit a pull request or create an issue if you have ideas for improvements or have found a bug.

## License

This project is open-sourced under the MIT License. See the [LICENSE](./LICENSE) file for more information.

## Acknowledgements

- The YOLOv8 model and implementation are from Ultralytics. Visit their [GitHub repository](https://github.com/ultralytics/ultralytics) for more information.