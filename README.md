🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLOv8-License-Plate-Insights

This repository demonstrates license plate recognition using the YOLOv8 object detection algorithm, showcasing the versatility of the YOLO architecture in real-world scenarios such as vehicle identification, traffic monitoring, and geospatial analysis. Additionally, the project integrates Google Cloud Platform (GCP) Vision AI for Optical Character Recognition (OCR), enabling accurate license plate identification while capturing crucial media capture data, including date, time, and geolocation. This metadata extraction enhances data understanding and supports applications like traffic analysis and location-based analytics.

## Features

* `License Plate Recognition`: Utilising YOLOv8, the project excels at identifying and extracting license plate numbers from images and videos.

* `Media Capture Data`: Beyond license plate information, the project now retrieves essential media capture data, including the date, time, and geographical coordinates (latitude and longitude). This data enriches the analysis and extends the project's usability to various real-world applications.

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
    git clone https://github.com/yihong1120/YOLOv8-License-Plate-Insights.git
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

1. Modify the `weights_path` in `train.py` to point to your YOLOv8 weights file.  To train the model on Google Colab, you can utilise this [file](License_Plate_Detection.ipynb):

    ```python
    python train.py
    ```

2. Run the license plate detection script.  To enable Vision AI api, you can refer to [Vision-API-Tutorial](manual/Vision_API_Tutorial.md):

    ```sh
    python inference.py
    ```

3. Results will be displayed and can be saved as images with the recognised license plates highlighted, you can see the image output below.  The following is the message returned after execution.

![Output](medias/Scooter.png)

    ```
    {'DateTime': '2023:11:17 19:01:29', 'GPSLatitude': 24.15218611111111, 'GPSLongitude': 120.67495555555556}

    0: 384x640 1 license, 196.7ms
    Speed: 6.6ms preprocess, 196.7ms inference, 16.4ms postprocess per image at shape (1, 3, 384, 640)
    License: NNG 9569
    Saved the image with the license plate to ./medias/yolov8_Scooter.jpg
    ```

## Custom Training

For Linux of Mac user, you can simply execute the command below complete the holistic process.

    bash demo.sh

If you wish to train your own YOLOv8 model on custom license plate data, you'll need to follow these additional steps:

1. Prepare your dataset according to the YOLOv8 requirements (annotate your images in the proper format).
2. Update `data.yaml` with the paths to your training and validation datasets, and the number of classes.
3. Run the training script:

    ```sh
    python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov8n.pt
    ```

4. After training, you can find the best performing weights in the `runs/train/exp*/weights` directory.

## Experiment

We conducted an experiment with the following parameters:
- num_augmentations = 5
- epochs = 100
- model_name = yolov8n.pt

![train_output](./medias/train_output.png)

1. `Confusion Matrix Normalized`: This chart shows the normalised results of the model's predictions. The strong diagonal entries (0.95 for 'license' and 1.00 for 'background') indicate that the model is performing very well, with high true positive rates and low false negatives and false positives.

2. `Precision-Recall Curve`: This curve is used to evaluate the performance of the object detection model at different thresholds. The high area under the curve (AUC) of 0.928 is excellent, suggesting that the model is able to differentiate between the 'license' class and all other classes with high precision and recall.

3. `F1-Confidence Curve`: This curve represents the F1 score (a harmonic mean of precision and recall) across various confidence thresholds. The peak F1 score of 0.90 at a confidence threshold of 0.408 indicates that at this threshold, the model achieves a good balance between precision and recall.

## Deployment

The model shall be deployed in Google App Engine(GAE), serving as API.  We decided to utilise Flask to generate the API functionalities.  You can refer to [app.py](./app.py) and execute the API locally with the command below:

    gunicorn app:app

The model can be deployed to GAE, you can refer to this [manual](./manual/Flask_GAE_Deployment_Tutorial.md).

## License

This project is open-sourced under the MIT License. See the [LICENSE](./LICENSE) file for more information.

## Acknowledgements

- The YOLOv8 model and implementation are from Ultralytics. Visit their [GitHub repository](https://github.com/ultralytics/ultralytics) for more information.