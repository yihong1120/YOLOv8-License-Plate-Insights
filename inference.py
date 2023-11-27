import os
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw
from ultralytics import YOLO
import matplotlib.pyplot as plt
from typing import List, Tuple
from google.cloud import vision

class CarLicensePlateDetector:
    """
    A class to detect and recognize license plates on cars using the YOLO model and OCR.

    Attributes:
        model (YOLO): An instance of the YOLO object detection model.
    """

    def __init__(self, weights_path: str):
        """
        Initializes the CarLicensePlateDetector with the given weights.

        Args:
            weights_path (str): The path to the weights file for the YOLO model.
        """
        self.model = YOLO(weights_path)

    def recognize_license_plate(self, img_path: str) -> np.ndarray:
        """
        Recognizes the license plate in an image and draws a rectangle around it.

        Args:
            img_path (str): The path to the image file containing the car.

        Returns:
            np.ndarray: The image with the license plate region marked and annotated with the recognized text.
        """
        img = self.load_image(img_path)
        results = self.model.predict(img, save=False)
        boxes = results[0].boxes.xyxy
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            # Extract license plate text from the ROI
            roi = img[y1:y2, x1:x2]
            license_plate = self.extract_license_plate_text(roi)

            # Draw a rectangle around the license plate
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            print(f"License: {license_plate}")
            img = self.draw_text(img, license_plate, (x1, y1 - 20))

        return img

    @staticmethod
    def draw_text(img: np.ndarray, text: str, xy: Tuple[int, int], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draws text on an image at a specified location.

        Args:
            img (np.ndarray): The image on which to draw text.
            text (str): The text to draw.
            xy (Tuple[int, int]): The (x, y) position where the text will be drawn on the image.
            color (Tuple[int, int, int], optional): The color for the text. Defaults to green.

        Returns:
            np.ndarray: The image with the text drawn on it.
        """
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        draw.text(xy, text, fill=color)
        return np.array(pil_img)

    @staticmethod
    def load_image(img_path: str) -> np.ndarray:
        """
        Loads an image from the specified path.

        Args:
            img_path (str): The path to the image file.

        Returns:
            np.ndarray: The loaded image.
        """
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img[:, :, ::-1].copy()  # Convert BGR to RGB

    @staticmethod
    def extract_license_plate_text(roi: np.ndarray) -> str:
        """
        Extracts the text from a region of interest (ROI) in an image using Google Cloud Vision API.

        Args:
            roi (np.ndarray): The region of interest in the image where the license plate is located.

        Returns:
            str: The recognized text from the license plate.
        """
        # Convert the ROI to bytes for the Vision API
        _, encoded_image = cv2.imencode('.jpg', roi)
        roi_bytes = encoded_image.tobytes()

        # Initialize the Google Cloud Vision client
        client = vision.ImageAnnotatorClient()

        # Prepare the image for the Vision API
        image = vision.Image(content=roi_bytes)

        # Perform text detection
        response = client.text_detection(image=image)

        # In case of errors
        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )

        # Extract and return the recognized text
        texts = response.text_annotations
        if texts:
            recognized_text = texts[0].description.strip()  # The first annotation contains the full detected text
            return recognized_text
        else:
            return ""


    def display_and_save(self, imgs: List[np.ndarray], save_path: str = "images/yolov8_car.jpg") -> None:
        """
        Displays and saves a list of images without altering their size.

        Args:
            imgs (List[np.ndarray]): A list of images to be displayed and saved.
            save_path (str): The file path where the image will be saved.
        """
        for i, img in enumerate(imgs):
            plt.subplot(1, len(imgs), i + 1)
            plt.axis("off")
            plt.imshow(img)
        plt.savefig(save_path, bbox_inches='tight')


    def process_video(self, video_path: str, output_path: str) -> None:
        """
        Processes a video file to detect and recognize license plates in each frame.

        Args:
            video_path (str): The path to the video file.
            output_path (str): The path where the output video will be saved.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Error opening video file")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, 
                              (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Process the frame
                annotated_frame = self.recognize_license_plate(frame)
                # Write the frame
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            else:
                break

        # Release everything when done
        cap.release()
        out.release()
        cv2.destroyAllWindows()


# [The rest of the class code remains unchanged] ...

if __name__ == '__main__':
    # Path to YOLO model weights
    weights_path: str = 'models/best.pt'
    # Instantiate the detector with the given weights
    detector = CarLicensePlateDetector(weights_path)

    # Decide whether to process an image or a video
    media_path: str = './media/car.jpg'  # Replace with the path to your image or video
    output_path: str = './media/yolov8_car.jpg'  # Replace with your output path, without file extension

    if media_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Recognize the license plate in the image
        recognized_img = detector.recognize_license_plate(media_path)
        # Save the result as a single image
        image_output_path = f"{output_path}.jpg"
        cv2.imwrite(image_output_path, cv2.cvtColor(recognized_img, cv2.COLOR_RGB2BGR))
        print(f"Annotated image saved to {image_output_path}")

    elif media_path.lower().endswith(('.mp4', '.mov', '.avi')):
        # Process the video
        video_output_path = f"{output_path}.mp4"
        detector.process_video(media_path, video_output_path)
        print(f"Processed video saved to {video_output_path}")

    else:
        print("Unsupported media format")