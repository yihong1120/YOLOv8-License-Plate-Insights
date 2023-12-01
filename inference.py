import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Dict, Any
from google.cloud import vision
from PIL.ExifTags import TAGS
import exifread
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
from fractions import Fraction


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
        recognized_text = None

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            # Extract license plate text from the ROI
            roi = img[y1:y2, x1:x2]
            license_plate = self.extract_license_plate_text(roi)

            # If license plate text is not empty, update recognized_text
            if license_plate:
                recognized_text = license_plate

            # Draw a rectangle around the license plate
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Print recognized text if available
        if recognized_text:
            print(f"License: {recognized_text}")
            img = self.draw_text(img, recognized_text, (x1, y1 - 20))

        # Prepare info
        image_info = self.get_image_info(img_path)
        info = {
            'DateTime': image_info.get('DateTime', None),
            'GPSLatitude': image_info.get('GPSLatitude', None),
            'GPSLongitude': image_info.get('GPSLongitude', None),
            'License': recognized_text
        }

        return info, img

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
                _, annotated_frame = self.recognize_license_plate(frame)
                # Write the frame
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            else:
                break

        # Release everything when done
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def get_media_info(self, file_path: str) -> Union[str, Dict[str, Any]]:
        """
        Get media information from a file.

        Args:
            file_path (str): The path to the media file.

        Returns:
            Union[str, Dict[str, Any]]: A dictionary containing media information or an error message.

        Raises:
            Exception: If an error occurs while reading the file.
        """
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return self.get_image_info(file_path)
        elif file_path.lower().endswith(('.mp4', '.mov', '.avi')):
            return self.get_video_info(file_path)
        else:
            return "Unsupported file format"

    def get_image_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information from an image file.

        Args:
            file_path (str): The path to the image file.

        Returns:
            Dict[str, Any]: A dictionary containing image information.

        Raises:
            Exception: If an error occurs while reading the image data.
        """
        try:
            image = Image.open(file_path)
            raw_exif_data = image._getexif()

            if raw_exif_data is None:
                return {"Error": "No EXIF data found in the image."}

            exif_data = {
                TAGS[key]: value
                for key, value in raw_exif_data.items()
                if key in TAGS and value
            }
            datetime = exif_data.get('DateTime', 'Unknown')
            gps_info = self.extract_gps_data(file_path)
            return {'DateTime': datetime, **gps_info}
        except Exception as e:
            return f"Error reading image data: {e}"

    def extract_gps_data(self, file_path: str) -> Dict[str, Any]:
        """
        Extract GPS data from an image file.

        Args:
            file_path (str): The path to the image file.

        Returns:
            Dict[str, Any]: A dictionary containing GPS information.

        Raises:
            Exception: If an error occurs while extracting GPS data.
        """
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f)
        gps_info = {}
        for tag in tags.keys():
            if tag.startswith("GPS"):
                gps_info[tag] = tags[tag]
        return self.parse_gps_info(gps_info)

    def parse_gps_info(self, gps_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Parse GPS information from a dictionary.

        Args:
            gps_info (Dict[str, Any]): A dictionary containing GPS information.

        Returns:
            Dict[str, float]: A dictionary containing parsed GPS information.

        Raises:
            Exception: If an error occurs while parsing GPS data.
        """
        gps_data = {}
        if 'GPS GPSLatitude' in gps_info and 'GPS GPSLatitudeRef' in gps_info:
            gps_data['GPSLatitude'] = self.convert_to_degrees(gps_info['GPS GPSLatitude'].values)
            if gps_info['GPS GPSLatitudeRef'].printable != 'N':
                gps_data['GPSLatitude'] = -gps_data['GPSLatitude']
        if 'GPS GPSLongitude' in gps_info and 'GPS GPSLongitudeRef' in gps_info:
            gps_data['GPSLongitude'] = self.convert_to_degrees(gps_info['GPS GPSLongitude'].values)
            if gps_info['GPS GPSLongitudeRef'].printable != 'E':
                gps_data['GPSLongitude'] = -gps_data['GPSLongitude']
        return gps_data

    def convert_to_degrees(self, value: Tuple[int, int, int]) -> float:
        """
        Convert GPS coordinate values to degrees.

        Args:
            value (Tuple[int, int, int]): A tuple containing degrees, minutes, and seconds.

        Returns:
            float: The coordinate value in degrees.
        """
        d, m, s = value
        d = float(d.numerator) / float(d.denominator)
        m = float(m.numerator) / float(m.denominator)
        s = float(s.numerator) / float(s.denominator)
        return d + (m / 60.0) + (s / 3600.0)

    def get_video_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information from a video file.

        Args:
            file_path (str): The path to the video file.

        Returns:
            Dict[str, Any]: A dictionary containing video information or an error message.

        Raises:
            Exception: If an error occurs while reading the video data.
        """
        try:
            parser = createParser(file_path)
            if not parser:
                return "Unable to parse video file"
            with parser:
                metadata = extractMetadata(parser)
            return metadata.exportDictionary() if metadata else "No metadata found in video"
        except Exception as e:
            return f"Error reading video data: {e}"


if __name__ == '__main__':
    weights_path: str = 'models/best.pt'
    detector = CarLicensePlateDetector(weights_path)

    file_path = 'medias/taiwan_taxi.jpg'
    info, _ = detector.recognize_license_plate(file_path)
    print(info)

    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Use the already loaded image
        _, recognized_img = detector.recognize_license_plate(file_path)
        image_output_path = './medias/yolov8_Scooter.jpg'
        cv2.imwrite(image_output_path, cv2.cvtColor(recognized_img, cv2.COLOR_RGB2BGR))
        print(f"Saved the image with the license plate to {image_output_path}")
    elif file_path.lower().endswith(('.mp4', '.mov', '.avi')):
        video_output_path = '/path/to/save/processed.mp4'
        detector.process_video(file_path, video_output_path)
        print(f"Saved the processed video to {video_output_path}")
    else:
        print("Unsupported media format")
