import os
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw
from ultralytics import YOLO
import matplotlib.pyplot as plt

class CarLicensePlateDetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def recognize_license_plate(self, img_path):
        img = self.load_image(img_path)
        results = self.model.predict(img, save=False)
        boxes = results[0].boxes.xyxy
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            license_plate = self.extract_license_plate_text(img[y1:y2, x1:x2])
            print(f"License: {license_plate}")
            img = self.draw_text(img, license_plate, (x1, y1 - 20))
        return img

    @staticmethod
    def draw_text(img, text, xy, color=(0, 255, 0)):
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        draw.text(xy, text, fill=color)
        return np.array(pil_img)

    @staticmethod
    def load_image(img_path):
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img[:, :, ::-1].copy()  # Convert BGR to RGB

    @staticmethod
    def extract_license_plate_text(roi):
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        license_plate = pytesseract.image_to_string(gray_roi, lang='eng', config='--psm 11')
        return license_plate.strip()

    def display_and_save(self, imgs, save_path="yolov8_car.jpg"):
        plt.figure(figsize=(12, 9))
        for i, img in enumerate(imgs):
            plt.subplot(2, 3, i + 1)
            plt.axis("off")
            plt.imshow(img)
        plt.savefig(save_path)


if __name__ == '__main__':
    weights_path = './runs/detect/train/weights/best.pt'
    detector = CarLicensePlateDetector(weights_path)

    img_path = './car.png'  # Can be replaced with any image path
    recognized_img = detector.recognize_license_plate(img_path)

    # Display and save the result as a single image
    detector.display_and_save([recognized_img])

    # If you want to run inference on multiple images:
    # img_dir = "./valid/images"
    # processed_imgs = [detector.recognize_license_plate(os.path.join(img_dir, file))
    #                   for file in os.listdir(img_dir)[:6]]
    # detector.display_and_save(processed_imgs)
