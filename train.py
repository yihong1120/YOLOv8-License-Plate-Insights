import os
from ultralytics import YOLO

class YOLOModelHandler:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.load_model()

    def load_model(self):
        if self.model_name.endswith('.yaml'):
            # Build a new model from scratch
            self.model = YOLO(self.model_name)
        elif self.model_name.endswith('.pt'):
            # Load a pretrained model (recommended for training)
            self.model = YOLO(self.model_name)
        else:
            raise ValueError("Unsupported model format. Use '.yaml' or '.pt'")

    def train_model(self, data_config, epochs):
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Train the model
        self.model.train(data=data_config, epochs=epochs)

    def validate_model(self):
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Evaluate model performance on the validation set
        return self.model.val()

    def predict_image(self, image_path):
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Predict on an image
        return self.model(image_path)

    def export_model(self, export_format="onnx"):
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Export the model to the desired format
        return self.model.export(format=export_format)


if __name__ == '__main__':
    handler = YOLOModelHandler("yolov8n.pt")  # or "yolov8n.yaml"

    # Train the model if needed
    handler.train_model(data_config="data.yaml", epochs=100)

    # Validate the model
    metrics = handler.validate_model()

    # Predict on an image
    image_url = "https://ultralytics.com/images/bus.jpg"
    results = handler.predict_image(image_url)

    # Export the model to ONNX format
    onnx_path = handler.export_model(export_format="onnx")

    print("Prediction results:", results)
    print("ONNX model exported to:", onnx_path)
