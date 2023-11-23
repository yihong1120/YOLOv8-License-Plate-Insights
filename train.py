import argparse
from typing import Any, Optional
from ultralytics import YOLO

class YOLOModelHandler:
    """Handles loading, training, validating, and predicting with YOLO models.

    Attributes:
        model_name (str): The name of the model file to be loaded.
        model (YOLO, Optional): The loaded YOLO model object.
    """

    def __init__(self, model_name: str):
        """
        Initialises the YOLOModelHandler with a specified model.

        Args:
            model_name (str): The name of the model file (either .yaml or .pt).
        """
        self.model_name: str = model_name
        self.model: Optional[YOLO] = None
        self.load_model()

    def load_model(self) -> None:
        """Loads the YOLO model specified by the model name."""
        if self.model_name.endswith('.yaml'):
            # Build a new model from scratch
            self.model = YOLO(self.model_name)
        elif self.model_name.endswith('.pt'):
            # Load a pre-trained model (recommended for training)
            self.model = YOLO(self.model_name)
        else:
            raise ValueError("Unsupported model format. Use '.yaml' or '.pt'")

    def train_model(self, data_config: str, epochs: int) -> None:
        """
        Trains the YOLO model using the specified data configuration and for a number of epochs.

        Args:
            data_config (str): The path to the data configuration file.
            epochs (int): The number of training epochs.

        Raises:
            RuntimeError: If the model is not loaded properly before training.
        """
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Train the model
        self.model.train(data=data_config, epochs=epochs)

    def validate_model(self) -> Any:
        """
        Validates the YOLO model on the validation dataset.

        Returns:
            The validation results.

        Raises:
            RuntimeError: If the model is not loaded properly before validation.
        """
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Evaluate model performance on the validation set
        return self.model.val()

    def predict_image(self, image_path: str) -> Any:
        """
        Makes a prediction using the YOLO model on the specified image.

        Args:
            image_path (str): The path to the image file for prediction.

        Returns:
            The prediction results.

        Raises:
            RuntimeError: If the model is not loaded properly before prediction.
        """
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Predict on an image
        return self.model(image_path)

    def export_model(self, export_format: str = "onnx") -> str:
        """
        Exports the YOLO model to the specified format.

        Args:
            export_format (str): The format to export the model to.

        Returns:
            The path to the exported model file.

        Raises:
            RuntimeError: If the model is not loaded properly before exporting.
        """
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Export the model to the desired format
        return self.model.export(format=export_format)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Handle YOLO model training, validation, prediction, and exporting.')
    
    parser.add_argument('--data_config', type=str, default='data.yaml', help='Path to the data configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--model_name', type=str, default='yolov8n.pt', help='Name of the YOLO model file')
    parser.add_argument('--export_format', type=str, default='onnx', help='Format to export the model to')
    parser.add_argument('--onnx_path', type=str, default=None, help='Path to save the exported ONNX model')

    args = parser.parse_args()

    # Initialise the handler with a model name
    handler = YOLOModelHandler(args.model_name)

    # Train the model if needed
    handler.train_model(data_config=args.data_config, epochs=args.epochs)

    # Validate the model
    metrics = handler.validate_model()

    # Predict on an image
    image_url = "https://ultralytics.com/images/bus.jpg"  # 这里仍然使用固定的图片 URL
    results = handler.predict_image(image_url)

    # Export the model to the specified format
    export_path = handler.export_model(export_format=args.export_format) if args.onnx_path is None else args.onnx_path

    # Output the results
    print("Prediction results:", results)
    print(f"{args.export_format.upper()} model exported to:", export_path)