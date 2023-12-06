from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from inference import CarLicensePlateDetector
import os
import tempfile
import cv2
from typing import Tuple, Union

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size limit

# Assuming CarLicensePlateDetector class and necessary libraries are correctly imported
detector = CarLicensePlateDetector('models/best.pt')

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e: RequestEntityTooLarge) -> Tuple[str, int]:
    """Handle errors for files that exceed the maximum size limit.

    Args:
        e (RequestEntityTooLarge): Exception object for the error.

    Returns:
        Tuple[str, int]: Error message and HTTP status code.
    """
    return jsonify({"error": "File is too large. Maximum file size is 100MB."}), 413

@app.errorhandler(500)
def handle_internal_error(e: Exception) -> Tuple[str, int]:
    """Handle internal server errors.

    Args:
        e (Exception): Exception object for the error.

    Returns:
        Tuple[str, int]: Error message and HTTP status code.
    """
    return jsonify({"error": "Internal Server Error"}), 500

@app.route('/upload', methods=['POST'])
def upload_file() -> Union[str, Tuple[str, int]]:
    """Handle file upload requests.

    Returns:
        Union[str, Tuple[str, int]]: Processed file or error message with HTTP status code.
    """
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error": "No file provided or file name is empty"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(file_path)

    try:
        return process_file(filename, file_path)
    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        return jsonify({"error": str(e)}), 500

def process_file(filename: str, file_path: str) -> Union[str, Tuple[str, int]]:
    """Process the uploaded file based on its format.

    Args:
        filename (str): The name of the file.
        file_path (str): The path to the file.

    Returns:
        Union[str, Tuple[str, int]]: Processed file or error message with HTTP status code.
    """
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return process_image(file_path, filename)
    elif filename.lower().endswith(('.mp4', '.mov', '.avi')):
        return process_video(file_path, filename)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

def process_image(file_path: str, filename: str) -> str:
    """Process an image file for license plate detection.

    Args:
        file_path (str): The path to the image file.
        filename (str): The name of the file.

    Returns:
        str: Path to the processed image.
    """
    info, processed_image = detector.recognize_license_plate(file_path)
    output_path = os.path.join(tempfile.gettempdir(), 'processed_' + filename)
    cv2.imwrite(output_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
    return send_file(output_path, attachment_filename='processed_' + filename)

def process_video(file_path: str, filename: str) -> str:
    """Process a video file for license plate detection in each frame.

    Args:
        file_path (str): The path to the video file.
        filename (str): The name of the file.

    Returns:
        str: Path to the processed video.
    """
    video_output_path = os.path.join(tempfile.gettempdir(), 'processed_' + filename)
    detector.process_video(file_path, video_output_path)
    return send_file(video_output_path, attachment_filename='processed_' + filename)

if __name__ == '__main__':
    app.run()