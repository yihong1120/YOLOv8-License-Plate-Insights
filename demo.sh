#!/bin/bash
# A shell script to set up an environment for a Car License Plate recognition project.

# Define the .zip filename.
filename="Car-License-Plate.zip"

# Remove the .zip extension to get the folder name.
foldername="${filename%.*}"

# Check the Operating System.
OS=$(uname)

# Function to install Tesseract and language data.
install_tesseract() {
    sudo apt update
    sudo apt install -y tesseract-ocr
    sudo apt install -y tesseract-ocr-eng
    sudo apt install -y tesseract-ocr-chi-sim
    sudo apt install -y tesseract-ocr-chi-tra
}

# Depending on the OS, execute the appropriate commands.
case "$OS" in
    "Linux")
        # Create a directory with the same name as the .zip file if it doesn't already exist.
        mkdir -p "$foldername"
        
        # Unzip the file into the corresponding directory.
        unzip -o "$filename" -d "$foldername"
        
        # Install Tesseract and required language packs.
        install_tesseract
        ;;
        
    "Darwin")
        # For macOS, check if Homebrew is installed, and install if it isn't.
        if ! command -v brew &>/dev/null; then
            echo "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install Tesseract using Homebrew.
        brew install tesseract
        ;;
        
    "MINGW"|"MSYS")
        # For Windows, use a different setup or notify the user to set up manually.
        echo "Please set up your environment manually for Windows."
        ;;
        
    *)
        echo "Unsupported OS."
        exit 1
        ;;
esac

# Execute the Python scripts for preparing the dataset and training the model.
python3 xml2txt.py
python3 dataset_preparation.py
python3 data_augmentation.py
python3 train.py
python3 inference.py