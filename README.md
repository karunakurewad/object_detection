# Feature Detection in Images using OpenCV and Streamlit

## Introduction
This project is a **Streamlit-based web application** that performs **feature detection** in images using OpenCV's **Haar Cascade Classifiers**. The application allows users to upload an image and detect faces, eyes, and smiles using pre-trained models.

## Features
- **Face Detection**: Identifies human faces in an image.
- **Eye Detection**: Detects eyes within the image.
- **Smile Detection**: Detects smiles within the identified face regions.
- **Streamlit UI**: Provides an interactive user interface for easy image upload and feature selection.

## Installation
### Prerequisites
Make sure you have Python installed (recommended version: **3.7 or later**).

### Install Required Libraries
Run the following command to install the necessary dependencies:
```bash
pip install opencv-python-headless streamlit numpy
```

## How to Run the Application
Run the following command in your terminal or command prompt:
```bash
streamlit run app.py
```
Replace `app.py` with the filename where you have saved the script.

## Usage
1. **Upload an image** (JPG, PNG, or JPEG format).
2. **Select the features** you want to detect (Face, Eyes, or Smile).
3. **Click 'Detect Features'** to process the image.
4. The application will display the **original** and **processed** image with detected features highlighted using bounding boxes.

## Technologies Used
- **OpenCV**: Used for image processing and feature detection.
- **Streamlit**: Used for building an interactive web-based UI.
- **NumPy**: Used for handling image data arrays.

## Notes
- Haar cascades may not always be 100% accurate. Some detections might be incorrect or missed due to lighting, image quality, or face orientation.
- If detection is not working well, try uploading a **clearer, front-facing** image.

## License
This project is open-source and available for modification and distribution under the MIT License.

## Author
Developed by **Karuna Kurewad**.

