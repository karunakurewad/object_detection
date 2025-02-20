import cv2
import streamlit as st
import numpy as np

# Load pre-trained Haar Cascade classifiers for face, eyes, smile, and upper body
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Function to detect features in an image
def detect_features(image, detect_eyes, detect_face, detect_smile):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = image.copy()

    if detect_face:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(features, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Detect smiles only within the face region
            if detect_smile:
                roi_gray = gray[y:y + h, x:x + w]  # Region of interest (face region)
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(features, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 0, 255), 2)

    if detect_eyes:
        eyes = eye_cascade.detectMultiScale(gray)
        for (x, y, w, h) in eyes:
            cv2.rectangle(features, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return features

# Streamlit UI
st.title("Feature Detection in Image")
st.sidebar.title("Select Features to Detect")
st.sidebar.write("Choose what you want to detect from the options below:")

# Allow user to choose what to detect via the sidebar
detect_face = st.sidebar.checkbox("Detect Face", True)
detect_eyes = st.sidebar.checkbox("Detect Eyes", False)
detect_smile = st.sidebar.checkbox("Detect Smile", False)

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display the image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, 1)

    # Display original image
    st.subheader("Original Image")
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)


    if st.button("Detect Features"):
        # Perform feature detection
        result = detect_features(image, detect_eyes, detect_face, detect_smile)
        
        # Display the processed image after detection
        st.subheader("Processed Image (After Prediction)")
        st.image(result, channels="BGR", caption="Detected Features", use_column_width=True)
        # Display a message about prediction accuracy
        st.write("Note: The detection is not 100% accurate. It is approximately 60% correct. \n If it's not detecting the correct then  you can change your image also, sometime is depends on image also")
        
