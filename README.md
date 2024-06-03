# Plant Disease Detection Model
![pd](https://github.com/FajanSunusara/Plant_Disease_Detection_Model/assets/49346372/81fbb9b2-6cd7-478a-adb6-42c9dbda16ad)


Welcome to the Plant Disease Detection Model repository! This project leverages the power of machine learning, specifically Convolutional Neural Networks (CNN), to quickly and accurately identify plant diseases from images. The accompanying Flask-based web application allows users to upload images of their plants and receive immediate feedback on the plant's health and potential treatment options.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Web Application](#web-application)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Plant health is crucial for agriculture and gardening. Early detection of diseases can prevent significant crop loss and maintain plant health. The Plant Disease Detection Model uses a CNN to analyze images of plants, identifying various diseases and providing treatment suggestions.

## Features

- ![Accuracy](https://img.icons8.com/color/48/000000/ok.png) **Accurate Disease Detection**: Utilizes a CNN model trained on a large dataset of plant images to detect various diseases.
- ![Speed](https://img.icons8.com/color/48/000000/fast-forward.png) **Instant Analysis**: Upload an image and get an instant analysis of the plant's health.
- ![Treatment](https://img.icons8.com/color/48/000000/syringe.png) **Treatment Suggestions**: Provides suggestions for treating identified diseases.
- ![Web](https://img.icons8.com/color/48/000000/internet.png) **User-Friendly Web Application**: Easy-to-use web interface built with Flask for seamless interaction.

## Working
![working](https://github.com/FajanSunusara/Plant_Disease_Detection_Model/assets/49346372/c71af147-6d11-417d-bc11-786900859e95)


## Installation

To run this project locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/Plant_Disease_Detection_Model.git
   cd Plant_Disease_Detection_Model
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Model**
   - Ensure you have the trained CNN model saved in the appropriate directory as specified in your code.

## Usage

### Running the Web Application

1. **Start the Flask Server**
   ```bash
   flask run
   ```

2. **Open Your Browser**
   - Navigate to `http://127.0.0.1:5000/`.

3. **Upload an Image**
   - Use the web interface to upload an image of a plant.
   - Receive an instant diagnosis and treatment suggestions.

## Model Details

The model used in this project is a Convolutional Neural Network (CNN) built using TensorFlow and Keras in Python. Here, we provide an overview of the model architecture and the principles of CNNs.

### Convolutional Neural Network (CNN) Overview

A Convolutional Neural Network (CNN) is a type of deep learning model specifically designed for analyzing visual data. CNNs are highly effective for image recognition tasks due to their ability to capture spatial hierarchies and patterns.

![cnn](https://github.com/FajanSunusara/Plant_Disease_Detection_Model/assets/49346372/38d4aa23-8607-4937-8510-d473ff3154f3)

### CNN Architecture

1. **Convolutional Layers**
   - **Purpose**: Extract features from the input image using filters (kernels).
   - **Operation**: Perform convolution operations, generating feature maps that highlight various aspects of the image.
   - ![Convolution Operation](https://img.icons8.com/color/48/000000/data-in-both-directions.png)

2. **Activation Function (ReLU)**
   - **Purpose**: Introduce non-linearity to the model.
   - **Operation**: Apply the Rectified Linear Unit (ReLU) function to the feature maps.
   - ![ReLU](https://img.icons8.com/color/48/000000/lightning-bolt.png)

3. **Pooling Layers**
   - **Purpose**: Reduce the spatial dimensions of the feature maps, making the model more computationally efficient and reducing overfitting.
   - **Operation**: Perform max pooling or average pooling.
   - ![Pooling](https://img.icons8.com/color/48/000000/compression.png)

4. **Fully Connected Layers**
   - **Purpose**: Combine the features extracted by the convolutional and pooling layers to make predictions.
   - **Operation**: Flatten the feature maps and pass them through dense layers.
   - ![Fully Connected Layers](https://img.icons8.com/color/48/000000/final-state-machine.png)

5. **Output Layer**
   - **Purpose**: Produce the final prediction.
   - **Operation**: Use a softmax activation function for classification.
   - ![Output Layer](https://img.icons8.com/color/48/000000/output.png)

### Model Implementation

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Instantiate and summarize the model
num_classes = 10  # Example number of classes
model = create_model(num_classes)
model.summary()
```

## Web Application

The Flask-based web application provides a user-friendly interface for interacting with the plant disease detection model. It includes:

- ![Upload](https://img.icons8.com/color/48/000000/upload.png) **Image Upload**: Upload plant images for analysis.
- ![Feedback](https://img.icons8.com/color/48/000000/feedback.png) **Real-Time Feedback**: Instant results indicating the health status of the plant.
- ![Suggestions](https://img.icons8.com/color/48/000000/treatment-plan.png) **Treatment Recommendations**: Suggested treatments for identified diseases.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please submit a pull request or open an issue. Follow these steps to contribute:

1. ![Fork](https://img.icons8.com/color/48/000000/code-fork.png) Fork the repository.
2. ![Branch](https://img.icons8.com/color/48/000000/git-branch.png) Create a new branch.
3. ![Commit](https://img.icons8.com/color/48/000000/git-commit.png) Make your changes.
4. ![Pull Request](https://img.icons8.com/color/48/000000/pull-request.png) Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact:

- **Your Name**
- ![Email](https://img.icons8.com/color/48/000000/email.png) **Email**: your.email@example.com
- ![GitHub](https://img.icons8.com/color/48/000000/github.png) **GitHub**: [your-username](https://github.com/your-username)

Thank you for using the Plant Disease Detection Model! Your contributions and feedback are highly appreciated.
```

This README file includes:
- Real URLs for icons from Icons8.
- Placeholder image URLs that can be replaced with actual images.
- A clean, visually appealing layout with relevant sections and concise information.
