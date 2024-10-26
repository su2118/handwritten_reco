# Handwritten Recognition Using Artificial Neural Networks

This project involves the development of a handwritten digit recognition system using Convolutional Neural Networks (CNNs), a type of Artificial Neural Network (ANN). The system is trained on the **MNIST dataset** to recognize handwritten digits from 0 to 9 and can predict these digits from both static images and live drawings. The project is implemented in Python using machine learning libraries such as TensorFlow and Keras.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Tools and Libraries](#tools-and-libraries)
- [Future Work](#future-work)
- [Contributing](#contributing)

## Project Overview

Handwritten digit recognition is a widely studied problem in machine learning and pattern recognition. This project aims to develop a high-accuracy model that can recognize handwritten digits from images or live inputs. The model is built using a CNN based on the **LeNet-5 architecture**, with additional layers to improve performance.

Key features of the project:
- **Digit recognition** from static images and live drawings.
- **High accuracy** achieved through deep learning using CNN.
- Real-time predictions using a pre-trained model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/su2118/handwritten_reco.git
   cd handwritten_reco
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
1. ### Training the model
   If you'd like to train the model from scratch, you can use the `ModelHR.py` file to build and train the model based on the **MNIST dataset**:
   
   ```bash
   python ModelHR.py

3. ### Running Predictions
   To run the real-time handwritten digit recognition using the trained model, use the       `software.py` file:
   
    ```bash
   python software.py
   
This will launch a canvas where you can draw digits, and the system will predict the digit      using the pre-trained model.

### Files Description
1. LeNetv2.csv - This file contains the saved weights and biases from training the LeNet-5 based CNN model. It stores the pre-trained model parameters which are used for testing or real-time predictions without needing to retrain the model.

2. ModelHR.py - This is the script that builds and trains the Convolutional Neural Network (CNN) using the MNIST dataset. It contains the architecture for the CNN and the training process. After training, it saves the trained model into a file.

3. software.py - This script runs a graphical user interface (GUI) that allows users to draw digits on a canvas, and the system will predict the drawn digit using the pre-trained model stored in LeNetv2.csv. It uses OpenCV to capture the drawing and TensorFlow/Keras to perform the prediction.

## Model Architecture

The model is a deep convolutional neural network (CNN) based on LeNet-5, enhanced with additional layers for better performance. The architecture includes:

- Convolutional layers for feature extraction
- Max pooling layers for down-sampling
- Fully connected layers for final classification
- Softmax output layer for predicting the digit class (0-9)

## Results
The model achieved 99% accuracy on the MNIST test set. The training and validation accuracies improve consistently with each epoch.

For detailed results, refer to the output from `ModelHR.py` or the logs generated during training.

## Tools and Libraries
The following tools and libraries were used in this project:

- Python 3.8
- TensorFlow 2.x - for building and training the CNN model
- Keras - for model creation and neural network utilities
- OpenCV - for capturing live drawings
- Matplotlib - for visualizing training results
- NumPy, Pandas, Scikit-learn - for data processing

## Future Work
Possible future enhancements include:

- Extending the system for multi-language handwritten character recognition.
- Developing mobile or web-based applications to make the model more accessible.
- Improving accuracy further by using larger datasets or transfer learning.

## Contributing
Contributions are welcome! If you have any ideas or suggestions, feel free to open an issue or submit a pull request.
