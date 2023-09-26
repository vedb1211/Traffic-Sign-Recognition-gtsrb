# Traffic-Sign-Recognition-gtsrb

## Introduction

This project focuses on Traffic Sign Recognition using machine learning models. The goal is to recognize and classify traffic signs into different categories. We explore various models and techniques for this task, including Convolutional Neural Networks (CNN), Transfer Learning with Inception, and Regularized CNN.

## Getting Started

### Prerequisites

Before running the code, you need to have the following prerequisites installed:

- Python (3.6+)
- TensorFlow (2.0+)
- Pandas
- Matplotlib
- Pillow
- Numpy
- Jupyter Notebook (optional)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/vedb1211/traffic-sign-recognition-gtsrb.git
   cd traffic-sign-recognition-gtsrb
   pip install -r requirements.txt



# Download the dataset from Kaggle
!kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

# Visualizing 25 Images
![image](https://github.com/vedb1211/Traffic-Sign-Recognition-gtsrb/assets/106091820/be835406-426c-4c32-adb5-952fdbdd5102)


# Extract the dataset
zip = zipfile.ZipFile('gtsrb-german-traffic-sign.zip', 'r')<br>
zip.extractall()<br>
zip.close()

# Run this code to organize the test data
python organize_test_data.py


# Model Training
Data Augmentation: 
We apply data augmentation techniques to the training data, including rotation, shifting, shearing, zooming, and flipping, to increase the dataset's diversity

![image](https://github.com/vedb1211/Traffic-Sign-Recognition-gtsrb/assets/106091820/f773614c-57f6-48e6-a391-24ba0b68bb60)

Model Architectures:
We experiment with different model architectures, including ANN-Model, CNN-Model, Inception, Fine-tune Inception, and Regularized-CNN. Each model is defined and compiled with appropriate loss functions, optimizers, and metrics.

Training and Evaluation:
We train each model on the training data and evaluate its performance on the validation and test datasets. Loss and accuracy metrics are plotted to visualize model performance.

Model Comparison:
We compare the performance of different models using bar graphs. Two separate graphs are generated, one for loss and the other for accuracy.

# Conclusion
In this project, we explored various machine learning models for traffic sign recognition. We organized the data, implemented data augmentation, trained multiple models, and compared their performance. The results provide insights into the effectiveness of different model architectures for this task.

Feel free to experiment with different hyperparameters, model architectures, or techniques to further improve the accuracy of traffic sign recognition.

# Acknowledgments
Dataset: GTSRB - German Traffic Sign Recognition Benchmark<br>
TensorFlow: https://www.tensorflow.org/<br>
Kaggle: https://www.kaggle.com/

