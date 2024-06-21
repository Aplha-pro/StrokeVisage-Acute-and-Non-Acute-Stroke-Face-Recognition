# StrokeVisage: Acute and Non-Acute Stroke Face Recognition

## Overview
**Course**: CS456 - Computer Vision (A2)  
**Submitted to**: Mr. Zeeshan Khan  
**Submitted by**: Muhammad Ahmed (F2021376015)  

**Department**: Artificial Intelligence  
**University**: University of Management and Technology, Lahore  

## Introduction

### Background
Stroke is a critical medical condition characterized by the sudden loss of blood flow to the brain, leading to potential brain damage and loss of function. Prompt and accurate diagnosis is crucial to initiate treatment and improve patient outcomes. However, traditional diagnostic methods such as CT scans and MRIs are often time-consuming and require significant medical resources. With advancements in machine learning and computer vision, there is an opportunity to develop non-invasive, rapid diagnostic tools that can assist in early stroke detection.

### Objectives
The primary objective of "StrokeVisage" is to develop an automated system that uses deep learning techniques to classify facial images into acute stroke and non-stroke categories. This system aims to provide a quick and reliable preliminary diagnostic tool that can be used in various healthcare settings.

## Problem Statement
Stroke diagnosis is traditionally reliant on imaging techniques and clinical evaluations that are resource-intensive and not always immediately available. This project addresses the following problems:

1. **Timely Diagnosis**: Reducing the time taken to diagnose a stroke, thereby potentially improving patient outcomes.
2. **Resource Allocation**: Minimizing the reliance on expensive and less accessible imaging technologies.
3. **Non-Invasive Methods**: Providing a non-invasive method for preliminary stroke detection.

## Related Work
The application of deep learning in medical image analysis has shown promising results in various fields:

- **Tumor Detection**: CNNs have been successfully used to detect tumors in medical images, demonstrating high accuracy.
- **Retinal Disease Diagnosis**: Deep learning models have been employed to identify retinal diseases from fundus photographs.
- **Skin Cancer Classification**: CNNs have also been used to classify skin lesions with performance comparable to dermatologists.

However, the specific application of CNNs for stroke detection using facial images is relatively unexplored. Some studies have focused on using physiological signals and imaging data, but the unique approach of analyzing facial features for stroke classification represents a novel research direction.

## Tools and Techniques

### Data Collection
The dataset for this project will be sourced from the Kaggle dataset: [Face Images of Acute Stroke and Non-Acute Stroke](https://www.kaggle.com/datasets/danish003/face-images-of-acute-stroke-and-non-acute-stroke). This dataset contains labeled facial images of individuals categorized into acute stroke and non-acute stroke.

### Data Preprocessing
- **DataFrame Creation**: Organize images into a pandas DataFrame with corresponding labels.
- **Train-Test Split**: Split the dataset into training and testing sets using an 80-20 ratio with stratification to ensure balanced class distribution.
- **Data Augmentation**: Apply augmentation techniques (rescaling, rotation, flipping, zooming) to enhance the model's robustness.

### Model Development
A Convolutional Neural Network (CNN) will be developed with the following architecture:
1. **Input Layer**: Accepts 128x128 RGB images.
2. **Convolutional Layers**: Multiple layers with ReLU activation and max pooling for feature extraction.
3. **Fully Connected Layers**: Dense layers for classification, including dropout for regularization.
4. **Output Layer**: A sigmoid activation function to output probabilities for binary classification.

### Model Training
- **Compilation**: Use the Adam optimizer and binary cross-entropy loss function.
- **Training**: Train the model for a specified number of epochs with early stopping to prevent overfitting, and validate on the test set.

### Model Evaluation
Evaluate the model using metrics such as accuracy, F1 score, precision, recall, and ROC AUC score. Additionally, the model will be tested on individual images to demonstrate its practical application.

### Prediction Function
Develop a function to predict the class of a single image:
1. **Load and Preprocess**: Load the image, resize it to 128x128, convert it to a numpy array, and rescale pixel values.
2. **Predict and Output**: Use the trained model to predict the class and output the predicted label and confidence score.

## Conclusion
"StrokeVisage" aims to leverage deep learning and computer vision to develop a non-invasive, rapid diagnostic tool for stroke detection using facial images. By addressing the limitations of current diagnostic methods, this project has the potential to significantly enhance early stroke detection, improve patient outcomes, and optimize resource allocation in healthcare. The successful implementation of this project could pave the way for further research and development in the field of medical image analysis and non-invasive diagnostics.

---

For more information, visit the [Kaggle dataset](https://www.kaggle.com/datasets/danish003/face-images-of-acute-stroke-and-non-acute-stroke).
