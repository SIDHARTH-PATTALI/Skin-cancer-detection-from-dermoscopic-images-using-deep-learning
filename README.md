# Skin Cancer Detection Using Deep Learning

This project implements a skin cancer detection system using deep learning. It utilizes transfer learning with AlexNet and ResNet50 for classification, along with Fuzzy C-means for image segmentation. The system includes a MATLAB GUI for users to upload images, analyze them, and display the predicted skin cancer type and confidence score.

## Motivation

- Skin cancer is more common than other cancers combined.
- Early detection significantly increases the survival rate.
- This tool helps in early detection, offering an accessible way to assess mole images.

## Data

The model is trained on a dataset of labeled skin lesion images, including both malignant and benign moles.do data augmenations.

## Development

- **Data Preprocessing**: The images are resized, cropped, and preprocessed for model training.
- **CNN Model**: A custom CNN model is developed and fine-tuned using transfer learning with pre-trained models like AlexNet and ResNet50 to enhance accuracy.
- **Evaluation**: The model's performance is assessed using metrics like Confussion matrix and AUC score to determine the best threshold for classification.

## Requirements

- MATLAB (version 2020 or later)
- Deep Learning Toolbox
- Image Processing Toolbox
