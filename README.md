# Dogs vs. Cats Redux - Image Classification with Transfer Learning

This repository contains a deep learning pipeline for classifying dog and cat images using transfer learning on the ResNet50 architecture. It was built for the Kaggle competition [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition).

## Project Overview

The goal of this competition is to build a binary image classifier that can distinguish between images of dogs and cats. The model is trained on a dataset of labeled images and evaluated on an unseen test set.

## Dataset

- **Source**: [Kaggle Dogs vs. Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)
- **Training Set**: 25,000 labeled images (dogs and cats)
- **Test Set**: 12,500 unlabeled images for submission

## Model Architecture

- **Base model**: ResNet50 (ImageNet pretrained)
- **Transfer learning**:
  - Initial phase: Freeze all layers except the top classifier head
  - Fine-tuning: Unfreeze deeper layers for final optimization
- **Loss Function**: Binary Crossentropy
- **Evaluation Metric**: Accuracy

## Key Steps

1. Load and preprocess images using `ImageDataGenerator`
2. Build the model using `ResNet50` as the backbone
3. Train in two phases (frozen and unfrozen)
4. Generate predictions and format for Kaggle submission

## Results

- Achieved validation accuracy of over 95% on a 20% holdout set
- Final test predictions submitted in CSV format for Kaggle evaluation

## Dependencies

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn
