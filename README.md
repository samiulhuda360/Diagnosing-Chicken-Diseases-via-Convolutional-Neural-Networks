# Poultry Disease Detection with a Convolutional Neural Network

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samiulhuda360/Diagnosing-Chicken-Diseases-via-Convolutional-Neural-Networks/blob/main/Chichen_disease_detection.ipynb)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Task](https://img.shields.io/badge/task-image%20classification-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Deep learning image classifier that screens chickens for Coccidiosis and Salmonella from photos of their
droppings: a cheap, non-invasive early-warning tool for poultry farms.**

Coccidiosis and Salmonella spread fast through a flock and are usually caught late, after birds are visibly
sick. Faecal samples change appearance early, so a phone photo plus a trained model can flag a problem days
before a vet visit. This project builds and evaluates that model end to end in TensorFlow/Keras.

| | |
|---|---|
| **Problem** | 3-class image classification: healthy, Coccidiosis, Salmonella |
| **Data** | 2,536 faecal images (919 healthy, 814 Coccidiosis, 803 Salmonella), 80/20 train/validation split |
| **Model** | Convolutional neural network built from scratch: 3 convolution + max-pooling blocks, 512-unit dense layer, dropout 0.5, softmax |
| **Result** | **84.8% validation accuracy**, macro F1 0.83, ROC-AUC 0.97 / 0.96 / 0.91 (Coccidiosis / healthy / Salmonella) |

![Sample images labelled Coccidiosis](docs/images/samples-coccidiosis.png)

## Results

| Class | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Coccidiosis | 0.85 | **0.94** | 0.90 | 0.97 |
| Healthy | 0.91 | 0.75 | 0.82 | 0.96 |
| Salmonella | 0.72 | 0.79 | 0.76 | 0.91 |
| **Overall** | 0.83 | 0.83 | **0.83** | |

| Confusion matrix (validation, 505 images) | ROC curves, one-vs-rest |
|---|---|
| ![Confusion matrix](docs/images/confusion-matrix.png) | ![ROC curves](docs/images/roc-curves.png) |

**Reading the results.** Coccidiosis is caught 94% of the time, which matters most on a farm because it spreads
fastest. The main weakness is that 41 of 183 healthy samples were flagged as Salmonella: a false alarm costs a
check, a missed infection costs a flock, so the error leans the safer way, but it is the first thing to improve.

![Training and validation curves](docs/images/training-curves.png)

## How it works

1. **Data checks:** class balance, image-size distribution, colour histograms per class, and a pass that
   removes corrupted files.
2. **Preprocessing and augmentation** with `ImageDataGenerator`: resize to 300 x 300, rescale to [0, 1], random
   rotation, shifts, zoom and horizontal flips, so the model learns the sample and not the photo angle.
3. **CNN training** with Adam, categorical cross-entropy and early stopping on validation loss.
4. **Evaluation:** accuracy, per-class precision, recall and F1, confusion matrix and one-vs-rest ROC-AUC.

## Run it

The notebook runs on Google Colab with a free GPU: click **Open in Colab** above, upload the image archive to your
Google Drive, and update `zip_path` in the third cell. The archive is expected to contain `healthy/`, `cocci/` and
`salmo/` folders. Locally:

```bash
pip install -r requirements.txt
jupyter notebook Chichen_disease_detection.ipynb
```

The dataset is not included in this repository.

## Known issues and next steps

- **Single-image prediction cell:** the cell that classifies an uploaded photo reuses a validation prediction and
  maps class indices in a different order from the training generator (`cocci`, `healthy`, `salmo`). Use
  `train_generator.class_indices` to label predictions. Fixing this is the first item on the list below.
- **Transfer learning:** a pretrained backbone (EfficientNet or ResNet) fine-tuned on this data should beat a
  from-scratch CNN trained for 5 epochs, especially on the healthy vs Salmonella confusion.
- **Held-out test set** and per-farm splits, so the score reflects photos from farms the model has never seen.
- **Web demo:** upload a photo, get the class, the confidence and a heat map of the region the model used.

## Tech stack

Python · TensorFlow / Keras · OpenCV · NumPy · pandas · scikit-learn · Matplotlib · seaborn · Google Colab

**Keywords:** computer vision, deep learning, convolutional neural network, image classification, data
augmentation, agriculture AI, AgriTech, poultry health, animal disease detection, Coccidiosis, Salmonella,
precision and recall, ROC-AUC, confusion matrix.

## Author

[Samiul Huda](https://github.com/samiulhuda360) · MSc Data Science · Auckland, New Zealand. MIT licence.
