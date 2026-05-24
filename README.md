# MRI Classification

A deep learning project for classifying MRI images using TensorFlow/Keras. This project processes DICOM medical imaging data and uses convolutional neural networks to perform binary classification on MRI scans.

## Project Overview

This project implements a machine learning pipeline for MRI image classification, including:
- DICOM image preprocessing and normalization
- CNN-based model architecture with batch normalization and dropout regularization
- Model training with callbacks for checkpointing and early stopping
- Inference and prediction generation for test datasets

## Files Description

### MRI.ipynb
The main Jupyter notebook containing:
- **Setup**: Installation of dependencies 
- **Data Loading**: Reading and processing DICOM files
- **Preprocessing**: 
  - Image resizing and cropping to 256x256 pixels
  - Normalization to [0, 1] range
  - Handling variable number of slices
- **Model Architecture**: CNN with convolutional layers, max pooling, batch normalization, and dropout
- **Training**: Model training with data augmentation and callbacks
- **Evaluation**: Performance metrics, confusion matrix, and classification reports
- **Visualization**: Exploratory data analysis and results plotting

### submission.py
A standalone Python script for making predictions on new MRI data:
- Loads the pre-trained model (`model.best.h5`)
- Processes DICOM series from disk
- Applies the same preprocessing pipeline as training
- Generates probability predictions for each series
- Outputs results as a CSV file

### model.best.h5
Model with optimized weights saved during training. Used by `submission.py` for inference.

---

### Making Predictions (Standalone Script)

Use the `submission.py` script to generate predictions on test data:

```bash
python submission.py --series_path /path/to/test/data --csv_path output.csv
```

**Arguments:**
- `--series_path`: Path to directory containing DICOM series
- `--csv_path`: Path to save output predictions CSV

**Output CSV Format:**
```
SeriesInstanceUID,prediction
series_1,0.95
series_2,0.42
...
```

## Future Improvements

- Implement transfer learning with pretrained models
- Implement cross-validation for robust evaluation
- Add `class_weight` to `model.fit()` to weight minority class higher during training without data loss
- Data Augmentation: Apply rotations, flips, ... to minority slices to increase effective training data while preserving patterns
- Replace `BinaryCrossentropy` with focal loss to focus training on hard-to-classify examples
- Threshold Tuning: Adjust decision threshold from 0.5 to optimize for specific business metrics (precision vs recall trade-off)
- Balanced Batch Sampling: Ensure each batch has equal class representation to improve gradient estimates for minority class
- Oversampling: Generate realistic minority samples instead of simple duplication
- A **hybrid approach** (combining light undersampling + class weighting + oversampling) would be more effective while preserving data.
- Create REST API for model serving

## Contact
For contributions, questions, or collaborations, please don't hesitate to reach out via email :) hosseini.sc95@gmail.com