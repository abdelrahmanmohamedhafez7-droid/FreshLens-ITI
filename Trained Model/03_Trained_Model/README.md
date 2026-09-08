# Deliverable 3 — Trained Model

`best_model.keras` is the original selected trained checkpoint, preserved byte-for-byte from the submitted combined model export. It is a complete Keras model, not just weights. It was selected using validation loss and is the model used by the final FreshLens application.

## Files

- `best_model.keras`: MobileNetV2 with frozen base (`transfer_head_best`).
- `class_names.json`: ordered list of 84 general product labels.
- `model_config.json`: input settings, versions, selection rule and limitations.
- `category_mapping.json`: original-to-general category mapping.
- `selected_model.txt`: selected training checkpoint name.

## Loading

Use TensorFlow 2.20.0 and Keras 3.13.2, matching the exported configuration:

```python
from tensorflow import keras
model = keras.models.load_model("best_model.keras", compile=False)
```

Image preprocessing must match the application: EXIF orientation correction, RGB conversion, PIL bilinear white padding to 160x160, JPEG encoding at quality 95, TensorFlow JPEG decoding and raw pixel values 0–255. Normalization is included in the model. Use the inference code from deliverable 1 or the complete application; do not substitute the older 100x100 preprocessing.

The file uses the native `.keras` format. No `.h5` conversion has been performed. If the evaluator specifically requires HDF5 rather than accepting a Keras model, a genuine conversion and prediction-equivalence check are required. Renaming the extension does not convert the format.

## Recorded performance

Controlled test accuracy: 48.25% (47,875 images, 37 categories).
Natural test accuracy: 35.34% (116 images, 17 categories).
The model has limited generalization; high confidence does not guarantee correctness.

## Packaging checks

The Keras archive integrity, output class count and selected-model metadata were checked. No retraining or new model inference was performed during this packaging step. The application inference was validated in the earlier application delivery.
