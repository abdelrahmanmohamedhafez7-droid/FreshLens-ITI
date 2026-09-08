# FreshLens — Combined Model Edition

A local English interface with Light / Dark mode, recognition, and evidence for all three training runs.

## Windows setup

1. Extract the ZIP into a new folder. Do not mix it with the older app.
2. Install Python 3.11 (64-bit), including the Python launcher, if needed.
3. Double-click `setup_windows.bat` once. Internet access is needed to install dependencies.
4. Double-click `run_windows.bat`. Keep its terminal window open.
5. Open http://localhost:8501 if the browser does not open automatically.

For later sessions, use `run_windows.bat`. No Colab, Drive mount, GPU or retraining is needed. You can also open this folder in VS Code and use the same launchers.

## Existing environment

If the old app already runs, extract this edition into a separate folder, open a terminal there and run the old environment's Python with `-m streamlit run app.py`. Keep its dependency versions consistent with requirements.txt.

## Interface

- The sidebar Dark mode switch changes the appearance; turn it off for Light mode.
- Recognize: upload JPG, JPEG, PNG or WEBP and click Analyze product.
- Model results & evidence: comparisons by domain, curves for each model, learning rates, selected-model confusion matrices, per-class reports, errors and downloads.
- About: prediction steps, supported categories and scope.

The selected model is MobileNetV2 with a frozen base. Its saved test accuracy is 48.25% on controlled images and 35.34% on 116 natural images across 17 categories. These figures show limited generalization; a high prediction score does not guarantee correctness.

The preprocessing matches the new notebook: EXIF orientation, RGB, white padding to 160x160, JPEG quality 95, TensorFlow JPEG decoding, normalization inside the model. Do not replace this with the old 100x100 preprocessing.

Learning curves and aggregate scores are available for all three runs. Per-class reports and example errors are only available for the selected model. No missing measurements are invented. Raw photos and unselected model weights are not included.

Dataset attribution and licenses are in `references`. Original exported reports are kept in `reports`.
