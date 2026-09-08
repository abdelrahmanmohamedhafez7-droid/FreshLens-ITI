# FreshLens

### Retail Product Recognition with Computer Vision

FreshLens is an ITI training project that classifies a single fruit or vegetable product image into one of **84 general categories**. It combines dataset auditing, label review, CNN training, transfer learning, and an interactive Streamlit application.

> **Project status:** Educational prototype. Performance on natural photographs remains limited. Predictions require human review and are not suitable for unattended checkout.
**[Application Link](https://freshlens-iti.streamlit.app/)**
## Contents

- [Features](#features)
- [Dataset access](#dataset-access)
- [Methodology](#methodology)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Installation and launch](#installation-and-launch)
- [Training notebooks](#training-notebooks)
- [Limitations](#limitations)

## Features

- Single-image classification into general product categories.
- Top-three predictions with model scores.
- Light and dark interface modes.
- Training curves, model comparisons, and prediction-error examples.
- Evidence and reference material for reviewing model behavior.
- Exported model, class labels, and preprocessing configuration.

Model scores are not a guarantee of correctness. The application does not locate or count multiple products in one photograph.

## Dataset access

**[Download the complete dataset from Google Drive](https://drive.google.com/drive/folders/1_qpaKwM5-O5TCzvFriyjNELGpvj2Fkwu?usp=sharing)**

The complete dataset is hosted separately to avoid distributing hundreds of thousands of images through Git.

| Folder | Contents |
| --- | --- |
| `Original/` | Original Fruits-360 source archives |
| `Processed/Data_Review/` | Reviewed labels and source-image review records |
| `Processed/Dataset_Audit/` | Inventories, duplicate checks, source documentation, and audit summaries |
| `Processed/prepared_827d2a4ded81c2e1/` | Prepared images, manifest, cache identifier, and read-error record |
| `Processed/class_coverage.csv` | Class coverage information |
| `Processed/split_manifest.csv` | Saved training, validation, and test assignments |

Original archives:

- `fruits-360-100x100-main.zip`
- `fruits-360-original-size-main.zip`
- `fruits-360-3-body-problem-main.zip`
- `fruits-360-multi-main.zip`
- `fruits-360-meta-main.zip`

These sources have different roles. The metadata source supplies reference information, rather than training images. Exact duplicates are removed; downloading five archives does not mean every source contributes additional unique training images.

The preparation stage produced **289,908 images/regions before exact deduplication**, including **626 reviewed natural-photo regions**. These figures describe preparation output, not the final training-set size.

The application does **not** require the complete dataset. Download the dataset to reproduce data preparation or training, and configure the paths in the notebooks. Existing Google Colab paths may refer to `/content/drive/MyDrive/ITI_Retail_Project`.

Preserve the source license and attribution files supplied with the dataset. This repository does not replace the original dataset licenses.

## Methodology

1. Audit source archives, labels, image readability, and duplicates.
2. Map product varieties to general categories and review natural-photo regions.
3. Prepare RGB images at **160 × 160** pixels while preserving aspect ratio with white padding.
4. Freeze the dataset splits and remove exact duplicates.
5. Train a CNN from scratch and a MobileNetV2 transfer-learning model.
6. Fine-tune part of the MobileNetV2 backbone as a separate training stage.
7. Select a model using validation results, then evaluate the fixed test subsets.
8. Export the selected model and its matching labels and preprocessing configuration.

Controlled images use grouping by normalized variety. Natural-photo splits use capture-year information: 2018 for validation and 2022 for testing, with other eligible known years used for training. This is a more demanding evaluation than a random image split; it does not establish independence of every physical product across sources.

The models are:

| Identifier | Approach |
| --- | --- |
| `cnn_best` | CNN trained from scratch |
| `transfer_head_best` | ImageNet-pretrained MobileNetV2 with a frozen backbone and trained classification head |
| `transfer_fine_best` | MobileNetV2 with partial backbone fine-tuning |

The selected export is **`transfer_head_best`**. Selection uses the mean of controlled and natural validation losses, rather than choosing the highest test accuracy.

## Results

### Final combined-dataset test evaluation

| Model | Controlled accuracy | Natural accuracy | Controlled macro F1 | Natural macro F1 |
| --- | ---: | ---: | ---: | ---: |
| CNN | 24.18% | 9.48% | 13.65% | 4.81% |
| MobileNetV2 — frozen backbone | **48.25%** | **35.34%** | **34.39%** | **25.17%** |
| MobileNetV2 — fine-tuned | 50.40% | 31.03% | 36.84% | 25.23% |

Bold values identify the selected model, not the maximum in every column.

| Evaluation subset | Images/regions | Represented categories |
| --- | ---: | ---: |
| Controlled validation | 36,172 | 25 |
| Natural validation | 81 | 25 |
| Controlled test | 47,875 | 37 |
| Natural test | 116 | 17 |

Although the classifier has 84 outputs, each evaluation subset covers only the categories listed above. Natural test accuracy for the selected model corresponds to **41 correct predictions out of 116**. The small sample limits conclusions about general real-world performance.

The fine-tuned model has the highest controlled test accuracy, while the frozen-backbone model has higher natural test accuracy. Earlier experiments with different splits are not directly comparable to these final results.

Training and validation curves show overfitting. The evaluation also exposes difficulty generalizing between controlled and natural photographs. The project reports these limitations rather than presenting training accuracy as deployment accuracy.

## Repository structure

| Path | Purpose |
| --- | --- |
| `README.md` | Project overview, results, dataset access, and setup |
| `FreshLens_App/Retail_Combined_App/` | Streamlit application and its supporting files |
| `FreshLens_App/Retail_Combined_App/models/` | Application model and matching configuration |
| `FreshLens_App/Retail_Combined_App/reports/` | Metrics, curves, manifests, and visual examples |
| `FreshLens_App/Retail_Combined_App/references/` | Reference metadata |
| `notebooks/` | Audit, review, and combined-training notebooks |
| `notebooks/training_assets/` | Mapping files and reviewed training inputs |
| `notebooks/excution 03/` | Executed training record; folder spelling matches the submitted project |
| `Trained Model/03_Trained_Model/` | Standalone selected-model export |
| `demo_images/` | Images for demonstrating predictions |
| `FreshLens_ITI_Presentation.pptx` | Project presentation |
| `Project Report.docx` | Written project report |

The complete `datasets/` directory is distributed through Google Drive. Local virtual environments and Python caches are excluded from Git.

## Installation and launch

Use **Python 3.11**. The following commands are for **Windows CMD**.

### 1. Download the repository

Git LFS is needed for the model and tracked report files.

```bat
git lfs install
git clone https://github.com/abdelrahmanmohamedhafez7-droid/FreshLens-ITI.git
cd FreshLens-ITI
git lfs pull
```

If you already have the project folder, open CMD there and skip cloning. If LFS downloads fail, check the error and the repository's LFS availability; a pointer file is not a usable trained model.

### 2. Create a virtual environment

```bat
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

### 3. Install dependencies

```bat
python -m pip install tensorflow==2.20.0 keras==3.13.2 streamlit==1.49.1 pillow==11.3.0 pandas==2.3.2 numpy==2.2.6 altair==5.5.0
python -m pip check
```

### 4. Start the application

```bat
cd FreshLens_App\Retail_Combined_App
python -m streamlit run app.py
```

Open the local address printed by Streamlit. Upload a single-product image, run recognition, and review the predictions and evidence. CPU inference is supported; full training is better run with a GPU in Colab.

Keep the complete application folder together, including `models`, `reports`, `references`, and its configuration files. The standalone model export is also provided for assessment; the application uses its own `models` folder.

## Training notebooks

Run these notebooks in order when reproducing the workflow:

1. `notebooks/01_Dataset_Audit.ipynb`
2. `notebooks/02_Data_Review.ipynb`
3. `notebooks/03_Combined_Training.ipynb`

The executed training record is provided separately in `notebooks/excution 03/Executed_Training_Record.ipynb`. Configure dataset and output paths before running. Preserve the saved split assignments when comparing results. Training can take substantial time, and repeated runs may differ because of random initialization and runtime behavior.

Use the exported model for the application; retraining is not required to launch the demo.

## Limitations

- Predictions on unfamiliar photographs can be wrong, even with a high score.
- Natural-photo test coverage is limited to 116 regions across 17 categories.
- A clean background does not guarantee a correct prediction.
- The classifier always chooses among known categories; it is not a reliable unknown-product detector.
- Image classification is not object detection, counting, weighing, or price estimation.
- The application should not be used as an autonomous retail checkout system.

## Business relevance and future work

FreshLens demonstrates a potential product-identification assistant for retail workflows. A practical deployment would require broader independent natural-image evaluation, better coverage of difficult categories, calibrated uncertainty, and human confirmation. No checkout-time savings or business revenue improvement was measured in this project.

Future work includes improving split and label quality, testing stronger regularization and augmentation, evaluating class imbalance, and collecting representative retail photographs. Progress should be measured on a fixed independent evaluation set.

## Project deliverables

- Source notebooks and application code.
- Original and processed datasets through Google Drive.
- Selected trained model in `.keras` format with label and preprocessing files.
- Written report and presentation.
- Evaluation reports and demonstration images.

Team and supervisor details are included in the project report.
