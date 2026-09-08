# Deliverable 1 — Source Code and Jupyter Notebooks

## Contents

- `notebooks/01_Dataset_Audit.ipynb`: source inventory and duplicate audit.
- `notebooks/02_Data_Review.ipynb`: preparation of review materials.
- `notebooks/03_Combined_Training.ipynb`: clean notebook for a new combined training run, followed by model selection, evaluation and export.
- `notebooks/archive/Executed_Training_Record.ipynb`: unchanged executed experiment, including historical results and Colab recovery cells. This is evidence, not a Run All template.
- `Retail_Combined_App/`: Streamlit application source, inference, styles, dependencies and launchers.
- `training_assets/`: reviewed region annotations, label mappings and the notebook preprocessing helper.

## New Colab training run

Use the notebooks in the order Audit → Review → Combined Training. Configure the Drive paths inside each notebook before running. The training notebook expects the Drive project root `/content/drive/MyDrive/ITI_Retail_Project`, audit outputs under `dataset_audit/20260906_144352`, and a review ZIP under `data_review/20260906_150238_Data_Review.zip`. If generating new audit/review outputs, update these paths to the actual locations and verify that review image IDs match the supplied annotations.

ZIP the contents of this package's `training_assets` folder, with the five files directly at the archive root, into `Retail_Combined_Training_Pack.zip`. Place it in the Drive project root. The five Fruits-360 source ZIPs are also required there. Audit/Review datasets are handled in submission item 2.

The clean notebook removes two Colab-recovery cells, the fixed historical OUTPUT path in the plotting cell, and the automatic upload dialog call. It preserves the model architecture, training settings, split logic and selection rule. Its outputs have been cleared to avoid presenting earlier results as a rerun of the edited source. Training was not rerun during packaging.

For the presentation, open the unchanged executed record to show historical results, and use the clean notebook to explain the source flow. The saved model selected by validation was `transfer_head_best`. Natural-photo test accuracy was 35.34%; controlled accuracy was 48.25%. These measurements show limited generalization.

## Run the existing application

This is the source-code deliverable only. It does not duplicate the datasets, reports or model binary. To run it, use the complete existing application folder, or add its matching `models`, `reports` and `references` directories beside this package's `app.py`. Do not combine model exports from different training runs.

On Windows with Python 3.11, run `setup_windows.bat` once, then `run_windows.bat`. Local inference uses the CPU and requires no Colab session. The complete application already supplied remains the demonstration build.

## Review scope

Python source and notebook code syntax were checked. Colab shell/magic commands were excluded from Python AST checks. The executed training record and application sources are byte-identical to the submitted project. The clean notebook was reviewed statically but was not trained again. No model accuracy improvement is claimed.
