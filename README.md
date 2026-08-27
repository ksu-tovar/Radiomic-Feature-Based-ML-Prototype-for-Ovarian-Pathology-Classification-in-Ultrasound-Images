# Radiomic-Feature-Based ML Prototype for Ovarian Pathology Classification in Ultrasound Images

MATLAB prototype that extracts radiomic features (first-order, GLCM, LBP, morphology) from segmented ovarian ultrasound lesions and classifies them into three risk groups using SVM, Random Forest and KNN.

| Class | Label | Original MMOTU categories |
|---|---|---|
| 1 | Low risk | 4 simple cyst, 5 normal ovary |
| 2 | Borderline | 0 chocolate cyst, 1 serous cystadenoma, 2 teratoma, 3 theca cell tumor, 6 mucinous cystadenoma |
| 3 | Malignant | 7 high-grade serous carcinoma |

## Project structure

```
.
├── Main.m                          # Entry point, runs the pipeline stage by stage
├── Procesamiento.m                 # Reads images and masks, writes a feature CSV
├── ExtraerCaracteristicas.m        # Computes 22 radiomic features from an ROI
├── Entrenamiento.m                 # Grid-search training of SVM, RF and KNN
├── Validacion.m                    # Metrics and confusion matrices
├── Inferencia.m                    # Manual inference on a single image
├── APP.mlapp                       # App Designer GUI for inference
├── eda_smote.ipynb                 # EDA and SMOTE balancing (Python)
├── modeloYparametros.mat           # Trained models plus z-score mu and sigma
├── Dataset_train.csv               # 1000 rows (227 / 735 / 38 per class)
├── Dataset_train_balanceado.csv    # 1227 rows after SMOTE (340 / 735 / 152)
└── Dataset_val.csv                 # 469 rows (106 / 348 / 15)
```

## Requirements

MATLAB R2024b with:

- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
- Computer Vision Toolbox (for `extractLBPFeatures`)

Python 3.13 for the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## Data

Images and masks are not included in this repository. They come from the MMOTU **OTU_2d** subset: 1469 2D transvaginal ultrasound images from Beijing Shijitan Hospital, pre-split into 1000 training and 469 validation images.

- Dataset and download link: https://github.com/cv516Buaa/MMOTU_DS2Net
- Paper: https://arxiv.org/abs/2207.06799

Place the downloaded files in the working directory like this:

```
.
├── Imagenes/           # JPG ultrasound images
├── Anotaciones/        # PNG binary masks, same base filename as the image
└── txt files/
    ├── train_cls.txt   # "<filename> <class 0-7>" per line
    └── val_cls.txt
```

The upstream folders are named `images` and `annotations`, so rename them or edit the paths at the top of `Main.m`. Those paths use a Windows separator (`txt files\train_cls.txt`), so switch to `/` on macOS or Linux.

## How to run

`Main.m` contains every stage, with all but validation commented out. Open it, uncomment one block at a time, and run.

Steps 1 to 4 only need to be run once. The CSVs and the trained `.mat` are already committed, so you can go straight to step 5 or 6.

### 1. Extract features from the training split

Uncomment the TRAIN block in `Main.m`.

Reads `train_cls.txt`, crops each lesion to its bounding box, applies a 3x3 median filter, extracts 22 features and writes `Dataset_train.csv`.

### 2. Extract features from the validation split

Uncomment the VALIDATION block. Writes `Dataset_val.csv`.

### 3. Run EDA and SMOTE

Run `eda_smote.ipynb` from the repository root, cells in order.

It inspects class distributions and feature correlations, then applies SMOTE with `random_state=42` to oversample the 38 malignant and 227 low-risk cases, and writes `Dataset_train_balanceado.csv`.

This step is required before training. `Entrenamiento.m` reads the balanced CSV, and only the notebook produces it.

### 4. Train the models

Uncomment the training block.

Z-scores 15 of the 22 features, then grid-searches with stratified 5-fold cross-validation (seed 42):

- SVM with RBF and polynomial kernels, wrapped in ECOC
- Random Forest with 300 bagged trees
- KNN with k between 3 and 25

Saves all three models plus `mu` and `sigma` to `modeloYparametros.mat`.

### 5. Validate

This is the block currently active in `Main.m`.

Prints macro precision, sensitivity, specificity and F1 for each model, then draws three confusion matrices.

### 6. Run inference

Uncomment the last block, or call `Inferencia('modeloYparametros.mat')` directly.

Prompts for an image, opens a freehand tool to trace the lesion, and prints predictions from all three models.

## Using the app

Open `APP.mlapp` in MATLAB (double-click it, or run `appdesigner APP.mlapp`) and press **Run**.

Load an image, draw the lesion contour, then press **Diagnosticar**.

Run it from the repository root, since the app loads `modeloYparametros.mat` from the working directory. Note that the app predicts with the KNN model only, not with all three.

## Disclaimer

Research prototype. Not a diagnostic tool.
