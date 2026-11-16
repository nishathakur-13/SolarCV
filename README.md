# Solar Cell Defect Classification

This project uses transfer learning with **ResNet50** to classify electroluminescence (EL) images of photovoltaic cells as **good** or **faulty**. The notebook also reports validation metrics, saves the trained model, and visualizes predictions with Grad-CAM heatmaps.

The implementation is in [`Solar_CV_2.ipynb`](Solar_CV_2.ipynb), a Python 3 Jupyter/Google Colab notebook configured to use a GPU when one is available.

## What the notebook does

1. Downloads and extracts the [ELPV dataset](https://github.com/zae-bayern/elpv-dataset) into `elpv_data/`.
2. Reads the dataset's `labels.csv` and maps its per-image defect probability to a binary label: values greater than zero are labeled `Faulty_Cell`; zero values are labeled `Good_Cell`.
3. Resizes images to 224 × 224 pixels and applies the ResNet50 preprocessing function.
4. Trains a binary classifier in two stages: first with the ImageNet-pretrained ResNet50 backbone frozen, then with the `conv5` layers unfrozen at a lower learning rate.
5. Prints a classification report, plots a confusion matrix, and saves the model as `solar_crack_detection_resnet50.h5`.
6. Displays Grad-CAM visualizations for individual and randomly selected images.

## Run it

### Google Colab

1. Open `Solar_CV_2.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU). A GPU is optional, but training will generally be slower on CPU.
3. Run the notebook cells from top to bottom. The first cell downloads the dataset; training and visualization cells depend on the variables created earlier in the notebook.

### Local Jupyter

Use Python 3 and install the notebook's dependencies:

```bash
python -m pip install jupyter tensorflow pandas numpy matplotlib seaborn scikit-learn opencv-python
```

Then launch Jupyter and open the notebook:

```bash
jupyter notebook Solar_CV_2.ipynb
```

Run cells in order. TensorFlow uses a compatible GPU automatically when its GPU support is installed and available; otherwise it runs on CPU. The notebook downloads the dataset from GitHub, so the first run requires network access.

## Outputs

- **Console:** number of image files found, matched image/label count, and a scikit-learn classification report.
- **Plots:** confusion matrix and Grad-CAM overlays.
- **Model file:** `solar_crack_detection_resnet50.h5`, saved in the notebook's current working directory.

## Data and label interpretation

The notebook converts the ELPV dataset's label probability into a binary target using `probability > 0`. This means any image with a nonzero probability is treated as faulty; it does not preserve the original probability as a soft label. The model's sigmoid output is a score for this binary target, not a calibrated estimate of real-world defect risk.

## Important limitations

- The notebook takes the first 20% of the sorted image list for validation and uses the remainder for training. This is not a randomized or stratified split, and the same split is not guaranteed to represent independent cells or acquisition conditions. Treat the reported validation metrics as exploratory rather than as a reliable estimate of deployment performance.
- The label lookup uses image basenames. If different dataset folders contain duplicate filenames, labels can be matched incorrectly. The notebook also assumes that `labels.csv` exists and that every listed image can be matched.
- The notebook does not set random seeds, so training and sampled visualizations can vary between runs.
- Grad-CAM is a qualitative visualization of model activation, not proof that a highlighted region caused the prediction or identifies a physical defect.
- This is an educational/research workflow. Validate on a carefully designed, independent dataset before using predictions to make inspection or operational decisions.

## Repository contents

```text
.
├── README.md
└── Solar_CV_2.ipynb
```

