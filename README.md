# FER2013 Facial Emotion Recognition

This project recognizes facial emotions in real time with a Keras/TensorFlow CNN trained on the [FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) dataset. In display mode it reads frames from a webcam, finds faces with an OpenCV Haar cascade, classifies each 48x48 grayscale face crop into one of seven emotions (angry, disgusted, fearful, happy, neutral, sad, surprised), and draws the label above the face. The repository also includes trained weights for ResNet50 and AlexNet variants (`resnet.h5`, `alexnet.h5`).

The code is based on the open-source [Emotion-detection](https://github.com/atulapra/Emotion-detection) project by atulapra (see [Acknowledgements](#acknowledgements)).

## How it works

- **Data preparation.** `dataset_prepare.py` converts the FER-2013 CSV (`fer2013.csv`, 48x48 pixel strings) into PNG images, one folder per emotion. The first 28,709 rows become the training set and the remaining rows the test set.
- **Model.** `emotions.py` defines a sequential CNN: Conv(32) → Conv(64) → MaxPool → Dropout(0.25) → Conv(128) → MaxPool → Conv(128) → MaxPool → Dropout(0.25) → Dense(1024) → Dropout(0.5) → Dense(7, softmax).
- **Training.** Input is rescaled to [0, 1] with Keras `ImageDataGenerator`. The model trains with Adam (learning rate 1e-4, decay 1e-6) on categorical cross-entropy for 50 epochs at batch size 64, validating on the test folder. Accuracy and loss curves are saved to `plot.png`, and the weights to `model.h5`.
- **Real-time inference.** Each webcam frame is converted to grayscale, and faces are detected with `haarcascade_frontalface_default.xml`. Each face is resized to 48x48 and classified, and the predicted emotion is drawn on a 1600x960 preview window. Press `q` to quit.

## Results

`src/plot.png` shows the accuracy and loss curves for the CNN over 50 epochs. Validation accuracy levels off partway through training while training accuracy keeps rising, so later epochs mostly overfit.

![CNN training curves](src/plot.png)

`src/plot1.png` and `src/plot2.png` hold two more accuracy/loss plots in the same format. The repository does not record which model produced them.

## Repository structure

```
.
├── requirements.txt
└── src/
    ├── emotions.py                          # CNN definition, training (--mode train), webcam demo (--mode display)
    ├── dataset_prepare.py                   # FER-2013 CSV -> per-emotion PNG folders
    ├── haarcascade_frontalface_default.xml  # OpenCV frontal face detector
    ├── model.h5                             # trained CNN weights used by --mode display
    ├── resnet.h5                            # trained ResNet50-variant weights
    ├── alexnet.h5                           # trained AlexNet-variant weights
    └── plot.png, plot1.png, plot2.png       # training curves
```

The code that builds and trains the ResNet50 and AlexNet variants is not part of this repository. `emotions.py` only supports the `train` and `display` modes for the CNN above.

## Setup

`requirements.txt` pins the original environment:

```bash
pip install -r requirements.txt
```

This installs `numpy==1.17.4`, `opencv-python==4.2.0.32` and `tensorflow==2.5.3`. `emotions.py` also imports `matplotlib`, and `dataset_prepare.py` imports `pandas`, `Pillow` and `tqdm`. Install those separately if you need them.

## Data

Download `fer2013.csv` from the [Kaggle FER-2013 challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge). `emotions.py` expects the images under `src/data/` with one folder per emotion:

```
src/data/
├── train/{angry,disgusted,fearful,happy,neutral,sad,surprised}/*.png   # 28,709 images
└── test/{angry,disgusted,fearful,happy,neutral,sad,surprised}/*.png    # 7,178 images
```

`dataset_prepare.py` reads `./fer2013.csv` and creates the `data/train/<emotion>` and `data/test/<emotion>` folders. However, it saves images to `train/<emotion>/` and `test/<emotion>/` relative to the working directory. Either change the save paths to `data/train/...` and `data/test/...` or move the generated folders under `src/data/` before training.

## Usage

Run everything from `src/`. The data paths, weight file and Haar cascade are all loaded relative to that folder.

```bash
cd src
python emotions.py --mode train     # trains the CNN, writes model.h5 and plot.png
python emotions.py --mode display   # webcam demo using model.h5
```

## Acknowledgements

- The training/inference script, the dataset conversion script, the CNN architecture and the pinned `requirements.txt` follow [atulapra/Emotion-detection](https://github.com/atulapra/Emotion-detection). Refer to that repository for its license terms.
- `haarcascade_frontalface_default.xml` is the OpenCV frontal face cascade by Rainer Lienhart and is distributed under the Intel License Agreement included in the file header.
- FER-2013 was introduced in I. Goodfellow et al., "Challenges in Representation Learning: A report on three machine learning contests", ICONIP 2013.
