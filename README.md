# Playing Card Classifier

A PyTorch image classifier that uses a convolutional neural network to identify playing cards from images, via transfer learning.

## Project structure

- **`app/dataset.py`** (`PlayingCardDataset`): wraps `torchvision.datasets.ImageFolder` to load card images from `dataset/train`, `dataset/valid`, and `dataset/test` directories, where each subfolder name is treated as a class label (e.g. one folder per card such as "ace of spades").
- **`app/classifier.py`**: code shared by training and prediction:
  - **Preprocessing** (`transform`): resizes every image to 128x128 and converts it to a tensor via `torchvision.transforms`.
  - **Model** (`SimpleCardClassifier`): a neural network built from a pretrained `efficientnet_b0` backbone (via `timm`, a convolutional neural network) with its classification head removed, followed by a single linear layer that maps the 1280-dimensional feature output to the number of card classes (53 by default).
  - **Device selection** (`get_device`): uses an NVIDIA GPU (CUDA) if available, then Apple Silicon GPU (MPS), otherwise CPU.
  - **`CHECKPOINT_PATH`**: location of the saved model file, `card_classifier.pth` in the project root.
- **`app/model.py`** (training): trains the model for 5 epochs using the Adam optimizer and cross-entropy loss. After each epoch it evaluates on the validation set and prints the training/validation loss. When training finishes, it saves the learned weights and class names to `card_classifier.pth`.
- **`app/predict.py`** (testing): loads the saved weights from `card_classifier.pth`, predicts the class of a single image, and shows a matplotlib figure with the image next to the probability of each class. It doesn't retrain.

## Requirements

- Python 3.10
- `torch`, `torchvision`, `timm`, `numpy`, `matplotlib`, `tqdm`

## Dataset setup

The `dataset/` folder is not committed to this repo (it's listed in `.gitignore`) because it's too large. Before running `model.py`, you need to download it yourself:

1. Download the dataset from Kaggle: [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification?resource=download)
2. Unzip it and place it in a `dataset/` folder in the project root, matching the layout below.

## Data layout

```
dataset/
  train/
    <class_name>/*.jpg
  valid/
    <class_name>/*.jpg
  test/
    <class_name>/*.jpg
```

## Usage

Run all commands from the project root (the folder containing `dataset/`).

### 1. Train the model (once)

```
python app/model.py
```

This trains the model and saves its weights and class names to `card_classifier.pth` in the project root. The file is ignored by git.

To retrain the model, for example after changing the number of epochs or the dataset, simply rerun `python app/model.py`. It overwrites `card_classifier.pth` with the new weights.

### 2. Test the model on an image

```
python app/predict.py
```

This loads the saved weights from `card_classifier.pth` without retraining and shows a matplotlib figure with the image next to the predicted probability of each class. By default it uses `dataset/test/five of spades/4.jpg`. To test a different image, pass its path:

```
python app/predict.py "dataset/test/ace of spades/1.jpg"
```

If `card_classifier.pth` doesn't exist yet, `predict.py` tells you to run `model.py` first.
