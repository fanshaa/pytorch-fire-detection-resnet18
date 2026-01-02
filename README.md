# Fire Detection – Transfer Learning with ResNet18 (PyTorch)

This project demonstrates an end-to-end computer vision classification pipeline using PyTorch:
- ImageFolder dataset loading (normal/ vs fire/)
- Transfer learning with pretrained ResNet18
- Train/test split (stratified)
- Training + evaluation (accuracy, confusion matrix, classification report)
- Inference on a folder of test images
- Best model checkpoint saving

## Tech
Python, PyTorch, torchvision, scikit-learn

## Data
Dataset is not included for privacy reasons.
Expected structure:
disaster_dataset/
  normal/
  fire/

## Run
1) Update paths in `train.py`:
- `data_dir`
- `test_images_path`

2) Install dependencies:
pip install -r requirements.txt

3) Train:
python train.py
