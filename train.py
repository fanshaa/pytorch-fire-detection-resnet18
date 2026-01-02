# =========================================
# Fire Detection with ResNet18 (PyTorch)
# =========================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# --------------------------------
# 1) Reproducibility
# --------------------------------
def set_seed(seed=44):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(44)


# --------------------------------
# 2) Device
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------
# 3) Paths (CHANGE THESE)
# --------------------------------
data_dir = r"C:\Users\Chadwick\Downloads\fire archive\disaster_dataset"
test_images_path = r"C:\Users\Chadwick\Downloads\fire archive\test_images"


# --------------------------------
# 4) Image Transforms
# --------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------
# 5) Load Dataset
# --------------------------------
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes   # ['normal', 'fire']
print("Classes:", class_names)

targets = dataset.targets
indices = list(range(len(dataset)))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=44,
    stratify=targets
)

train_dataset = Subset(dataset, train_idx)
test_dataset  = Subset(dataset, test_idx)


# --------------------------------
# 6) DataLoaders
# --------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)


# --------------------------------
# 7) Model (Transfer Learning)
# --------------------------------
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

model = model.to(device)


# --------------------------------
# 8) Loss & Optimizer
# --------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# --------------------------------
# 9) Training Loop
# --------------------------------
epochs = 5
best_acc = 0.0

for epoch in range(1, epochs + 1):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    # Evaluation
    model.eval()
    correct_t, total_t = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total_t += labels.size(0)
            correct_t += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100.0 * correct_t / total_t

    print(
        f"Epoch [{epoch}/{epochs}] | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Test Acc: {test_acc:.2f}%"
    )

    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "fire_detection_resnet18.pth")

print(f"\nBest Test Accuracy: {best_acc:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))


# --------------------------------
# 10) Inference on New Images
# --------------------------------
if os.path.isdir(test_images_path):
    print("\nRunning inference on test images...")
    model.load_state_dict(
        torch.load("fire_detection_resnet18.pth", map_location=device)
    )
    model.eval()

    image_files = [
        f for f in os.listdir(test_images_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for img_name in image_files:
        img_path = os.path.join(test_images_path, img_name)
        img = Image.open(img_path).convert("RGB")
        tensor_img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor_img)
            _, pred = torch.max(output, 1)

        print(f"{img_name}: {class_names[pred.item()]}")
