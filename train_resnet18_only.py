import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms, models

# Define a custom dataset to load images and labels from CSV
class DeepfakePaintingDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        """
        Args:
            data_frame (pd.DataFrame): DataFrame with columns 'path', 'label', 'split', etc.
            transform (callable, optional): Transform to apply to images.
        """
        self.data = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['path']
        label_str = row['label']
        # Convert label to an integer (e.g., 1 for "real", 0 for "fake")
        label = 1 if label_str.lower() == 'real' else 0
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Define image transforms (ResNet18 expects 224x224 images)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_datasets(csv_file):
    # Read the CSV file containing image paths, labels, and splits
    df = pd.read_csv(csv_file)
    # Split the data based on the 'split' column (case-insensitive)
    train_df = df[df['split'].str.lower() == 'train']
    test_df = df[df['split'].str.lower() == 'test']
    train_dataset = DeepfakePaintingDataset(train_df, transform=transform)
    test_dataset = DeepfakePaintingDataset(test_df, transform=transform)
    return train_dataset, test_dataset

# Define the ResNet18-based classifier
def get_resnet18_model(num_classes=2):
    model = models.resnet18(pretrained=True)
    # Replace the final fully connected layer with one matching our task
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
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
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total if total > 0 else 0
    print(f"Test Accuracy: {acc * 100:.2f}%")
    return acc

if __name__ == '__main__':
    # This ensures compatibility with Windows and macOS when using DataLoader with multiple workers.
    # import multiprocessing
    # multiprocessing.freeze_support()

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    csv_file = "deepfake_dataset.csv"  # Ensure this CSV file exists with proper columns
    train_dataset_raw, test_dataset = load_datasets(csv_file)
    total_size = len(train_dataset_raw)
    train_size  = int(0.8 * total_size)
    test_size   = total_size - train_size

    train_dataset, test_dataset = random_split(train_dataset_raw, [train_size, test_size])

    # Create DataLoaders (if you run into issues with multiple workers, set num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # print(train_dataset.size)
    # print(test_dataset.size)

    # Get the model, move it to the device
    model = get_resnet18_model(num_classes=2).to(device)
    in_features = model.fc.in_features

    # Replace model.fc with Dropout + Linear
    model.fc = nn.Sequential(
        nn.Dropout(p=0.1),           # drop 50% of features at random
        nn.Linear(in_features, 2)    # your 2‑class output
    )
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    num_epochs = 10
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

    # Evaluate the model on the test set
    evaluate_model(model, test_loader, device)
