import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Custom dataset for chest X-ray images
class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_training (bool): If True, returns two augmented views of the same image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        
        # Get all image file paths
        self.image_paths = []
        for filename in os.listdir(root_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(root_dir, filename))
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.is_training:
            if self.transform:
                img1 = self.transform(image)
                img2 = self.transform(image)
                return img1, img2
        else:
            if self.transform:
                img = self.transform(image)
                return img, os.path.basename(img_path)
        
        return image

# Data augmentation for contrastive learning - adapted for medical images
class MedicalContrastiveTransform:
    def __init__(self, size=224):
        # Calculate kernel size and ensure it's odd
        kernel_size = int(0.1 * size)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Less aggressive augmentations for medical images
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.8, 1.0)),  # Less aggressive crop
            transforms.RandomHorizontalFlip(),  # Flipping is valid for chest X-rays
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),  # Gentler color jitter
            transforms.GaussianBlur(kernel_size=kernel_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.eval_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# SimCLR Implementation (Contrastive Learning)
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        
        # Base encoder (e.g., ResNet)
        self.encoder = base_encoder
        
        # Get the feature dimension from the base encoder
        if hasattr(base_encoder, 'fc'):
            feature_dim = base_encoder.fc.in_features
        else:
            feature_dim = 512  # Default for ResNet18
            
        # Remove the final classification layer
        self.encoder.fc = nn.Identity()
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)
    
    def get_features(self, x):
        return self.encoder(x)

# NT-Xent loss for SimCLR (Modified for stability)
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, eps=1e-8):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        
        # Check for NaN values
        if torch.isnan(z_i).any() or torch.isnan(z_j).any():
            print("Warning: NaN values in embeddings")
            # Replace NaNs with zeros
            z_i = torch.nan_to_num(z_i, nan=0.0)
            z_j = torch.nan_to_num(z_j, nan=0.0)
        
        # Full batch of embeddings
        features = torch.cat([z_i, z_j], dim=0)
        
        # Calculate similarity matrix with added stability
        sim_matrix = torch.matmul(features, features.t()) / (self.temperature + self.eps)
        
        # Numerical stability checks
        if torch.isnan(sim_matrix).any() or torch.isinf(sim_matrix).any():
            print("Warning: NaN or Inf values in similarity matrix")
            sim_matrix = torch.nan_to_num(sim_matrix, nan=0.0, posinf=1e5, neginf=-1e5)
        
        # Discard diagonal elements
        sim_matrix.fill_diagonal_(-float('inf'))
        
        # Create positive pairs
        pos_idxs = torch.arange(batch_size, device=z_i.device)
        
        # Labels for cross-entropy: positives are the corresponding indices from the other view
        labels = torch.cat([pos_idxs + batch_size, pos_idxs])
        
        # Check if sim_matrix has valid values for softmax
        if torch.isinf(sim_matrix).all(dim=1).any():
            # If a row has all inf values, replace with zeros (will give uniform distribution)
            inf_rows = torch.isinf(sim_matrix).all(dim=1)
            sim_matrix[inf_rows] = torch.zeros_like(sim_matrix[inf_rows])
        
        loss = self.criterion(sim_matrix, labels)
        
        # Avoid division by zero
        return loss / max(2 * batch_size, 1)

# Few-shot learning component with prototypical network approach
class FewShotLearner:
    def __init__(self, encoder, device):
        self.encoder = encoder
        self.encoder.eval()
        self.device = device
        self.normal_prototypes = None
    
    def compute_prototypes(self, normal_dataloader):
        features = []
        
        with torch.no_grad():
            for images, _ in normal_dataloader:
                images = images.to(self.device)
                feats = self.encoder.get_features(images)
                features.append(feats)
        
        if not features:
            raise ValueError("No features extracted from normal dataloader")
            
        features = torch.cat(features, dim=0)
        self.normal_prototypes = features.mean(dim=0)
        
        # Calculate covariance for Mahalanobis distance
        centered_features = features - self.normal_prototypes.unsqueeze(0)
        self.covariance = torch.matmul(centered_features.t(), centered_features) / (features.size(0) - 1)
        # Add stronger regularization to ensure invertibility
        self.covariance += torch.eye(self.covariance.size(0), device=self.device) * 1e-3
        
        # Check if covariance matrix is invertible
        try:
            self.precision = torch.inverse(self.covariance)
        except RuntimeError:
            print("Warning: Covariance matrix is not invertible. Using pseudo-inverse.")
            # Use pseudo-inverse as a fallback
            self.precision = torch.pinverse(self.covariance)
        
        return self.normal_prototypes
    
    def detect_anomalies(self, test_dataloader, threshold=None):
        anomaly_scores = []
        image_names = []
        
        with torch.no_grad():
            for images, names in test_dataloader:
                images = images.to(self.device)
                feats = self.encoder.get_features(images)
                
                # Calculate Mahalanobis distance for each sample
                for i, feat in enumerate(feats):
                    diff = feat - self.normal_prototypes
                    mahalanobis_dist = torch.sqrt(torch.mm(torch.mm(diff.unsqueeze(0), self.precision), diff.unsqueeze(1)))
                    anomaly_scores.append(mahalanobis_dist.item())
                    image_names.append(names[i])
        
        # If threshold is provided, return binary predictions
        if threshold is not None:
            predictions = [1 if score > threshold else 0 for score in anomaly_scores]
            return predictions, anomaly_scores, image_names
        
        return anomaly_scores, image_names

def train_simclr(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch_idx, (img1, img2) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            img1, img2 = img1.to(device), img2.to(device)
            
            # Forward pass
            z1 = model(img1)
            z2 = model(img2)
            
            # Compute loss
            loss = criterion(z1, z2)
            
            # Check for NaN or inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Loss is {loss.item()} at batch {batch_idx}. Skipping update.")
                continue
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
        # Print epoch statistics
        avg_loss = running_loss/max(len(dataloader), 1)  # Avoid division by zero
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Early stopping if loss is NaN or Inf
        if np.isnan(avg_loss) or np.isinf(avg_loss):
            print("Early stopping due to NaN or Inf loss")
            break
    
    return model

def main():
    # Configuration
    data_dir = "chest_xray_data"
    normal_samples_dir = os.path.join(data_dir, "normal")       # Healthy lung X-rays
    test_samples_dir = os.path.join(data_dir, "test")           # Mix of healthy and pneumonia
    few_shot_anomaly_dir = os.path.join(data_dir, "few_shot")   # Few examples of pneumonia
    
    # Create directories if they don't exist
    os.makedirs(normal_samples_dir, exist_ok=True)
    os.makedirs(test_samples_dir, exist_ok=True)
    os.makedirs(few_shot_anomaly_dir, exist_ok=True)
    
    # Check if directories contain data
    for dir_path, dir_name in [(normal_samples_dir, "normal"), 
                               (test_samples_dir, "test"), 
                               (few_shot_anomaly_dir, "few_shot")]:
        file_count = len([f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f"{dir_name} directory contains {file_count} images")
        
        # Create sample files for demonstration if directory is empty
        if file_count == 0 and dir_name == "few_shot":
            print(f"Warning: {dir_name} directory is empty. Creating sample files for demonstration.")
            # Create sample normal and pneumonia files for the few-shot directory
            sample_img = np.random.rand(224, 224, 3) * 255
            sample_img = Image.fromarray(sample_img.astype('uint8'))
            sample_img.save(os.path.join(dir_path, "sample_normal.jpg"))
            sample_img.save(os.path.join(dir_path, "sample_pneumonia.jpg"))
    
    # Smaller batch size and learning rate for stability
    batch_size = 16  # Reduced from 32
    lr = 1e-4  # Reduced from 3e-4
    weight_decay = 1e-4  # Increased from 1e-6
    epochs = 10
    projection_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create transforms
    transforms_obj = MedicalContrastiveTransform()
    
    # Create datasets
    train_dataset = ChestXrayDataset(
        root_dir=normal_samples_dir,
        transform=transforms_obj.train_transform,
        is_training=True
    )
    
    test_dataset = ChestXrayDataset(
        root_dir=test_samples_dir,
        transform=transforms_obj.eval_transform,
        is_training=False
    )
    
    few_shot_dataset = ChestXrayDataset(
        root_dir=few_shot_anomaly_dir,
        transform=transforms_obj.eval_transform,
        is_training=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=min(batch_size, len(train_dataset)),  # Ensure batch size isn't larger than dataset
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=min(batch_size, len(test_dataset)),
        shuffle=False, 
        num_workers=4
    )
    
    few_shot_loader = DataLoader(
        few_shot_dataset, 
        batch_size=min(batch_size, len(few_shot_dataset)),
        shuffle=False, 
        num_workers=4
    )
    
    # Print dataset sizes
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Few-shot samples: {len(few_shot_dataset)}")
    
    # Check if we have enough data to proceed
    if len(train_dataset) < 10:
        print("Warning: Very small training set. Results may not be reliable.")
        if len(train_dataset) == 0:
            print("Error: No training samples found. Exiting.")
            return
    
    # Initialize model
    # Base encoder (ResNet-18 instead of 50 for smaller dataset)
    base_encoder = models.resnet18(pretrained=True)
    
    # SimCLR model
    model = SimCLR(base_encoder=base_encoder, projection_dim=projection_dim).to(device)
    
    # Loss and optimizer
    criterion = NTXentLoss(temperature=0.1)  # Reduced temperature from 0.5 to 0.1 for better stability
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Only train if there are enough samples
    if len(train_dataset) >= batch_size:
        # Train
        model = train_simclr(model, train_loader, optimizer, criterion, device, epochs)
        
        # Save the model
        torch.save(model.state_dict(), "chest_xray_anomaly_detection.pth")
    else:
        print(f"Warning: Training dataset size {len(train_dataset)} is smaller than batch size {batch_size}. Skipping training.")
    
    # Few-shot learning for anomaly detection
    few_shot_learner = FewShotLearner(model, device)
    
    try:
        # Compute prototypes of normal samples
        normal_prototype = few_shot_learner.compute_prototypes(train_loader)
        
        # Detect anomalies in test set
        anomaly_scores, image_names = few_shot_learner.detect_anomalies(test_loader)
        
        # Plotting anomaly score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(anomaly_scores, bins=30, alpha=0.7)
        plt.title('Anomaly Score Distribution for Chest X-ray Images')
        plt.xlabel('Anomaly Score (Higher = More Likely Abnormal)')
        plt.ylabel('Count')
        plt.savefig('chest_xray_anomaly_scores.png')
        
        # Check if few_shot dataset has files
        if len(few_shot_dataset) == 0:
            print("Error: No few-shot samples found. Cannot determine optimal threshold.")
            return
        
        # Determine optimal threshold using a small set of labeled examples
        # Extract file names from the few-shot dataset
        few_shot_files = [os.path.basename(path) for path in few_shot_dataset.image_paths]
        
        # Create labels based on filenames
        few_shot_labels = []
        for name in few_shot_files:
            # Check if 'pneumonia' is in the filename (case insensitive)
            if 'pneumonia' in name.lower():
                few_shot_labels.append(1)  # Pneumonia (anomaly)
            else:
                few_shot_labels.append(0)  # Normal
        
        print(f"Few-shot labels distribution: {few_shot_labels}")
        
        # Run anomaly detection on few-shot samples
        few_shot_scores, few_shot_names = few_shot_learner.detect_anomalies(few_shot_loader)
        
        # Ensure both classes are present
        if len(set(few_shot_labels)) > 1:
            # Calculate ROC curve and find optimal threshold
            precision, recall, thresholds = precision_recall_curve(few_shot_labels, few_shot_scores)
            
            # Calculate F1 score for each threshold
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            
            # Find threshold with best F1 score
            if len(thresholds) > 0:
                optimal_idx = np.argmax(f1_scores[:-1])  # Skip the last element as precision_recall_curve returns n+1 precisions
                optimal_threshold = thresholds[optimal_idx]
                best_f1 = f1_scores[optimal_idx]
                
                print(f"Optimal threshold: {optimal_threshold} (F1 score: {best_f1:.4f})")
                
                # Apply threshold to get final predictions
                predictions, _, _ = few_shot_learner.detect_anomalies(test_loader, threshold=optimal_threshold)
                
                # Save results
                with open('chest_xray_anomaly_detection_results.txt', 'w') as f:
                    for name, score, pred in zip(image_names, anomaly_scores, predictions):
                        f.write(f"{name}: Score={score:.4f}, Prediction={'Potential Pneumonia' if pred == 1 else 'Normal'}\n")
                
                print("Anomaly detection completed. Results saved to 'chest_xray_anomaly_detection_results.txt'")
            else:
                print("Warning: No thresholds found. Cannot determine optimal threshold.")
        else:
            print("Warning: Few-shot samples don't contain both normal and anomalous examples. Can't determine optimal threshold.")
            print("Ensure the few_shot directory contains files with 'pneumonia' in their names for anomalous examples.")
            
    except Exception as e:
        print(f"Error during anomaly detection: {str(e)}")

if __name__ == "__main__":
    main()