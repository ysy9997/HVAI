import io
import os
import sys
import glob
import json
import utils
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import open_clip
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split



class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []

        if is_test:
            # 테스트셋: 라벨 없이 이미지 경로만 저장
            for fname in sorted(os.listdir(root_dir)):
                if fname.lower().endswith(('.jpg')):
                    img_path = os.path.join(root_dir, fname)
                    self.samples.append((img_path,))
        else:
            # 학습셋: 클래스별 폴더 구조에서 라벨 추출
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.jpg')):
                        img_path = os.path.join(cls_folder, fname)
                        label = self.class_to_idx[cls_name]
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            img_path = self.samples[idx][0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        else:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label



class CarClassifier(nn.Module):
    """Classifier head options for OpenCLIP features"""
    def __init__(self, input_dim=1152, num_classes=391, classifier_type='linear'):
        super().__init__()
        self.classifier_type = classifier_type
        
        if classifier_type == 'linear':
            # Simple linear classifier
            self.classifier = nn.Linear(input_dim, num_classes)
        elif classifier_type == 'mlp':
            # Multi-layer perceptron
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        elif classifier_type == 'attention':
            # Attention-based classifier
            self.attention = nn.MultiheadAttention(input_dim, num_heads=8, batch_first=True)
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def forward(self, x):
        if self.classifier_type == 'attention':
            # Add sequence dimension for attention
            x = x.unsqueeze(1)  # [B, 1, 1152]
            x, _ = self.attention(x, x, x)
            x = x.squeeze(1)  # [B, 1152]
        
        return self.classifier(x)



def fetch_model(hf_model_card, device):
    model, _, preprocess = open_clip.create_model_and_transforms(hf_model_card, device=device)
    tokenizer = open_clip.get_tokenizer(hf_model_card)
    return model, preprocess, tokenizer


def train_epoch(clip_model, classifier, train_loader, optimizer, criterion, device, freeze_clip=True):
    """Train for one epoch"""
    if freeze_clip:
        clip_model.eval()
    else:
        clip_model.train()
    
    classifier.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc=f"Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Extract features from CLIP
        with torch.set_grad_enabled(not freeze_clip):
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Forward through classifier
        outputs = classifier(image_features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(clip_model, classifier, val_loader, criterion, device):
    """Evaluate the model"""
    clip_model.eval()
    classifier.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validation"):
            images, labels = images.to(device), labels.to(device)
            
            # Extract features from CLIP
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Forward through classifier
            outputs = classifier(image_features)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def main(args):
    # fix random seed
    utils.seed_everything(args.seed)

    # create save directory
    os.makedirs(args.save_dir, exist_ok=True)


    device = torch.device(f'cuda:{args.device}')
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    
    # Load CLIP model
    clip_model, _, _ = fetch_model(args.hf_model_card, device=device)
    
    # Freeze CLIP parameters if specified
    if args.freeze_clip:
        for param in clip_model.parameters():
            param.requires_grad = False
        print("CLIP model parameters frozen")
    
    # 전체 데이터셋 로드
    full_dataset = CustomImageDataset(args.train_dir, transform=None)
    print(f"총 이미지 수: {len(full_dataset)}")

    targets = [label for _, label in full_dataset.samples]
    class_names = full_dataset.classes

    # Stratified Split
    train_idx, val_idx = train_test_split(
        range(len(targets)), test_size=0.2, stratify=targets, random_state=42
    )

    # Subset + transform 각각 적용
    train_dataset = Subset(CustomImageDataset(args.train_dir, transform=train_transform), train_idx)
    val_dataset = Subset(CustomImageDataset(args.train_dir, transform=val_transform), val_idx)
    print(f'train 이미지 수: {len(train_dataset)}, valid 이미지 수: {len(val_dataset)}')


    # DataLoader 정의
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Create classifier
    classifier = CarClassifier(
        input_dim=1152, 
        num_classes=391, 
        classifier_type=args.classifier_type
    ).to(device)
    
    # Setup training
    if args.freeze_clip:
        # Only optimize classifier parameters
        optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Optimize both CLIP and classifier with different learning rates
        clip_params = list(clip_model.parameters())
        classifier_params = list(classifier.parameters())
        optimizer = optim.Adam([
            {'params': clip_params, 'lr': args.lr * 0.1},  # Lower LR for CLIP
            {'params': classifier_params, 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            clip_model, classifier, train_loader, optimizer, criterion, device, args.freeze_clip
        )
        
        # Evaluate
        val_loss, val_acc, _, _ = evaluate(clip_model, classifier, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'classifier_state_dict': classifier.state_dict(),
                'clip_state_dict': clip_model.state_dict() if not args.freeze_clip else None,
                'args': args,
                'best_acc': best_acc
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"New best model saved! Accuracy: {best_acc:.2f}%")
    
    print(f"\nTraining completed. Best accuracy: {best_acc:.2f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier on OpenCLIP features for car classification")
    
    # random seed
    parser.add_argument('--seed', type=int, default=2025)
    
    # Data arguments
    parser.add_argument("--train_dir", type=str, default="/workspace/open/train", help="Training image directory")
    parser.add_argument("--save_dir", type=str, default="ViT-SO400M-14-SigLIP2", help="Directory to save models")
    
    # Model arguments
    parser.add_argument("--hf-model-card", type=str, default="hf-hub:timm/ViT-SO400M-14-SigLIP2", 
                       help="HuggingFace model card for OpenCLIP models")
    parser.add_argument("--classifier_type", type=str, default="linear", choices=["linear", "mlp", "attention"],
                       help="Type of classifier head")
    parser.add_argument("--freeze_clip", action="store_true", help="Freeze CLIP parameters")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--step_size", type=int, default=10, help="LR scheduler step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR scheduler gamma")
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for DataLoader")
    parser.add_argument("--device", type=str, default="0", help="GPU device")
    
    args = parser.parse_args()
    
    # save directory
    args.save_dir = args.save_dir + '/' + args.classifier_type
    
    main(args)