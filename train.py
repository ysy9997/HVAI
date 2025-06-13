import os
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn, optim
import processingtools as pt

from sklearn.metrics import log_loss
import configs.default_config as cfg
import utils
import timm
import dataloader.default_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

utils.seed_everything(cfg.CFG['SEED']) # Seed 고정

recoder = pt.EnvReco(cfg.CFG['SAVE_PATH'], varify_exist=False)
recoder.record_gpu()


if __name__ == "__main__":
    train_root = '/workspace/dataset/train'
    test_root = '/workspace/dataset/test'
    resume = ''

    train_transform = transforms.Compose([
        transforms.Resize((cfg.CFG['IMG_SIZE'], cfg.CFG['IMG_SIZE'])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(cfg.CFG['IMG_SIZE'], scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.CFG['MEAN'],
                            std=cfg.CFG['STD'])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((cfg.CFG['IMG_SIZE'], cfg.CFG['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.CFG['MEAN'],
                            std=cfg.CFG['STD'])
    ])

    # 전체 데이터셋 로드
    full_dataset = dataloader.default_loader.CalibDataset(train_root, transform=None)
    print(f"총 이미지 수: {len(full_dataset)}")

    targets = [label for _, label in full_dataset.samples]
    class_names = full_dataset.classes

    # Stratified Split
    train_idx, val_idx = train_test_split(
        range(len(targets)), test_size=0.2, stratify=targets, random_state=42
    )

    # Subset + transform 각각 적용
    train_dataset = Subset(dataloader.default_loader.CalibDataset(train_root, transform=train_transform), train_idx)
    val_dataset = Subset(dataloader.default_loader.CalibDataset(train_root, transform=val_transform), val_idx)
    print(f'train 이미지 수: {len(train_dataset)}, valid 이미지 수: {len(val_dataset)}')


    # DataLoader 정의
    train_loader = DataLoader(train_dataset, batch_size=cfg.CFG['BATCH_SIZE'], shuffle=True, num_workers=cfg.CFG['NUM_WORKERS'])
    val_loader = DataLoader(val_dataset, batch_size=cfg.CFG['BATCH_SIZE'] * 2, shuffle=False, num_workers=cfg.CFG['NUM_WORKERS'])


    model = timm.create_model('convnext_base', pretrained=True)
    model.head.fc = nn.Linear(model.head.in_features, len(class_names), bias=True)
    model = model.to(device)
    best_logloss = float('inf')

    # 손실 함수
    criterion = dataloader.default_loader.LabelSmoothingCrossEntropyLoss()

    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=cfg.CFG['LEARNING_RATE'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True, min_lr=1e-6
    )

    start_epoch = 0
    if resume:
        # 모델과 옵티마이저 상태를 로드
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_logloss = checkpoint['best_logloss']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        recoder.print(f"Resuming from epoch {start_epoch}")

    # 학습 및 검증 루프
    for epoch in range(start_epoch, cfg.CFG['EPOCHS']):
        # Train
        model.train()
        train_loss = 0.0
        for images, labels, confidences in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{cfg.CFG['EPOCHS']}] Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # logits
            loss = criterion(outputs, confidences)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels, confidences in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{cfg.CFG['EPOCHS']}] Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, confidences)
                val_loss += loss.item()

                # Accuracy
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # LogLoss
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_names))))

        scheduler.step(val_logloss)
        current_lr = optimizer.param_groups[0]['lr']

        # 결과 출력
        recoder.print(f"[{epoch + 1}/{cfg.CFG['EPOCHS']}] Train Loss : {avg_train_loss:.4f} || Valid Loss : {avg_val_loss:.4f} | Valid Accuracy : {val_accuracy:.4f}%")

        # Best model 저장
        if val_logloss < best_logloss:
            best_logloss = val_logloss
            save_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch, 'best_logloss': best_logloss, 'scheduler_state_dict': scheduler.state_dict(),}
            torch.save(save_dict, os.path.join(cfg.CFG['SAVE_PATH'], 'best_model.pth'))
            recoder.print(f"📦 Best model saved at epoch {epoch+1} (logloss: {val_logloss:.4f})")

        print(f"{pt.s_text(f'Current LR: {current_lr:.6f}', f_rgb=(100, 10, 80))} | {pt.s_text(f'Best LogLoss: {best_logloss:.4f}', f_rgb=(10, 100, 80))} | {pt.s_text(f'Current LogLoss: {val_logloss:.4f}', f_rgb=(10, 80, 200))}", end='\n\n')

    test_dataset = dataloader.default_loader.CalibDataset(test_root, transform=val_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.CFG['BATCH_SIZE'] * 2, shuffle=False, num_workers=cfg.CFG['NUM_WORKERS'])

    # 저장된 모델 로드
    model = timm.create_model('convnext_base', pretrained=True)
    model.head.fc = nn.Linear(model.head.in_features, len(class_names), bias=True)
    model.load_state_dict(torch.load(os.path.join(cfg.CFG['SAVE_PATH'], 'best_model.pth'), map_location=device)['model_state_dict'])
    model.to(device)

    # 추론
    model.eval()
    results = []
    need2erase = ['718_박스터_2017_2024', 'K5_3세대_하이브리드_2020_2022', 'RAV4_2016_2018', 'RAV4_5세대_2019_2024', '디_올뉴니로_2022_2025']
    out_of_distribution_classes = '_cat_or_dog'
    out_of_distribution_classes_idx = class_names.index(out_of_distribution_classes)
    class_names = [_ for _ in class_names if _ != out_of_distribution_classes]

    pt.sprint(f'start inference on test dataset.', f_rgb=(128, 128, 0), styles=['tilt'])
    with torch.no_grad():
        for images in tqdm(test_loader, total=len(test_loader)):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            # 각 배치의 확률을 리스트로 변환
            for prob in probs.cpu():  # prob: (num_classes,)
                if torch.argmax(prob) == out_of_distribution_classes_idx:
                    prob = torch.ones_like(prob) / len(class_names)
                prob = torch.cat([prob[:out_of_distribution_classes_idx], prob[out_of_distribution_classes_idx + 1:]])

                result = {
                    class_names[i]: prob[i].item()
                    for i in range(len(class_names))
                }
                results.append(result)

                for key in need2erase:
                    result[key] = 0.0  # 필요 없는 클래스 확률을 0으로 설정
                
    pred = pd.DataFrame(results)

    submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')

    # 'ID' 컬럼을 제외한 클래스 컬럼 정렬
    class_columns = submission.columns[1:]
    pred = pred[class_columns]

    submission[class_columns] = pred.values
    submission.to_csv(os.path.join(cfg.CFG['SAVE_PATH'], 'baseline_submission.csv'), index=False, encoding='utf-8-sig')
