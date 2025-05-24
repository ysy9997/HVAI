import os
import pandas as pd

from PIL import Image
from tqdm import tqdm 

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn, optim
import processingtools as pt

from sklearn.metrics import log_loss
import configs.default_config as cfg
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

utils.seed_everything(cfg.CFG['SEED']) # Seed 고정

recoder = pt.EnvReco(cfg.CFG['SAVE_PATH'], varify_exist=False)
recoder.record_gpu()

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


train_root = '/workspace/dataset/train'
test_root = '/workspace/dataset/test'

train_transform = transforms.Compose([
    transforms.Resize((cfg.CFG['IMG_SIZE'], cfg.CFG['IMG_SIZE'])),
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
full_dataset = CustomImageDataset(train_root, transform=None)
print(f"총 이미지 수: {len(full_dataset)}")

targets = [label for _, label in full_dataset.samples]
class_names = full_dataset.classes

# Stratified Split
train_idx, val_idx = train_test_split(
    range(len(targets)), test_size=0.2, stratify=targets, random_state=42
)

# Subset + transform 각각 적용
train_dataset = Subset(CustomImageDataset(train_root, transform=train_transform), train_idx)
val_dataset = Subset(CustomImageDataset(train_root, transform=val_transform), val_idx)
print(f'train 이미지 수: {len(train_dataset)}, valid 이미지 수: {len(val_dataset)}')


# DataLoader 정의
train_loader = DataLoader(train_dataset, batch_size=cfg.CFG['BATCH_SIZE'], shuffle=True, num_workers=cfg.CFG['NUM_WORKERS'])
val_loader = DataLoader(val_dataset, batch_size=cfg.CFG['BATCH_SIZE'] * 2, shuffle=False, num_workers=cfg.CFG['NUM_WORKERS'])


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)  # ResNet18 모델 불러오기
        self.feature_dim = self.backbone.fc.in_features 
        self.backbone.fc = nn.Identity()  # feature extractor로만 사용
        self.head = nn.Linear(self.feature_dim, num_classes)  # 분류기

    def forward(self, x):
        x = self.backbone(x)       
        x = self.head(x) 
        return x


model = BaseModel(num_classes=len(class_names)).to(device)
best_logloss = float('inf')

# 손실 함수
criterion = nn.CrossEntropyLoss()

# 옵티마이저
optimizer = optim.Adam(model.parameters(), lr=cfg.CFG['LEARNING_RATE'])

# 학습 및 검증 루프
# for epoch in range(cfg.CFG['EPOCHS']):
#     # Train
#     model.train()
#     train_loss = 0.0
#     for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{cfg.CFG['EPOCHS']}] Training"):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)  # logits
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     avg_train_loss = train_loss / len(train_loader)

#     # Validation
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0
#     all_probs = []
#     all_labels = []

#     with torch.no_grad():
#         for images, labels in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{cfg.CFG['EPOCHS']}] Validation"):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()

#             # Accuracy
#             _, preds = torch.max(outputs, 1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#             # LogLoss
#             probs = F.softmax(outputs, dim=1)
#             all_probs.extend(probs.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     avg_val_loss = val_loss / len(val_loader)
#     val_accuracy = 100 * correct / total
#     val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_names))))

#     # 결과 출력
#     recoder.print(f"[{epoch + 1}/f{cfg.CFG['EPOCHS']}] Train Loss : {avg_train_loss:.4f} || Valid Loss : {avg_val_loss:.4f} | Valid Accuracy : {val_accuracy:.4f}%")

#     # Best model 저장
#     if val_logloss < best_logloss:
#         best_logloss = val_logloss
#         torch.save(model.state_dict(), os.path.join(cfg.CFG['SAVE_PATH'], 'best_model.pth'))
#         recoder.print(f"📦 Best model saved at epoch {epoch+1} (logloss: {val_logloss:.4f})")

test_dataset = CustomImageDataset(test_root, transform=val_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=cfg.CFG['BATCH_SIZE'] * 2, shuffle=False, num_workers=cfg.CFG['NUM_WORKERS'])

# 저장된 모델 로드
model = BaseModel(num_classes=len(class_names))
model.load_state_dict(torch.load(os.path.join(cfg.CFG['SAVE_PATH'], 'best_model.pth'), map_location=device))
model.to(device)

# 추론
model.eval()
results = []

with torch.no_grad():
    for images in tqdm(test_loader, total=len(test_loader)):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        # 각 배치의 확률을 리스트로 변환
        for prob in probs.cpu():  # prob: (num_classes,)
            result = {
                class_names[i]: prob[i].item()
                for i in range(len(class_names))
            }
            results.append(result)
            
pred = pd.DataFrame(results)

submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')

# 'ID' 컬럼을 제외한 클래스 컬럼 정렬
class_columns = submission.columns[1:]
# pred = pred[class_columns]

submission[class_columns] = pred.values
submission.to_csv(os.path.join(cfg.CFG['SAVE_PATH'], 'baseline_submission.csv'), index=False, encoding='utf-8-sig')
