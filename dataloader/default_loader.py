from torch.utils.data import Dataset
import os
from PIL import Image
import pickle
import numpy as np
import torch


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


class CalibDataset(CustomImageDataset):
    def __init__(self, root_dir, transform=None, is_test=False, pickle_path: str = 'dataloader/valid_conf.pkl', temperature: float = 1.05):
        super().__init__(root_dir, transform, is_test)
        with open(pickle_path, 'rb') as f:
            confidences = pickle.load(f)

        self.confidences = self.temperature(confidences, temperature=temperature)

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
            return image, label, self.clib_label(label)

    def clib_label(self, label: int) -> np.ndarray:
        return self.confidences[label]

    @staticmethod
    def temperature(confidences: dict, temperature) -> dict:
        for key in confidences.keys():
            probs = confidences[key]
            logits = np.log(probs + 1e-12)

            scaled_logits = logits / temperature
            exp_logits = np.exp(scaled_logits)
            smoothed_probs = exp_logits / np.sum(exp_logits)
            confidences[key] = smoothed_probs

        return confidences


class LabelSmoothingCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred: (batch_size, num_classes)
        # target: (batch_size, num_classes)

        # Compute log softmax of the predictions
        log_prob = torch.nn.functional.log_softmax(pred, dim=-1)

        # Compute the negative log-likelihood (cross entropy)
        loss = -(target * log_prob).sum(dim=-1)
        return loss.mean()