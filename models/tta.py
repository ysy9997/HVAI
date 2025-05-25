import torch
from torchvision import transforms
import configs.default_config as cfg
import warnings
import torch


class TTAWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        """
        TTAWrapper for applying Test Time Augmentation (TTA) to a model.
        :param model: The model to apply TTA to
        """

        self.model = model
        self.device = next(self.model.parameters()).device

        # 기본적으로 항상 적용할 transform (리사이즈, 텐서 변환, 정규화 등)
        self.default_transform = transforms.Compose([
            transforms.Resize((cfg.CFG['IMG_SIZE'], cfg.CFG['IMG_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.CFG['MEAN'], std=cfg.CFG['STD']),
        ])

        # TTA로 적용할 transform만 리스트로 정의
        self.tta_transforms = [
            transforms.RandomHorizontalFlip(p=1.0),
            # transforms.RandomVerticalFlip(p=1.0), # 필요시 추가
            # transforms.RandomRotation(15),        # 필요시 추가
        ]

        # 최종적으로 적용할 transform 조합 리스트 (기본, 기본+TTA)
        self.transforms_list = [self.default_transform]
        for tta in self.tta_transforms:
            self.transforms_list.append(
                transforms.Compose([
                    tta,
                    *self.default_transform.transforms  # 기본 transform 뒤에 붙이기
                ])
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        forward method for TTAWrapper
        :param image: input function
        return input function output
        """

        if self.model.training:
            warnings.warn("TTAWrapper should be used in evaluation mode. Switching to eval mode.")
        self.model.eval()

        preds = []
        with torch.no_grad():
            for t in self.transforms_list:
                img = t(image).unsqueeze(0).to(self.device)
                output = self.model(img)
                prob = torch.softmax(output, dim=1)
                preds.append(prob.cpu())
        mean_pred = torch.mean(torch.stack(preds), dim=0)

        return mean_pred.squeeze(0)