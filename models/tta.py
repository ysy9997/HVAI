import torch
from torchvision import transforms
import torch
import torchvision.transforms.functional as TF
import configs.default_config as cfg


class TTAWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        """
        TTAWrapper for applying Test Time Augmentation (TTA) to a model.
        :param model: The model to apply TTA to
        """

        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, method: str = 'mean') -> torch.Tensor:
        """
        forward method for TTAWrapper
        :param image: input function
        :param method: method to combine predictions ('sum', 'mean')
        return input function output
        """

        if self.model.training:
            return self.model(image)

        image_agmented = [image, torch.flip(image.detach().clone(), dims=[-1])]
        image_agmented.append(self.adjust_brightness(image.detach().clone(), 1.1))
        image_agmented.append(self.adjust_brightness(image.detach().clone(), 0.9))

        preds = []
        for image in image_agmented:
            preds.append(self.model(image))
        total_pred = torch.sum(torch.stack(preds), dim=0) if method == 'sum' else torch.mean(torch.stack(preds), dim=0)

        return total_pred.squeeze(0)
    
    @staticmethod
    def adjust_brightness(image: torch.Tensor, factor: float) -> torch.Tensor:
        """
        Adjust brightness of the image.
        :param image: input image tensor
        :param factor: brightness adjustment factor
        :return: brightness adjusted image tensor
        """
        mean = torch.tensor(cfg.CFG['MEAN'], device=image.device).view(-1, 1, 1)
        std = torch.tensor(cfg.CFG['STD'], device=image.device).view(-1, 1, 1)

        return (image * std + mean * factor - mean) / std
    