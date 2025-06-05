import io
import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
 
import open_clip
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
 
 
 
class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess
 
    def __len__(self):
        return len(self.image_paths)
 
    def __getitem__(self, idx):
        # load path and parse metadata from path
        path = self.image_paths[idx]
        name = path.split('/')[-2]
 
        # fetch image & preprocess
        image = Image.open(path).convert("RGB")
        image = self.preprocess(image)
        return image, path



def main(args):
    # save directory
    os.makedirs(args.save_dir, exist_ok=True)
 
    # set device
    assert torch.cuda.is_available(), "CUDA device not available."
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device(f"cuda:{args.device}")
 
    # prepare model
    model, _, preprocess = open_clip.create_model_and_transforms(args.hf_model_card, device=device)

    # random transformation : overwrite openclip's validation preprocess
    if args.use_tta:
        preprocess = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.2), interpolation=Image.Resampling.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        args.epochs = 1  # single forward pass is enough if no augmentation is used
 
    # create dataset and dataloader
    image_paths = sorted(glob.glob(f"{args.image_dir}/**/*.jpg", recursive=True))
    image_dataset = ImageDataset(image_paths, preprocess)
    image_loader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
   
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    feature_dict = {}
    for epoch in range(args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}] (TTA {'ON' if args.use_tta else 'OFF'})")
        # loop through images
        for batch, (image, path) in tqdm(enumerate(image_loader), desc="Processing images"):
            image = image.to(device)
            with torch.no_grad():
                image_features = model.encode_image(image).cpu()
            for p,f in zip(path, image_features):
                if p not in feature_dict.keys():
                    feature_dict[p] = f
                else:
                    feature_dict[p] += f
    # get average of features
    for p in feature_dict.keys():
        feature_dict[p] /= args.epochs
    
    # save features
    save_name = "feature_dict_tta.pt" if args.use_tta else "feature_dict.pt"
    torch.save(feature_dict, f"{args.save_dir}/{save_name}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument("--image_dir", type=str, default="/workspace/open/train", help="Cropped image directory")
    # parser.add_argument("--save_dir", type=str, default="/home/haiqv/workspace/hvai/ViT-H-14-laion2B-s32B-b79K", help="Directory to save images with label disagreement")
    # parser.add_argument("--hf-model-card", type=str, default="hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K", help="HuggingFace model card for OpenCLIP models")
    parser.add_argument("--save_dir", type=str, default="ViT-SO400M-14-SigLIP2", help="HuggingFace model card for OpenCLIP models")
    parser.add_argument("--hf-model-card", type=str, default="hf-hub:timm/ViT-SO400M-14-SigLIP2", help="HuggingFace model card for OpenCLIP models")
 
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for processing images")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for DataLoader")
    parser.add_argument("--device", type=str, default="0")

    # TTA: Test-Time Augmentation to get robust embeddings
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)

    args = parser.parse_args()
 
    main(args)