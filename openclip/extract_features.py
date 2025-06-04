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
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(Path(__file__).parent)  # for utils/class_mapping
from class_mapping import class_mapping
 
 
 
# global variables
MAKES = list(set(class_mapping.values()))  # list of car makes
 
 
 
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
        label = MAKES.index(class_mapping[name])
 
        # fetch image & preprocess
        image = Image.open(path).convert("RGB")
        image = self.preprocess(image)
        return image, label, path



def main(args):
    # save directory
    os.makedirs(args.save_dir, exist_ok=True)
 
    # set device
    assert torch.cuda.is_available(), "CUDA device not available."
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device(f"cuda:{args.device}")
 
    # prepare model
    model, _, preprocess = open_clip.create_model_and_transforms(args.hf_model_card, device=device)
 
    # create dataset and dataloader
    image_paths = sorted(glob.glob(f"{args.image_dir}/**/*.jpg", recursive=True))
    image_dataset = ImageDataset(image_paths, preprocess)
    image_loader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
   
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    feature_dict = {}
    with torch.no_grad():
        # loop through images
        for batch, (image, label, path) in tqdm(enumerate(image_loader), desc="Processing images"):
            image, label = image.to(device), label.to(device)
            image_features = model.encode_image(image).cpu()
            for p,f in zip(path, image_features):
                feature_dict[p] = f
    torch.save(feature_dict, f"{args.save_dir}/feature_dict.pt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument("--image_dir", type=str, default="/workspace/open/train", help="Cropped image directory")
    # parser.add_argument("--save_dir", type=str, default="/home/haiqv/workspace/hvai/ViT-H-14-laion2B-s32B-b79K", help="Directory to save images with label disagreement")
    # parser.add_argument("--hf-model-card", type=str, default="hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K", help="HuggingFace model card for OpenCLIP models")
    parser.add_argument("--save_dir", type=str, default="ViT-SO400M-14-SigLIP2", help="HuggingFace model card for OpenCLIP models")
    parser.add_argument("--hf-model-card", type=str, default="hf-hub:timm/ViT-SO400M-14-SigLIP2", help="HuggingFace model card for OpenCLIP models")
 
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for processing images")
    parser.add_argument("--num_workers", type=int, default=64, help="Number of workers for DataLoader")
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()
 
    main(args)