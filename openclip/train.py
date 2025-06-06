import os
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
 
# import MLDecoder
import sys
sys.path.append("/workspace/HVAI") # User-specific path
from ML_Decoder.src_files.ml_decoder.ml_decoder import MLDecoder
from utils.class_mapping import class_mapping
 
 
# global variables
MAKES = sorted(list(set(class_mapping.values()))) # Sorted for consistency
 
 
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
 
class VectorDataset(Dataset):
    def __init__(self, vector_path, make=None):
        self.vector_path = vector_path
        self.make = make
        vector_dict = torch.load(vector_path)
       
        # make classification
        if self.make is None or self.make == 'model':
            self.MODELS = None
            self.NC = len(MAKES) if self.make is None else len(set(class_mapping.keys()))
            self.paths, self.vectors = map(list, (vector_dict.keys(), vector_dict.values()))
        # model classification
        else:
            assert make in MAKES, f"{make} is not a valid make name. Choice: {MAKES}"
            self.MODELS = sorted([k for k,v in class_mapping.items() if v == make])
            self.NC = len(self.MODELS)
            self.paths, self.vectors = [], []
            for path, vector in vector_dict.items():
                model = path.split('/')[-2]
                if class_mapping[model] == make:
                    self.paths.append(path)
                    self.vectors.append(vector)
        self.feature_dim = self.vectors[0].shape[0] # Determine feature dimension from the first vector
 
    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, idx):
        vector, path = self.vectors[idx], self.paths[idx]
        model = path.split('/')[-2] # Assumes path structure like '.../car_make_model/image.jpg'
       
        if self.make is None:  # make classification
            label = MAKES.index(class_mapping[model])
        elif self.make == 'model':  # model classification
            label = list(class_mapping.keys()).index(model)
        else:  # model classification
            label = self.MODELS.index(model)
        return vector, torch.tensor(label, dtype=torch.long), path
 
 
def apply_gaussian_noise(vectors, std, device):
    """Applies Gaussian noise to the input vectors."""
    if std > 0:
        noise = torch.randn_like(vectors, device=device) * std
        return vectors + noise
    return vectors
 
def apply_mixup(vectors, labels, alpha, device):
    """Applies Mixup augmentation."""
    if alpha > 0:
        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)
       
        # Get a shuffled batch for mixing
        batch_size = vectors.size(0)
        index = torch.randperm(batch_size, device=device)
       
        mixed_vectors = lam * vectors + (1 - lam) * vectors[index, :]
        labels_a, labels_b = labels, labels[index]
        return mixed_vectors, labels_a, labels_b, lam
    return vectors, labels, labels, 1.0 # No mixup, lam=1 means use original labels_a
 
 
def main(args):
    # --- Setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
   
    # create save directory
    if len(args.make):
        args.save_dir = args.save_dir + "_" + args.make
    os.makedirs(args.save_dir, exist_ok=True)
    logging.info(f"Run outputs will be saved to: {args.save_dir}")
 
    # fix random seed
    seed_everything(args.seed)
 
    # set device
    device_str = f"cuda:{args.device}" if torch.cuda.is_available() and args.device.isdigit() else "cpu"
    device = torch.device(device_str)
    logging.info(f"Using device: {device}")
 
    # --- Data ---
    logging.info(f"Loading feature vectors from: {args.vector_path}")
   
    try:
        dataset = VectorDataset(args.vector_path, args.make if len(args.make) else None)
        NC = dataset.NC # Number of classes
        MODELS = dataset.MODELS
        logging.info(f"Number of classes: {NC}")
        train_idx, valid_idx = train_test_split(range(len(dataset)), test_size=args.valid_split_ratio)
        trainset = Subset(dataset, train_idx)
        validset = Subset(dataset, valid_idx)
        logging.info(f'#train samples: {len(trainset)}, #valid samples: {len(validset)}')
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return
       
    feature_dim = dataset.feature_dim
    logging.info(f"Dataset loaded: {len(dataset)} samples, feature dimension: {feature_dim}")
 
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
    validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
 
    # --- Model ---
    model = None
    if args.classifier == 'linear':
        model = nn.Linear(feature_dim, NC)
        logging.info("Using Linear classifier.")
    elif args.classifier == 'mlp':
        hidden_dim = feature_dim // 2
        model = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, NC)
        )
        logging.info(f"Using MLP classifier with hidden_dim: {hidden_dim}.")
    elif args.classifier == 'mldecoder':
        if MLDecoder is None:
            logging.error("MLDecoder class not found.")
            return
        model = MLDecoder(num_classes=NC,
                          initial_num_features=feature_dim,
                          decoder_embedding=args.mldecoder_embedding_dim,
                          num_of_groups=args.mldecoder_groups)
        logging.info(f"Using ML-Decoder classifier with embedding_dim={args.mldecoder_embedding_dim}, groups={args.mldecoder_groups}.")
    else:
        logging.error(f"Unsupported classifier type: {args.classifier}")
        return
 
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model '{args.classifier}' initialized with {total_params:,} trainable parameters.")
 
    # --- Training Components ---
    criterion = nn.CrossEntropyLoss() # For Mixup, we apply it manually
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
 
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        logging.info("Using CosineAnnealingLR scheduler.")
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)
        logging.info(f"Using StepLR scheduler with step_size={max(1, args.epochs // 3)}, gamma=0.1.")
 
    # --- Training Loop ---
    logging.info(f"Starting training for {args.epochs} epochs...")
    best_accuracy = 0.0
 
    for epoch in range(args.epochs):
        ######## train loop starts ########
        model.train()
        train_loss_accum = 0.0 # Use a different name to avoid conflict with 'loss' variable
        correct_predictions_train = 0
        total_samples_train = 0
 
        progress_bar_train = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for i, (vectors, labels, paths) in enumerate(progress_bar_train):
            vectors, labels = vectors.to(device), labels.to(device)
 
            # --- Apply Augmentations ---
            # 1. Gaussian Noise (applied first if both are used)
            if args.use_gaussian_noise:
                vectors = apply_gaussian_noise(vectors, args.gaussian_noise_std, device)
 
            # 2. Mixup
            is_mixup_applied = False
            if args.use_mixup and model.training: # Apply mixup only during training
                vectors_mixed, labels_a, labels_b, lam = apply_mixup(vectors, labels, args.mixup_alpha, device)
                if lam < 1.0: # Indicates mixup actually happened (alpha > 0 and some mixing occurred)
                    is_mixup_applied = True
                vectors_to_model = vectors_mixed # Use mixed vectors
            else:
                vectors_to_model = vectors # Use original/noise-augmented vectors
            # --- End Augmentations ---
 
            # Reshape input for ML-Decoder
            if args.classifier == 'mldecoder':
                vectors_to_model = vectors_to_model.unsqueeze(-1).unsqueeze(-1)
           
            optimizer.zero_grad()
            outputs = model(vectors_to_model)
           
            # Calculate loss
            if is_mixup_applied:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else: # No mixup or mixup resulted in lambda = 1 (original sample)
                loss = criterion(outputs, labels)
           
            loss.backward()
            optimizer.step()
 
            train_loss_accum += loss.item() * vectors.size(0) # Use original vector size for accumulation consistency
           
            # Accuracy for training (on original labels if mixup, or just labels)
            # For simplicity, train accuracy with mixup uses the primary label (labels_a or labels)
            # A more nuanced accuracy for mixup isn't standardly reported during training iteration.
            _, predicted = torch.max(outputs.data, 1)
            total_samples_train += labels.size(0)
            # If mixup, accuracy against labels_a might be one way, or skip detailed train acc per batch
            # Here, we use 'labels' which would be labels_a if mixup occurred and lam < 1.
            correct_predictions_train += (predicted == (labels_a if is_mixup_applied else labels)).sum().item()
 
            if i % (len(trainloader) // 5 + 1) == 0: # Log a few times per epoch
                 progress_bar_train.set_postfix(batch_loss=loss.item())
 
 
        epoch_train_loss = train_loss_accum / total_samples_train
        epoch_train_acc = correct_predictions_train / total_samples_train
        progress_bar_train.close()
        ######## train loop ends ########
 
        ######## valid loop starts ########
        model.eval()
        valid_loss_accum = 0.0
        correct_predictions_valid = 0
        total_samples_valid = 0
        valid_results = []
 
        progress_bar_valid = tqdm(validloader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]", leave=False)
        for i, (vectors, labels, paths) in enumerate(progress_bar_valid):
            vectors, labels = vectors.to(device), labels.to(device)
 
            if args.classifier == 'mldecoder':
                vectors = vectors.unsqueeze(-1).unsqueeze(-1)
           
            with torch.no_grad():
                outputs = model(vectors)
                loss = criterion(outputs, labels)
 
            valid_loss_accum += loss.item() * vectors.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples_valid += labels.size(0)
            correct_predictions_valid += (predicted == labels).sum().item()
 
            for pth, lab, prd in zip(paths, labels, predicted):
                if MODELS is None and args.make == '':
                    valid_results.append(f"{pth} {MAKES[lab.item()]} {MAKES[prd.item()]} {int(lab.item()==prd.item())}\n")
                elif MODELS is None and args.make == 'model':
                    valid_results.append(f"{pth} {list(class_mapping.keys())[lab.item()]} {list(class_mapping.keys())[prd.item()]} {int(lab.item()==prd.item())}\n")
                else:
                    valid_results.append(f"{pth} {MODELS[lab.item()]} {MODELS[prd.item()]} {int(lab.item()==prd.item())}\n")
               
       
        epoch_valid_loss = valid_loss_accum / total_samples_valid
        epoch_valid_acc = correct_predictions_valid / total_samples_valid
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar_valid.close()
       
        logging.info(
            f"Epoch {epoch+1}/{args.epochs} - LR: {current_lr:.6f} | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
            f"Valid Loss: {epoch_valid_loss:.4f}, Valid Acc: {epoch_valid_acc:.4f}"
        )
        ######## valid loop ends ########
 
        if scheduler:
            scheduler.step()
 
        if epoch_valid_acc > best_accuracy:
            best_accuracy = epoch_valid_acc
            best_model_path = os.path.join(args.save_dir, f'{args.classifier}_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_valid_loss,
                'accuracy': epoch_valid_acc,
                'results': valid_results,
                'args': args,
                'MAKES': MAKES,
                'MODELS': dataset.MODELS,
            }, best_model_path)
            with open(os.path.join(args.save_dir, f'{args.classifier}_best_results.txt'), "w") as fp:
                fp.writelines(valid_results)
            logging.info(f"Saved new best model to {best_model_path} (Valid Accuracy: {best_accuracy:.4f})")
 
    logging.info("Training finished.")
   
    final_model_path = os.path.join(args.save_dir, f'{args.classifier}_final_epoch{args.epochs}.pth')
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Saved final model state to {final_model_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification head on pre-computed image embeddings.")
   
    # Data and Setup
    parser.add_argument('--vector_path', type=str, default='/workspace/HVAI/openclip/ViT-SO400M-14-SigLIP2/feature_dict_tta.pt', help='Path to the .pt file containing the feature dictionary.')
    parser.add_argument('--save_dir', type=str, default='runs/train/mldecoder', help='Directory to save training runs and model checkpoints.')
    parser.add_argument('--make', type=str, default='', help='If set, the task becomes model classification among the given make.')
    parser.add_argument('--classifier', type=str, default='mldecoder', choices=['linear', 'mlp', 'mldecoder'], help='Type of classification head to use.')
    parser.add_argument('--valid_split_ratio', type=float, default=0.2, help='Ratio of dataset to use for validation.')
 
 
    # Training Hyperparameters
    parser.add_argument('--device', type=str, default='0', help='GPU index to use (e.g., "0", "1") or "cpu".')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.') # Increased epochs for aug
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.') # Potentially smaller BS with aug
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.') # Often lower LR with aug
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine', 'none'], help='Type of learning rate scheduler.')
 
    # Random seed
    parser.add_argument('--seed', type=int, default=42, help='random seed')
 
    # Augmentation Parameters
    parser.add_argument('--use_gaussian_noise', action='store_true', help='Enable Gaussian noise augmentation.')
    parser.add_argument('--gaussian_noise_std', type=float, default=0.01, help='Standard deviation for Gaussian noise.')
    parser.add_argument('--use_mixup', action='store_true', help='Enable Mixup augmentation.')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Alpha parameter for Beta distribution in Mixup (e.g., 0.1 to 0.4).')
   
    # ML-Decoder Specific Parameters (if used)
    parser.add_argument('--mldecoder_embedding_dim', type=int, default=768, help='Internal embedding dimension for ML-Decoder.')
    parser.add_argument('--mldecoder_groups', type=int, default=-1, help='Number of groups for ML-Decoder group-wise linear layers.')
 
 
    args = parser.parse_args()
    main(args)