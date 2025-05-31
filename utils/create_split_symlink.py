import os
import glob
import numpy as np
from tqdm import tqdm
 
base = '/workspace/open/train'
dest = '/workspace/vit-finetune/data/hai-symlink'
impaths = glob.glob(f"{base}/**/*.jpg")
 
# create random split 8:2
impaths = np.random.permutation(impaths).tolist()
N = len(impaths) // 10
train_set = impaths[:8*N]
valid_set = impaths[8*N:]
 
for name, split in zip(['train', 'val'], [train_set, valid_set]):
    for impath in tqdm(split):
        cls, fn = impath.split('/')[-2:]
        os.makedirs(f'{dest}/{name}/{cls}', exist_ok=True)
        os.symlink(impath, f'{dest}/{name}/{cls}/{fn}')