import os
import sys
import cv2
import argparse
from pathlib import Path
sys.path.append('/home/haiqv/workspace/hvai') # User-specific path, ensure it's correct
from open.class_mapping import class_mapping
 
 
# global variables
MAKES = sorted(list(set(class_mapping.values())))  # list of car makes
 
 
 
def main(args):
    # create save directory
    save_dir = Path(args.txtpath).parent / 'wrong_preds'
    os.makedirs(save_dir, exist_ok=True)
   
    with open(args.txtpath, 'r') as fp:
        lines = fp.readlines()
    for l in lines:
        path, gt, pred, corr = l.strip().split()
        fn = path.strip().split('/')[-1]
        if corr == "0":
            img = cv2.imread(path)
            cv2.putText(img, f"GT: {gt}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(img, f"Pr: {pred}", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
            cv2.imwrite(save_dir / fn, img)
 
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txtpath", type=str, default="/home/haiqv/workspace/hvai/openclip_playground/runs/train/mlp_augmented/mlp_best_results.txt")
    args = parser.parse_args()
    main(args)