import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO



def main(args):
    # save directory
    if not args.save_dir:
        args.save_dir = str(Path(args.weight).parent.parent)  # assuming {}/weights/*.pt
    os.makedirs(args.save_dir, exist_ok=True)

    
    # Load a model
    model = YOLO(args.weight)  # load an official model

    # Get all image paths
    image_paths = sorted(glob.glob(args.source+'/*.jpg'))

    # Batch processing
    batch_size = args.batch_size  # Adjust batch size based on your GPU memory
    results = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i+batch_size]
        
        # Predict on batch
        preds = model(batch_paths)
        
        # Process each prediction in the batch
        for pred in preds:
            result = {
                pred.names[j]: pred.probs.data[j].item() for j in range(len(pred.probs))
            }
            results.append(result)

    pred = pd.DataFrame(results)

    submission = pd.read_csv('/workspace/HVAI/sample_submission.csv', encoding='utf-8-sig')

    # 'ID' 컬럼을 제외한 클래스 컬럼 정렬
    class_columns = submission.columns[1:]
    pred = pred[class_columns]

    submission[class_columns] = pred.values
    submission.to_csv(f'{args.save_dir}/submission.csv', index=False, encoding='utf-8-sig')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weight', type=str)
    parser.add_argument('source', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    # inference
    main(args)