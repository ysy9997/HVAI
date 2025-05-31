import os.path
import timm
import processingtools.torch_tools as tt
import processingtools as pt
import torch
import shutil
import utils

 
if __name__ == '__main__':
    model_path = 'output_0.40/best_model_convnext.pth'

    model = timm.create_model('convnext_base')
    model.head.fc = nn.Linear(model.head.in_features, 396, bias=True)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    model = tt.AutoInputModel(model.cuda().eval(), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=(224, 224))
 
    classes = list(utils.class_mapping.keys())
    for class_name in classes:
        os.makedirs(os.path.join('../dataset/find_wrong', class_name), exist_ok=True)

    images = []
    for class_name in classes:
        images += pt.sorted_glob(os.path.join('../dataset/train', class_name, '*'))
    images = [_.replace('\\', '/') for _ in images]

    results = model(images, batch_size=64, num_workers=4)['results']
 
    for path, feature in pt.ProgressBar(results.items(), total=len(results), detail_func=lambda _: _[0].split('/')[-2]):
        infer_class = int(torch.argmax(feature))
        if classes[infer_class] != path.split('/')[-2]:
            filename = os.path.basename(path)
            name, ext = os.path.splitext(filename)
            new_filename = f'{name}@{classes[infer_class]}{ext}'
            shutil.copy(path, os.path.join('../dataset/find_wrong', path.split('/')[-2], new_filename))


