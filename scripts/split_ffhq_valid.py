import os
import json
import shutil
from tqdm import tqdm


with open('ffhq-dataset-v2.json', 'r') as f:
    data = json.load(f)

train_ids = [int(k) for k, v in data.items() if v['category'] == 'training']
valid_ids = [int(k) for k, v in data.items() if v['category'] == 'validation']

src_dir = 'data/ffhq'
train_dir = os.path.join(src_dir, 'train')
valid_dir = os.path.join(src_dir, 'valid')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

for img_file in tqdm(os.listdir(src_dir)):
    if not img_file.endswith('.png'):
        continue
    img_id = int(os.path.splitext(img_file)[0])
    src_path = os.path.join(src_dir, img_file)
    if img_id in train_ids:
        dst_path = os.path.join(train_dir, img_file)
    elif img_id in valid_ids:
        dst_path = os.path.join(valid_dir, img_file)
    else:
        continue
    shutil.copy2(src_path, dst_path)
