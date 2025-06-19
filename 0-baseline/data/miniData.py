import os
import pickle
import random

# Số lượng mẫu muốn lấy
N = 5

src_img_pkl = 'data/crohme/2014/images.pkl'
src_caption_txt = 'data/crohme/2014/caption.txt'
dst_dir = 'data/crohme/debug'
dst_img_pkl = os.path.join(dst_dir, 'images.pkl')
dst_caption_txt = os.path.join(dst_dir, 'caption.txt')

os.makedirs(dst_dir, exist_ok=True)

# Đọc images.pkl
with open(src_img_pkl, 'rb') as f:
    images_dict = pickle.load(f)  # {image_name: image_array}

# Đọc caption.txt thành dict
caption_dict = {}
with open(src_caption_txt, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        img_name, caption = parts[0], parts[1]
        caption_dict[img_name] = caption

# Lấy tập giao tên ảnh có cả ảnh và caption
valid_img_names = list(set(images_dict.keys()) & set(caption_dict.keys()))
sampled_img_names = random.sample(valid_img_names, min(N, len(valid_img_names)))

# Tạo dict và caption mới
new_images_dict = {img_name: images_dict[img_name] for img_name in sampled_img_names}
with open(dst_caption_txt, 'w', encoding='utf-8') as f_out:
    for img_name in sampled_img_names:
        f_out.write(f'{img_name}\t{caption_dict[img_name]}\n')

# Lưu images.pkl mới
with open(dst_img_pkl, 'wb') as f:
    pickle.dump(new_images_dict, f)

print(f'Done! Đã tạo {len(new_images_dict)} ảnh và caption cho debug tại {dst_dir}')