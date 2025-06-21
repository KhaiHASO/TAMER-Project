import os
import torch
import pickle
from lightning.pytorch import Trainer, seed_everything
from tamer.datamodule import HMEDatamodule, Batch
from tamer.lit_tamer import LitTAMER
from tamer.datamodule.vocab import vocab
from tamer.datamodule.dataset import HMEDataset
from tamer.datamodule.transforms import ScaleToLimitRange
import torchvision.transforms as tr
import random

# Set seed for reproducibility
seed_everything(7)

# Đường dẫn đến checkpoint model
checkpoint_path = "lightning_logs/version_0/checkpoints/"  # Điều chỉnh nếu cần
dictionary_path = "data/crohme/dictionary.txt"
test_folder = "2014"
data_folder = "data/crohme"

# Kiểm tra nếu có checkpoint
if os.path.exists(checkpoint_path) and os.listdir(checkpoint_path):
    checkpoint_file = os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])
    print(f"Sử dụng checkpoint: {checkpoint_file}")
else:
    print(f"Không tìm thấy checkpoint trong {checkpoint_path}")
    checkpoint_file = None
    
# Khởi tạo từ điển
if not os.path.exists(dictionary_path):
    print(f"Không tìm thấy file từ điển: {dictionary_path}")
    exit(1)

vocab.init(dictionary_path)
print(f"Đã khởi tạo từ điển với {len(vocab)} từ")

# Lấy một ảnh từ test set
test_images_path = os.path.join(data_folder, test_folder, "images.pkl")
test_captions_path = os.path.join(data_folder, test_folder, "caption.txt")

if not os.path.exists(test_images_path):
    print(f"Không tìm thấy file test images: {test_images_path}")
    exit(1)

if not os.path.exists(test_captions_path):
    print(f"Không tìm thấy file test captions: {test_captions_path}")
    exit(1)

print(f"Đọc dữ liệu test từ {test_folder}")
with open(test_images_path, "rb") as f:
    images = pickle.load(f)

with open(test_captions_path, "r") as f:
    captions = f.readlines()

print(f"Số lượng ảnh test: {len(images)}")
print(f"Số lượng caption test: {len(captions)}")

# Lựa chọn ngẫu nhiên một ảnh
random_idx = random.randint(0, len(captions) - 1)
caption_line = captions[random_idx].strip().split()
img_name = caption_line[0]
formula = caption_line[1:]

if img_name not in images:
    print(f"Không tìm thấy ảnh {img_name} trong dataset")
    exit(1)

img = images[img_name]

print(f"Đã chọn ảnh: {img_name}")
print(f"Công thức gốc: {' '.join(formula)}")
print(f"Kích thước ảnh: {img.shape}")

# Tiền xử lý ảnh
transform = tr.Compose([
    ScaleToLimitRange(w_lo=16, w_hi=1024, h_lo=16, h_hi=256),
    tr.ToTensor()
])

img_tensor = transform(img).unsqueeze(0)  # [1, 1, H, W]
height, width = img_tensor.shape[2], img_tensor.shape[3]
mask = torch.zeros(1, height, width, dtype=torch.bool)
indices = vocab.words2indices(formula)

# Tạo batch
batch = Batch(
    img_bases=[img_name],
    imgs=img_tensor,
    mask=mask,
    indices=[indices]
)

# Tải model
if checkpoint_file:
    model = LitTAMER.load_from_checkpoint(checkpoint_file)
    model.eval()
    
    print("Thực hiện dự đoán...")
    with torch.no_grad():
        hyps = model.approximate_joint_search(batch.imgs, batch.mask)
        pred_words = vocab.indices2words(hyps[0].seq)
        
    print("\nKết quả dự đoán:")
    print(f"Dự đoán: {' '.join(pred_words)}")
    print(f"Ground truth: {' '.join(formula)}")
    
    print("\nĐánh giá kết quả:")
    print(f"Đúng: {pred_words == formula}")
    if pred_words != formula:
        print(f"Độ dài dự đoán: {len(pred_words)}, Độ dài ground truth: {len(formula)}")
        
        # Đếm từ đúng
        correct_count = sum(1 for p, g in zip(pred_words[:min(len(pred_words), len(formula))], formula[:min(len(pred_words), len(formula))]) if p == g)
        print(f"Số từ dự đoán đúng: {correct_count}/{min(len(pred_words), len(formula))}")
else:
    print("Không thể tiến hành dự đoán do không tìm thấy checkpoint") 