# Hướng dẫn debug valExpRate luôn bằng 0

Tài liệu này giải thích các sửa đổi được thực hiện để debug và khắc phục vấn đề valExpRate luôn bằng 0 trong quá trình đào tạo.

## 1. Các vấn đề đã xác định

1. **Xác nhận (Validation) bị vô hiệu hóa**:
   - Code validation trong `lit_tamer.py` có một đoạn comment có thể khiến việc tính ExpRate bị bỏ qua nếu epoch hiện tại nhỏ hơn mốc milestone đầu tiên
   - Do milestones được đặt ở [300, 350], nên validation có thể không hoạt động cho đến epoch thứ 300

2. **Learning rate quá cao**:
   - Learning rate được đặt là 1.0 cho Adadelta, có thể gây bất ổn định

3. **Vấn đề trong ExpRateRecorder**:
   - Không có đủ logging để kiểm tra quá trình tính ExpRate
   - Có thể có lỗi khi so sánh chuỗi dự đoán và ground truth

4. **Beam search không ổn định**:
   - Thiếu logging để hiểu quá trình beam search
   - Có thể model không tạo ra đầu ra có ý nghĩa cho beam search

## 2. Các sửa đổi đã thực hiện

### 2.1. Bật Validation

Trong file `tamer/lit_tamer.py`, chúng tôi đã bỏ comment phần mã ngăn việc đánh giá ExpRate:

```python
# if self.current_epoch < self.hparams.milestones[0]:
#     self.log(
#         "val_ExpRate",
#         self.exprate_recorder,
#         prog_bar=True,
#         on_step=False,
#         on_epoch=True,
#     )
#     return
```

### 2.2. Thêm Logging cho ExpRateRecorder

Trong file `tamer/utils/utils.py`, chúng tôi đã thêm các dòng debug để xem quá trình tính ExpRate:

```python
def update(self, indices_hat: List[List[int]], indices: List[List[int]]):
    print("DEBUG: ExpRateRecorder.update được gọi")
    print(f"DEBUG: Số dự đoán: {len(indices_hat)}, Số ground truth: {len(indices)}")
    
    for idx, (pred, truth) in enumerate(zip(indices_hat, indices)):
        pred_label = vocab.indices2label(pred)
        truth_label = vocab.indices2label(truth)
        
        print(f"DEBUG: Sample {idx}:")
        print(f"DEBUG: Pred label: '{pred_label}'")
        print(f"DEBUG: Truth label: '{truth_label}'")

        is_same = pred_label == truth_label
        print(f"DEBUG: Match: {is_same}")

        if is_same:
            self.rec += 1
            print(f"DEBUG: Matches so far: {self.rec}")
```

### 2.3. Giảm Learning Rate và Điều chỉnh Milestones

Trong file `config/crohme.yaml`, chúng tôi đã thay đổi:

```yaml
# training
learning_rate: 0.1  # Giảm từ 1.0 xuống 0.1
patience: 20
milestones:  # Điều chỉnh milestones để debug nhanh hơn
  - 10
  - 20
```

### 2.4. Thêm Debugging cho Beam Search

Trong file `tamer/utils/generation_utils.py`, chúng tôi đã thêm nhiều debug prints để theo dõi quá trình beam search:

```python
print("DEBUG: Starting beam search with beam_size:", beam_size, "max_len:", max_len)
# ... và nhiều log khác
```

### 2.5. Debug Đầu ra của Mô hình trong Training

Trong file `tamer/lit_tamer.py`, chúng tôi đã thêm mã để kiểm tra đầu ra của mô hình trong training:

```python
if self.global_step % 10 == 0:
    print("DEBUG: Training step output check")
    # ... kiểm tra NaN/Inf và phân phối đầu ra
```

## 3. Scripts Mới để Hỗ trợ Debug

### 3.1. Script Đào tạo với Overfit Batches

File `overfit_debug.py` cho phép đào tạo mô hình trên một batch duy nhất để kiểm tra khả năng học:

```python
trainer = Trainer(
    callbacks=callbacks,
    logger=pl.loggers.CSVLogger("lightning_logs", name="overfit_debug"),
    overfit_batches=1,
    max_epochs=30,
    # ...
)
```

### 3.2. Script Kiểm tra Vocabulary

File `check_vocab.py` để kiểm tra vocabulary và quá trình chuyển đổi từ/indices:

```python
vocab.init(dictionary_path)
print(f"Kích thước từ điển: {len(vocab)}")
# ... kiểm tra các chuyển đổi cơ bản
```

### 3.3. Script Dự đoán Đơn lẻ

File `single_prediction.py` để thực hiện dự đoán với một ảnh duy nhất từ tập test và so sánh với ground truth:

```python
model = LitTAMER.load_from_checkpoint(checkpoint_file)
model.eval()

print("Thực hiện dự đoán...")
with torch.no_grad():
    hyps = model.approximate_joint_search(batch.imgs, batch.mask)
    pred_words = vocab.indices2words(hyps[0].seq)
```

## 4. Hướng dẫn Sử dụng

### 4.1. Kiểm tra Vocabulary

Chạy script để kiểm tra từ điển:

```bash
python check_vocab.py
```

### 4.2. Đào tạo với Overfit Batches

Để debug khả năng học của mô hình trên một batch duy nhất:

```bash
python overfit_debug.py
```

### 4.3. Kiểm tra Dự đoán Đơn lẻ

Sau khi đã có checkpoint, chạy:

```bash
python single_prediction.py
```

## 5. Các Điểm Cần Chú ý

1. **Kiểm tra đầu ra beam search**: Log sẽ giúp bạn hiểu mô hình đang tạo ra chuỗi nào và tại sao ExpRate có thể bằng 0.

2. **Kiểm tra sự hội tụ của model**: Với overfit_batches=1, mô hình nên có thể học được ít nhất một mẫu. Nếu không, có thể có vấn đề nghiêm trọng với kiến trúc.

3. **Xem xét phân phối đầu ra**: Log trong training_step sẽ cho bạn biết mô hình đang dự đoán những token nào với xác suất cao nhất.

4. **Kiểm tra ExpRateRecorder**: Log sẽ hiển thị sự khác biệt giữa dự đoán và ground truth, giúp bạn hiểu tại sao ExpRate bằng 0.

## 6. Hướng đi tiếp theo

Nếu vẫn gặp vấn đề sau khi áp dụng các sửa đổi này, bạn có thể:

1. **Đơn giản hóa mô hình**: Có thể thử với ít lớp decoder hơn hoặc cấu trúc đơn giản hơn
2. **Thay đổi thuật toán tối ưu hóa**: Thử chuyển từ Adadelta sang Adam hoặc AdamW
3. **Xem xét vấn đề dữ liệu**: Kiểm tra chất lượng và định dạng của dữ liệu đào tạo
4. **Xem xét quá trình tiền xử lý ảnh**: Đảm bảo ảnh được xử lý đúng cách trước khi đưa vào mô hình 