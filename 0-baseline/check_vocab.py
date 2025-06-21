import os
from tamer.datamodule.vocab import vocab

# Đường dẫn đến từ điển
dictionary_path = "data/crohme/dictionary.txt"

# Kiểm tra file tồn tại
if not os.path.exists(dictionary_path):
    print(f"CẢNH BÁO: File từ điển không tồn tại: {dictionary_path}")
else:
    print(f"Đọc từ điển từ: {dictionary_path}")
    
    # Khởi tạo từ điển
    vocab.init(dictionary_path)
    
    # Hiện kích thước từ điển
    print(f"Kích thước từ điển: {len(vocab)}")
    
    # In các token đặc biệt
    print(f"PAD_IDX: {vocab.PAD_IDX}, từ: '{vocab.indices2words([vocab.PAD_IDX])[0]}'")
    print(f"SOS_IDX: {vocab.SOS_IDX}, từ: '{vocab.indices2words([vocab.SOS_IDX])[0]}'")
    print(f"EOS_IDX: {vocab.EOS_IDX}, từ: '{vocab.indices2words([vocab.EOS_IDX])[0]}'")
    
    # Kiểm tra một vài chuyển đổi cơ bản
    test_words = ["a", "b", "c", "+", "-", "="]
    test_indices = vocab.words2indices(test_words)
    print("\nKiểm tra chuyển đổi từ -> indices:")
    print(f"Từ: {test_words}")
    print(f"Indices: {test_indices}")
    
    print("\nKiểm tra chuyển đổi indices -> từ:")
    recovered_words = vocab.indices2words(test_indices)
    print(f"Indices: {test_indices}")
    print(f"Từ: {recovered_words}")
    print(f"Giống nhau: {test_words == recovered_words}")
    
    # Kiểm tra chuyển đổi sang label
    label = vocab.indices2label(test_indices)
    print(f"\nĐộ dài label: {len(label)}")
    print(f"Label: '{label}'")
    
    # Kiểm tra từ phổ biến
    print("\nMột số từ phổ biến:")
    for idx in range(3, min(23, len(vocab))):
        try:
            word = vocab.indices2words([idx])[0]
            print(f"Index {idx}: '{word}'")
        except Exception as e:
            print(f"Index {idx}: <lỗi: {e}>")
    
    # Kiểm tra xem từ điển có các ký hiệu toán học cơ bản không
    math_symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
                   "+", "-", "*", "/", "=", "(", ")", "[", "]", "{", "}", 
                   "^", "_", "\\frac", "\\sqrt"]
    
    print("\nKiểm tra các ký hiệu toán học:")
    for sym in math_symbols:
        try:
            if sym in vocab.word2idx:
                print(f"'{sym}' có trong từ điển, index = {vocab.word2idx[sym]}")
            else:
                print(f"'{sym}' không có trong từ điển")
        except:
            print(f"Lỗi khi kiểm tra ký hiệu '{sym}'") 