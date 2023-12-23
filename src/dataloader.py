from PIL import Image
import os

class MyDataLoader:
    def __init__(self, image_root):
        self.image_root=image_root
        self.image_list=[]
        file_names = os.listdir(self.image_root)
        for file_name in file_names:
            # Kiểm tra nếu là tệp ảnh
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                full_path = os.path.join(self.image_root, file_name)
                self.image_list.append(full_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        _img = self.image_list[index]
        return _img