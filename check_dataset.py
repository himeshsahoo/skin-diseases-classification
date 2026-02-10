import os

dataset_path = r"C:\Users\Himes\Desktop\skin disease classification\train"

classes = os.listdir(dataset_path)
print("Classes found:", classes)

for cls in classes:
    cls_path = os.path.join(dataset_path, cls)
    print(cls, "->", len(os.listdir(cls_path)), "images")
