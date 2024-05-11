import os
import shutil

# 测试文件夹路径
test_folder = 'dataset/train'

# 输出文件夹路径
class1_folder = 'dataset/train/class1'
class2_folder = 'dataset/train/class2'

# 如果输出文件夹不存在则创建
if not os.path.exists(class1_folder):
    os.makedirs(class1_folder)
if not os.path.exists(class2_folder):
    os.makedirs(class2_folder)

# 遍历测试文件夹中的文件
for filename in os.listdir(test_folder):
    # 获取文件的完整路径
    filepath = os.path.join(test_folder, filename)
    # 如果是文件而不是文件夹
    if os.path.isfile(filepath):
        # 获取文件名的首字母
        first_letter = filename[0].upper()
        # 如果首字母是'H'，则移动到class1文件夹
        if first_letter == 'H':
            shutil.move(filepath, os.path.join(class1_folder, filename))
            print(f"Moved '{filename}' to {class1_folder}")
        # 如果首字母是'N'，则移动到class2文件夹
        elif first_letter == 'N':
            shutil.move(filepath, os.path.join(class2_folder, filename))
            print(f"Moved '{filename}' to {class2_folder}")
        # 如果首字母既不是'H'也不是'N'，则不移动
        else:
            print(f"Ignored '{filename}'")
