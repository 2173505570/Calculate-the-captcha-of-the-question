# -*- coding = utf-8 -*-
# @Time :2024/11/15 9:16
# @Author hai
# @File : rename.py
# @Software: PyCharm
import os

def rename_files(directory, prefix):
    # 获取目录下的所有文件名
    files = os.listdir(directory)
    # 过滤出图片文件（这里假设是.png格式）
    image_files = [f for f in files if f.endswith('.png')]
    # 按文件名排序，确保顺序一致
    image_files.sort()

    # 遍历图片文件并重命名
    for index, filename in enumerate(image_files):
        # 构建新的文件名
        new_name = f'{prefix}_{index + 1}.png'
        # 完整的旧文件路径和新文件路径
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_name)
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f'Renamed "{filename}" to "{new_name}"')

# 设置你的文件夹路径和前缀
directory_path = r'G:\python-project\训练模型\divide'
file_prefix = 'divide'

# 调用函数进行重命名
rename_files(directory_path, file_prefix)


