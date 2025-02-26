# -*- coding = utf-8 -*-
# @Time :2025/2/25 11:03
# @Author suixing_sir
# @File : 生成测试图片文件csv.py
# @Software: PyCharm
import os
import csv

# 图片目录路径
image_dir = r"G:\python-project\训练模型\images"

# 输出的 CSV 文件路径
output_csv = r"G:\python-project\训练模型\image_paths.csv"

# 获取目录下的所有 PNG 文件
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.png')]

# 将文件路径写入 CSV 文件
with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头
    writer.writerow(['file_path'])
    # 写入文件路径
    for file in image_files:
        writer.writerow([file])

print(f"已生成 CSV 文件：{output_csv}")