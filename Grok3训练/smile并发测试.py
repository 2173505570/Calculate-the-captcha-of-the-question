# -*- coding = utf-8 -*-
# @Time :2025/2/25 13:11
# @Author suixing_sir
# @File : smile并发测试.py
# @Software: PyCharm
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# FastAPI 接口地址
API_URL = "http://127.0.0.1:5000/predict"

# 图片目录
IMAGE_DIR = r"G:\python-project\训练模型\images"

# 获取图片文件列表
image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg'))]


# 并发请求函数
def send_request(image_path):
    try:
        # 记录开始时间
        start_time = time.time()

        # 构造请求
        with open(image_path, "rb") as file:
            files = {"file": (os.path.basename(image_path), file, "image/png")}
            response = requests.post(API_URL, files=files)

        # 计算响应时间
        elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒

        # 检查响应状态码
        if response.status_code == 200:
            result = response.json()
            return {
                "image": os.path.basename(image_path),
                "status": "success",
                "response_time_ms": elapsed_time,
                "code": result.get("code"),
                "prediction": result.get("prediction"),
                "result": result.get("result")
            }
        else:
            return {
                "image": os.path.basename(image_path),
                "status": "failure",
                "response_time_ms": elapsed_time,
                "error": response.text
            }
    except Exception as e:
        return {
            "image": os.path.basename(image_path),
            "status": "error",
            "response_time_ms": None,
            "error": str(e)
        }


# 主函数：模拟并发请求
def main():
    # 设置并发线程数
    concurrency_level = 10  # 可以根据需要调整并发数

    # 记录所有任务的结果
    results = []

    # 使用线程池执行并发请求
    with ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [executor.submit(send_request, image_file) for image_file in image_files]

        # 等待所有任务完成并收集结果
        for future in as_completed(futures):
            results.append(future.result())

    # 打印结果
    print("\n=== 并发请求结果 ===")
    for result in results:
        print(f"Image: {result['image']}, Status: {result['status']}, "
              f"Response Time: {result.get('response_time_ms', 'N/A')}ms")
        if result['status'] == "success":
            print(f"  Code: {result['code']}, Prediction: {result['prediction']}, Result: {result['result']}")
        elif result['status'] == "failure":
            print(f"  Error: {result['error']}")
        else:
            print(f"  Exception: {result['error']}")


if __name__ == "__main__":
    main()