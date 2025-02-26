from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from sympy import sympify

# 设置多线程
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# 加载 TFLite 模型
interpreter = tf.lite.Interpreter(model_path='captcha_char_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# 推理函数
def predict_tflite(segments):
    predictions = []
    for segment in segments:
        # 将单个样本扩展为 (1, 28, 28, 1) 的形状
        segment = np.expand_dims(segment, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], segment)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output)
    return np.array(predictions)


# 优化预处理
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (28, 28))
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return img.astype(np.float32) / 255.0


# 切分和识别
def split_and_recognize(image: Image.Image):
    # 固定切分点
    split_points = [0, 31, 49, 72]

    width, height = image.size
    # 切分图像
    regions = [image.crop((split_points[i], 0, split_points[i + 1], height))
               for i in range(len(split_points) - 1)]
    # 多线程预处理
    with ThreadPoolExecutor() as executor:
        segments = list(executor.map(preprocess_image, regions))
    # 组合成批量输入
    segments = np.array(segments).reshape(-1, 28, 28, 1)
    predictions = predict_tflite(segments)
    # 字符映射
    char_map = {i: str(i) for i in range(10)}
    char_map.update({10: '+', 11: '-', 12: '*', 13: '/'})
    return [char_map[np.argmax(pred)] for pred in predictions]


# 构建表达式
def build_expression(recognized_chars):
    return "".join(recognized_chars)


# 创建 FastAPI 应用
app = FastAPI()


@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    try:
        # 读取上传的图像文件
        image = Image.open(file.file)

        # 调用识别函数
        recognized_chars = split_and_recognize(image)
        expression = build_expression(recognized_chars)

        # 计算表达式结果
        try:
            result = sympify(expression).evalf()
        except Exception as e:
            result = f"Error evaluating the expression: {e}"

        # 返回结果
        return JSONResponse({
            "recognized_chars": recognized_chars,
            "expression": expression,
            "result": str(result)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)