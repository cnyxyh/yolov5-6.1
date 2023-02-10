import base64

import cv2
import numpy as np
from flask import Flask, render_template, Response, request

app = Flask(__name__)
@app.route('/predict/', methods=['POST'])
def get_prediction():
    response = request.get_json()
    data_str = response['image']
    print("4444", data_str)
    imgs = base64.b64decode(data_str)  # 解码
    # print("长度", len(imgs))
    # img = cv2.imdecode(np.frombuffer(imgs, np.uint8), cv2.IMREAD_COLOR)  # 二进制数据流转np.ndarray [np.uint8: 8位像素]
    # dict_new = pred_img(img)  # 得到img和坐标结果
    # print('打印结果', dict_new)
    # return dict_new
    return "我是返回"
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
