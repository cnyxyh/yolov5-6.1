# from models.experimental import attempt_load
# from utils.torch_utils import select_device
import cv2
from PIL import Image
import base64
import io
from flask import Flask, request, jsonify
import json
import numpy as np
# from backend.predict import predict
from pathlib import Path

# 传入__name__实例化Flask
app = Flask(__name__)

# 读取flask配置
# with open('./backend/flask_config.json','r',encoding='utf8')as fp:
#     opt = json.load(fp)
#     print('Flask Config : ', opt)

# 选择设备
# device = select_device(opt['device'])
# 加载模型
# model = attempt_load(opt['weights'], map_location=device)

@app.route('/predict/', methods=['POST'])
# 响应POST消息的预测函数




def get_prediction():
    # print("333")
    response = request.get_json()
    # print("333", response)
    data_str = response['image']
    print("4444", data_str)
    imgs = base64.b64decode(data_str)  # 解码
    print("长度",len(imgs))
    img = cv2.imdecode(np.frombuffer(imgs, np.uint8), cv2.IMREAD_COLOR)  # 二进制数据流转np.ndarray [np.uint8: 8位像素]



    # point = data_str.find(',')
    # print("555", point)

    # base64_str = data_str[point:]  # remove unused part like this: "data:image/jpeg;base64,"
    # image = base64.b64decode(base64_str) # base64图像解码
    # img = Image.open(io.BytesIO(image)) # 打开文件
    # if (img.mode != 'RGB'):
    #     img = img.convert("RGB")
    # save_path = str(Path(opt['source']) / Path("img4predict.jpg")) # 保存路径
    # img.save(save_path) # 保存文件
    # # img.save("./frontend/static/images/img4predict.jpg")
    #
    # # convert to numpy array.
    # img_arr = np.array(img)
    # # print('img_arr shape = %s \n' % str(img_arr.shape))
    #
    # # results = predict(opt, model, img_arr) # 预测图像

    # # return jsonify(results)
    # return jsonify("results")
    return "我是返回"

@app.after_request
def add_headers(response):
    # 允许跨域
    response.headers.add('Access-Control-Allow-Origin', '*') 
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    # print("1111", response)
    # print("222",  type(response))

    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    #app.run(debug=False, host='127.0.0.1')



