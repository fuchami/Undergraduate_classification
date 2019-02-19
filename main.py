# coding:utf-8

import requests, os
from flask import Flask, request, abort
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import gc

from linebot import (
    LineBotApi, WebhookHandler
)

from linebot.exceptions import (
    InvalidSignatureError
)

from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage,
)

app = Flask(__name__)

""" load model """
PRED_MODEL = load_model('./trained_model.h5')
print('load model')

# 環境変数
YOUR_CHANNEL_ACCESS_TOKEN = os.environ["YOUR_CHANNEL_ACCESS_TOKEN"]
YOUR_CHANNEL_SECRET = os.environ["YOUR_CHANNEL_SECRET"]

line_bot_api = LineBotApi(YOUR_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(YOUR_CHANNEL_SECRET)

header = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + YOUR_CHANNEL_ACCESS_TOKEN
}

# LINE APIにアプリがあることを知らせるためのもの
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body:" + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except:
        abort(400)
    
    return 'OK'

# メッセージがきたときの反応
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    print("handel_message:", event)
    # オウム返し: text=event.message.text
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text='あなたの顔画像を送信してください。工学部か法学部かどうかAIが判定します。\n\n※本botはジョークアプリです。判定結果に一切責任も負いません。'))

# 画像が来たときの反応？
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    classes = ["工学部", "法学部"]

    print("handel_message:", event)
    # オウム返し: text=event.message.text
    """
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text='送信された顔画像を解析します...'))
    """
    
    # 画像データを取得
    image = getImageLine(event)
    # 顔画像が含まれているかcheck
    face_img = check_face(event, image)

    if face_img is None :
        line_bot_api.reply_message(
            event.reply_token,
            TextMessage(text='あなたの顔が検出されませんでした。以下の点に注意してもう一度顔画像を送信してみてください。\n\n・明るい場所で撮影された顔画像\n・正面を向いている顔画像\n・1人だけの顔が映っている画像'))
        return
    else :
        # モデルを使って判定を行う
        print('モデルで判定を行う')
        pred_label, score = pred(face_img, PRED_MODEL)
        result_text = 'あなたは' + str(score *100) + 'の確率で' + classes[pred_label] + 'です。'
        print(result_text)
        line_bot_api.reply_message(
            event.reply_token,
            TextMessage(text=result_text))

# openCV -> keras 
def cvt_keras(img):
    # resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = cv2.resize(img, dsize=(224,224))
    img = img.reshape((1,) + img.shape)
    img /= 255
    return img

# message API用
def pred(img, pred_model):
    classes = ['engineering_faculty', 'law_department']

    print('kerasで読めるようにデータを加工')
    img = cvt_keras(img)

    # 予測
    print('予測')
    pred = pred_model.predict(img, batch_size=1)
    score = np.max(pred)
    pred_label = np.argmax(pred)
    print(pred_label)

    # メモリ解放
    del pred
    gc.collect()

    if pred_label == 0
        return 0, score
    else:
        return 1 ,score

# LINEから画像データを取得
def getImageLine(event):
    message_id = event.message.id
    line_url = 'https://api.line.me/v2/bot/message/' + message_id + '/content/'
    # 画像の取得
    result = requests.get(line_url, headers=header)

    return result

# 顔画像が含まれていれば切り抜いて返す,なければダメって言う
def check_face(event, result):
    src_img = Image.open(BytesIO(result.content))
    # PIL -> openCVへ
    src_img = np.asarray(src_img)

    # 顔画像を検出する
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    facerect = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=1, minSize=(180,180))
    print('len(facerect): ', len(facerect))
    if len(facerect)>0:
        for rect in facerect:
            src_img = src_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            print("顔画像見つけました")
            del cascade 
            gc.collect()

            return src_img
    else:
        print('顔画像が見つからなかった')
        return 

if __name__ == "__main__":
    # app.run()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)