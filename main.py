# coding:utf-8

import requests
from flask import Flask, request, abort
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import predict

from linebot import (
    LineBotApi, WebhookHandler
)

from linebot.exceptions import (
    InvalidSignatureError
)

from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage,
)

import os

app = Flask(__name__)

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
    if face_img:
        # モデルを使って判定を行う
        print('モデルで判定を行う')
        predict.pred(face_img)
    else :
        line_bot_api.reply_message(
            event.reply_token,
            TextMessage(text='あなたの顔が検出されませんでした。以下の点に注意してもう一度顔画像を送信してみてください。\n\n・明るい場所で撮影された顔画像\n・正面を向いている顔画像'))
        return

# LINEから画像データを取得
def getImageLine(event):
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)

    line_url = 'https://api.line.me/v2/bot/message/' + message_id + '/content/'
    # 画像の取得
    result = requests.get(line_url, headers=header)

    return 

# 顔画像が含まれていれば切り抜いて返す,なければダメって言う
def check_face(event, result):
    image = Image.open(BytesIO(result.content))
    if image is None:
        line_bot_api.reply_message(
            event.reply_token,
            TextMessage(text='画像が取得出来ませんでした。')
        )
        return '必要な情報が足りません'

    # PIL -> openCVへ
    src_img = np.asarray(image)
    print('convert PIL -> cv2')

    # 顔画像を検出する
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    facerect = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=1, minSize=(180,180))
    print(len(facerect))
    if len(facerect)>0:
        for rect in facerect:
            face_img = src_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            print("顔画像見つけました")
            return face_img
    else:
        print('顔画像が見つからなかった')
        return 

if __name__ == "__main__":
    # app.run()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)