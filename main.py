# coding:utf-8

import requests
from flask import Flask, request, abort
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

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
    "Authorization": "Bearer" + YOUR_CHANNEL_ACCESS_TOKEN
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
        TextSendMessage(text='あなたの顔画像を送信してください'))

# 画像が来たときの反応？
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    print("handel_message:", event)
    # オウム返し: text=event.message.text
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text='画像を解析します...'))
    
    # 画像データを取得
    getImageLine(event.message.id)
    # 顔画像が含まれているかcheck

# LINEから画像データを取得
def getImageLine(id):
    message_content = line_bot_api.get_message_content(id)
    image = BytesIO(message_content.content)
    print('image:', image)

    line_url = 'https://api.line.me/v2/bot/message/' + id + '/content/'
    
    # 画像の取得
    result = requests.get(line_url, headers=header)
    print('result:', result)

    return result

# 顔画像が含まれていれば切り抜いて返す,なければダメって言う
def check_face():
    return

if __name__ == "__main__":
    # app.run()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)