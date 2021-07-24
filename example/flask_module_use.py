from flask import Flask, request, jsonify
from use_model_module import Model
import requests as req
import json

def papago_api(url, text):
    with open("papago_api/pw.txt") as f:
        pw = f.readline().split("\n")[0]
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Naver-Client-Id": "hSGgxvmA0A_pFdvv6yyw",
        "X-Naver-Client-Secret": pw
    }

    params = {
        "source": "ko",
        "target": "en",
        "text": text
    }
    
    response = req.post(url, headers=headers, data=params)
    return response.json()["message"]["result"]["translatedText"]

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def main():
    req_text = json.loads(request.get_data())
    conn_papago = papago_api("https://openapi.naver.com/v1/papago/n2mt", req_text["text"])

    if Model(conn_papago).predict() > 0.5:
        return jsonify({
          "score" : str(Model(conn_papago).predict()),
          "res" : "positive",
          "translate" : conn_papago
        })
    else:
        return jsonify({
          "score" : str(Model(conn_papago).predict()),
          "res" : "negative",
          "translate" : conn_papago
        })
        
app.run()
