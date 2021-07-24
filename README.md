# Sentiment Classification
-   Author :  **Song Kitae**  (Department of AI Software, Dankook Software High School)
-   Title : sentiment_classification.ipynb
-   Use Module :  **`Tensorflow 2`**,  **`Keras`**....
-   Key Word : 딥러닝, 문장 인식, 감정 분석

#  Abstract
1.  영어로 된 문장 데이터셋 세 개(SST2, Twitter, Reddit)를 합치고, 이것을 학습하여 감정 분석 인공지능을 제작하였다.
2.  문장의 감정 상태를 예측하여 글을 쓴 사람의 감정 상태(긍-부정)을 판단한다.
3.  영어로 된 문장을 판별하는데 90%의 정확도를 얻어낼 수 있었다.

#  데이터셋
-   본 모델은 아래의 데이터들을 토대로 학습합니다. (총 138,785 문장)
    -   Stanford Sentiment Treebank (i.e. SST2) : (6,920 문장)
    -   Twitter : (82,914 문장)
    -   Reddit : (24,107 문장)
    -   여러 사이트를 돌아다니며 모은 문장 : (24,844 문장)

# Network
![](https://github.com/kitae0522/Sentiment-Classification/blob/main/resource/model_preview.png)
1. Embedding
	- Vocab Size : 10,000
	- Vector Size : 64
2. Global Average Pooling1d
3. Dense
	- Unit : 64
	- Activation func : ReLU
4. Dropout (0.33)
5. Dense
	- Unit : 1
	- Activation func : Sigmoid

- Optimizer : Adam
- Loss func : Binary Crossentropy
- Model Epochs : 40
- Model Batch-Size : 512

## Why used <b>`Binary Crossentropy`</b>at loss function?
이 모델은 0~1 사이의 값으로 사용자의 이진(Binary) 감정 상태(긍-부정)를 예측합니다. 0에 가까울수록 부정, 1에 가까울수록 긍정입니다. `Binary Crossentropy` 손실 함수는 모델이 예측한 값과 실제 라벨 값의 오차를 측정합니다.
만약 이 모델이 여러 개의 카테고리로 분류하는 다중 분류 모델이었다면, `MSE`나 다른 손실함수를 사용했겠지만 그렇지 않기 때문에 손실함수 중 확률을 핸들링하는데 적합한 `Binary Crossentropy` 함수를 사용했습니다.


# How to use it?
### [v] - example/use_model_module.py
```python
import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, test_text):
        self.model = tf.keras.models.load_model("../model.h5")
        self.test_text = test_text
        self.word_index = tf.keras.datasets.imdb.get_word_index()
        self._preprocessing()
        
    def _preprocessing(self):
        self.word_index = {k:(v+3) for k,v in self.word_index.items()}
        self.word_index["<PAD>"] = 0
        self.word_index["<START>"] = 1
        self.word_index["<UNK>"] = 2
        self.word_index["<UNUSED>"] = 3

    def _encode_review(self):
        arr = tf.keras.preprocessing.text.text_to_word_sequence(self.test_text)    
        res = []
        for k, v in enumerate(arr):
            try: res.append(self.word_index[v])
            except KeyError: res.append(self.word_index["<UNK>"])
        return np.array(res)
    
    def _padding(self):
        x = [self._encode_review()]
        x = tf.keras.preprocessing.sequence.pad_sequences(
                x,
                value=self.word_index["<PAD>"],
                padding="post",
                maxlen=512
            )
        return x
    
    def predict(self):
        result = self.model.predict(self._padding())
        return ("positive", result[0][0]) if result > 0.5 else ("negative", result[0][0])
```

### [v] - example/module_use.py
```python
from use_model_module import Model

print(Model("""I broke up with my boyfriend last week.""").predict())

# result
# ('negative', 0.08073172)
```

### [v] - example/flask_module_use.py
```python
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
```

```shell
# request
$ curl http://127.0.0.1:5000/predict \
-X POST \
-d '{"text":"당신을 만날 수 있어서 너무 기분이 좋아요."}'

# result
{"res":"positive","score":"0.87890893","translate":"I'm so happy to meet you."}
```
나는 <i>"번역기를 사용하면 영어 문장으로 학습한 모델을 여러 언어로 사용할 수 있지 않을까?"</i> 라는 생각을 하게 되었다.
네이버에서 제공하는 파파고 API를 사용하여 한글로 입력 받은 값을 영어로 번역해, 모델을 사용하는 코드이다.
결과를 보면 잘 예측하는 것을 확인할 수 있다.

# Give me feedback!
- Tel : 010-7447-1509
- E-Mail : kitae040522@gmail.com
- Telegram : @kitae_song
- IG : @kitae_song

인공지능을 배우고 있는 고등학생입니다. 피드백을 남겨주시면 감사하겠습니다.
