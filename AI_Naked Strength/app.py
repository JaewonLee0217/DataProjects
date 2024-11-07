from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# 전역 변수로 모델과 데이터프레임 초기화
model = RandomForestRegressor()
df = pd.DataFrame(columns=['투자액', '수익']) # 빈 df선언

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    investment = data['투자액']
    prediction = model.predict([[investment]])
    return jsonify({'예측 수익': prediction[0]})

@app.route('/train', methods=['POST'])
def train():
    global df, model
    new_data = request.get_json()
    df = df.append(new_data, ignore_index = True)
    X = df[['투자액']]
    y = df['수익']
    model.fit(X,y)
    return jsonify({'message': '모델이 학습되었습니다.'})

if __name__ == '__main__':
    app.run(debug=True)