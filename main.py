import pickle
from fastapi import FastAPI, File, Request, APIRouter, Response, Form, UploadFile
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import numpy as np
import ast
import pandas as pd
import io
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score, precision_score

import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, LSTM,Bidirectional,Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K 
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from sklearn.model_selection import KFold,StratifiedKFold, train_test_split
from numpy.random import seed
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


app = FastAPI()
router = APIRouter()

# 모델 불러오기
# model = tf.keras.models.load_model('./models/dbtrained_0604_1.h5')
model = tf.keras.models.load_model('./models/0427_model_withoutsiru.h5')

# pickle 파일에서 LabelEncoder 객체 읽어들이기
with open('./models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# pickle 파일에서 Scaler 객체 읽어들이기
with open('./scaler/db_minmax.pkl', 'rb') as f:
    scaler = pickle.load(f)


# 입력값을 받아들일 Pydantic 모델
class Input(BaseModel):
    input_file: bytes


class InputData(BaseModel):
    data: List[List[float]]

# 추가 코드: bytes를 str로 변환하는 함수
def bytes_to_str(bytes_data: bytes) -> str:
    return bytes_data.decode('utf-8')

# 수정된 API 엔드포인트 정의
@app.post("/predict")
async def predict(input_file: bytes = File(...)):
    # 바이너리 파일을 읽어와서 문자열로 변환
    input_data = pickle.loads(input_file)
    print(input_data)
    # 입력값을 모델에 적용하여 예측 결과를 반환하는 로직을 작성하세요.
    
    prediction = model(np.array(input_data).reshape(-1, 50, 6))

    # 결과 반환
    return {"prediction": prediction.numpy().tolist()}


# API 엔드포인트 정의
@app.post("/predict_str")
async def predict(request: Request):
    # 입력값을 받아들이기
    input_data = await request.json()
    # 문자열 리스트를 파이썬 리스트로 변환
    python_list = ast.literal_eval(input_data)

    # 파이썬 리스트를 numpy 배열로 변환
    np_array = np.array(python_list, dtype=float)

    # 입력값을 모델에 적용하여 예측 결과를 반환하는 로직을 작성하세요.
    prediction = model(np_array.reshape(-1, 50, 6))

    # 결과 반환
    return {"prediction": prediction.numpy().tolist()}


# API 엔드포인트 정의
@app.post("/predict_str_2")
async def predict(request: Request, input_data: InputData):
    # 문자열 리스트를 파이썬 리스트로 변환
    input_tensor = np.array(input_data.data, dtype=float)
    prediction = model(input_tensor.reshape(-1, 50, 6))
    # 각 행에서 가장 큰 값을 가지는 열의 인덱스를 찾음
    max_index = np.argmax(prediction, axis=1)
    # 예측한 클래스로 디코딩
    predicted_classes = le.inverse_transform(max_index)
    # 예측한 클래스의 확률값
    prediction_probabilities = np.max(prediction, axis=1)
    # 클래스와 확률 출력
    print(predicted_classes, prediction_probabilities)



    # 결과 반환
    return {"prediction": predicted_classes.tolist(), "probabilities": prediction_probabilities.tolist()}


# # API 엔드포인트 정의
# # scaler랑 label encoder까지 다 쓴거
# @app.post("/predict_str_3")
# async def predict(request: Request, input_data: InputData):
#     # 문자열 리스트를 파이썬 리스트로 변환
#     input_tensor = np.array(input_data.data, dtype=float)
#     input_tensor = input_tensor/32767
#     m = np.mean(input_tensor, axis=0)
#     s = np.std(input_tensor, axis=0)
#     # input_tensor = input_tensor.mask(abs(input_tensor-m) > 3 * s, m, axis=1)
#     input_tensor = np.ma.masked_array(input_tensor, mask=(abs(input_tensor - m) > 3 * s))
#     input_tensor -= np.mean(input_tensor, axis=0)
#     prediction = model(input_tensor.reshape(-1, 50, 6))

#     # 각 행에서 가장 큰 값을 가지는 열의 인덱스를 찾음
#     max_index = np.argmax(prediction, axis=1)
#     # 예측한 클래스로 디코딩
#     predicted_classes = le.inverse_transform(max_index)
#     prediction_probabilities = np.max(prediction, axis=1)
#     # 클래스와 확률 출력
#     print(predicted_classes, prediction_probabilities)
#     # 결과 반환
#     return {"prediction": predicted_classes.tolist(), "probabilities": prediction_probabilities.tolist()}

# API 엔드포인트 정의
# scaler랑 label encoder까지 다 쓴거
@app.post("/predict_str_3")
async def predict(request: Request, input_data: InputData):
    # 문자열 리스트를 파이썬 리스트로 변환
    input_tensor = np.array(input_data.data, dtype=float)
    # scaler 적용했음
    
    # prediction = model(scaler.transform(input_tensor).reshape(-1, 50, 6))

    prediction = model(input_tensor.reshape(-1, 50, 6))
    # 각 행에서 가장 큰 값을 가지는 열의 인덱스를 찾음
    max_index = np.argmax(prediction, axis=1)
    # 예측한 클래스로 디코딩
    predicted_classes = le.inverse_transform(max_index)
    prediction_probabilities = np.max(prediction, axis=1)
    # 클래스와 확률 출력
    print(predicted_classes, prediction_probabilities)

    # 결과 반환
    return {"prediction": predicted_classes.tolist(), "probabilities": prediction_probabilities.tolist()}


@router.post("/train")
async def create_csv(file: bytes = Form(...)):
    # 받은 파일을 DataFrame으로 변환합니다.
    file_object = io.BytesIO(file)
    db_data = pd.read_csv(file_object, index_col=0)
    db_data = db_data[db_data['Label']!='shaking']
    cut = 50 #50Hz
    fft_test_X, rms_test_X, test_y = fft_transformation(db_data, cut)
    fft_test_X = np.array(fft_test_X)
    rms_test_X = np.array(rms_test_X)   
    test_y = np.array(test_y, dtype='object')

    test_y = le.transform(test_y)
    test_y = to_categorical(test_y, num_classes=8)

    # 두부 데이터 나누기
    db_train_X, db_test_X, db_train_y, db_test_y = train_test_split(fft_test_X, test_y, test_size=0.2,shuffle=True, stratify=test_y, random_state=42)
    db_train_X, db_val_X, db_train_y, db_val_y = train_test_split(db_train_X, db_train_y, test_size=0.25,shuffle=True, stratify=db_train_y, random_state=42)

    # 기존에 학습된 모델을 불러옵니다.
    model_pre_trained = keras.models.load_model('/Users/teajun/Desktop/산종설/workspace/0427_두부/all_fft_0428.h5')

    # 기존 모델의 마지막 레이어를 제거합니다.
    new_output = model_pre_trained.layers[-1].output

    # 새로운 레이어를 추가합니다.
    new_output = keras.layers.Dense(8, activation='softmax')(new_output)

    # 새로운 모델을 정의합니다.
    new_model = keras.models.Model(inputs=model_pre_trained.input, outputs=new_output)

    # 기존 모델의 가중치를 새로운 모델에 복사합니다.
    for i in range(len(new_model.layers) - 1):
        new_model.layers[i].set_weights(model_pre_trained.layers[i].get_weights())

    # 새로운 모델을 컴파일합니다.
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    reLR = ReduceLROnPlateau(patience = 4,verbose = 1,factor = 0.5) 
    es =EarlyStopping(monitor='val_loss', patience=10, mode='min')

    # 전이학습이기 때문에 warm up, learning rate 처음부터 크게가면 가중치 망가짐

    # Warmup settings
    warmup_epochs = 5
    warmup_rate = 0.01
    base_rate = 0.001

    # Define optimizer with warmup
    optimizer = Adam(lr=warmup_rate, decay=0.0)
    new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with warmup
    new_model.fit(db_train_X, db_train_y, epochs=warmup_epochs, validation_data=(db_val_X, db_val_y), verbose=1, batch_size=64)
    # plot_history(history_warmup)

    # Change the learning rate and train the model
    new_model.fit(db_train_X, db_train_y, epochs = 100, validation_data= (db_val_X, db_val_y), 
                        verbose=1,batch_size=64,callbacks=[reLR])
    new_model.save('0507.h5')
    # # 변환된 DataFrame을 응답으로 반환합니다.
    return db_data.to_dict()
app.include_router(router)


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Upload CSV file and convert it to JSON.
    """
    df = pd.read_csv(file.file)
    json_data = df.to_json(orient="records")
    return json_data

# 데이터를 window size = 50 단위로 분할하는 함수
def fft_transformation(data, cut):
    sensor_columns=['GyroscopeX', 'GyroscopeY', 'GyroscopeZ', 'AccelerometerX', 'AccelerometerY', 'AccelerometerZ']
    col_len = 6
    fft_X = []
    #preqs = []
    rms_X = []
    y=[]
    sample_rate=50
    stack = 1

    for i in tqdm(range(len(data)-1)):
        seg_now = data.iloc[i]["SegmentID"]
        seg_next = data.iloc[i+1]["SegmentID"]
        if seg_now == seg_next:
            stack += 1
            if stack == cut:

                X_data = data.iloc[i+2-cut:i+2,:col_len]
                y_data = data.iloc[i+2-cut,col_len]

                #fft 수행
                fft_X_data = np.fft.fft(X_data[sensor_columns])
                #freqs = np.fft.fftfreq(len(X_data), 1 / sample_rate)

                # 선택된 주파수 성분에 대해 RMS 계산
                rms_X_data = np.sqrt(np.mean(np.square(np.abs(fft_X_data))))

                #데이터 저장
                fft_X.append(fft_X_data)
                #preqs.append(freqs)
                rms_X.append(rms_X_data)
                y.append(y_data)
                stack = 0
        else:
            stack = 1
    
    
    return fft_X, rms_X, y
