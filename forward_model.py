import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd

def train_forward_model():
    # 加载数据集
    data = pd.read_csv('../data/fdtd_data.csv')
    X = data[['d3', 'd2', 'd1', 'w', 'p']].values
    y_tm = data['TM'].values
    y_te = data['TE'].values
    y_er = data['ER'].values

    # 构建正向预测模型
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(5,)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(3))  # 输出3个光谱: TM, TE, ER

    # 编译模型
    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    history = model.fit(X, [y_tm, y_te, y_er], epochs=1000, batch_size=32, validation_split=0.2)

    # 保存模型
    model.save('../models/forward_prediction_model.h5')
    print("正向预测模型训练完成并保存到 models/forward_prediction_model.h5")
