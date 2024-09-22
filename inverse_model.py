import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd

def train_inverse_model():
    # 加载数据集
    data = pd.read_csv('../data/fdtd_data.csv')
    X = data[['d3', 'd2', 'd1', 'w', 'p']].values
    y_tm = data['TM'].values
    y_te = data['TE'].values
    y_er = data['ER'].values

    # 构建逆向设计模型
    inverse_model = models.Sequential()
    inverse_model.add(layers.Dense(128, activation='relu', input_shape=(3,)))  # 输入为光谱 (TM, TE, ER)
    inverse_model.add(layers.Dense(256, activation='relu'))
    inverse_model.add(layers.Dense(5))  # 输出为5个几何参数

    # 编译模型
    inverse_model.compile(optimizer='adam', loss='mse')

    # 训练逆向设计网络
    history = inverse_model.fit([y_tm, y_te, y_er], X, epochs=1000, batch_size=32, validation_split=0.2)

    # 保存模型
    inverse_model.save('../models/inverse_design_model.h5')
    print("逆向设计模型训练完成并保存到 models/inverse_design_model.h5")
