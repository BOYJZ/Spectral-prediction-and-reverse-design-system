from scripts.generate_data import generate_data
from scripts.forward_model import train_forward_model
from scripts.inverse_model import train_inverse_model
from scripts.optimize_spectrum import optimize_spectrum

if __name__ == '__main__':
    # 生成数据
    generate_data()
    
    # 训练正向预测模型
    train_forward_model()

    # 训练逆向设计模型
    train_inverse_model()

    # 优化光谱并进行模型对比
    optimize_spectrum()
