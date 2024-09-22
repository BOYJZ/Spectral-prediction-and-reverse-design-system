import lumapi
import numpy as np
import pandas as pd

def generate_data():
    # 连接FDTD仿真器
    fdtd = lumapi.FDTD()

    # 设置变量范围
    d3_range = np.linspace(0.1, 0.5, 100)
    d2_range = np.linspace(0.01, 0.2, 100)
    d1_range = np.linspace(0.01, 0.1, 100)
    w_range = np.linspace(0.5, 0.7, 100)
    p_range = np.linspace(0.4, 0.8, 100)

    # 存储数据
    data = []

    # 保证d2 > d1的限制条件
    for d3 in d3_range:
        for d2 in d2_range:
            for d1 in d1_range:
                if d2 > d1:
                    for w in w_range:
                        for p in p_range:
                            # 设置FDTD中的变量
                            fdtd.putvar("d3", d3)
                            fdtd.putvar("d2", d2)
                            fdtd.putvar("d1", d1)
                            fdtd.putvar("w", w)
                            fdtd.putvar("p", p)

                            # 运行FDTD仿真
                            fdtd.run()

                            # 提取输出光谱TM, TE, ER
                            tm_spectrum = fdtd.getresult("monitor", "TM")
                            te_spectrum = fdtd.getresult("monitor", "TE")
                            er_spectrum = fdtd.getresult("monitor", "ER")

                            # 保存结果
                            data.append([d3, d2, d1, w, p, tm_spectrum, te_spectrum, er_spectrum])

    # 将数据保存为文件
    df = pd.DataFrame(data, columns=['d3', 'd2', 'd1', 'w', 'p', 'TM', 'TE', 'ER'])
    df.to_csv('../data/fdtd_data.csv', index=False)
    print("数据生成完成并保存到 data/fdtd_data.csv")
