from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import Data
from skopt import gp_minimize
from skopt.space import Real, Integer
import numpy as np

X,y = Data.Load_expdata()
kernel = RBF(length_scale=1.0)
model = GaussianProcessRegressor(kernel=kernel, random_state=525)
model.fit(X,y)

space = [
    Real(20.0, 100.0, name='chemical_1'),  # 化学物质 1 的百分比
    Real(0.0, 30.0, name='chemical_2'),  # 化学物质 2 的百分比
    Real(0.0, 30.0, name='chemical_3'),  # 化学物质 3 的百分比
    Real(0.0, 30.0, name='chemical_4'),  # 化学物质 4 的百分比
    Real(0.0, 30.0, name='chemical_5'),  # 化学物质 5 的百分比
    Real(0.0, 30.0, name='chemical_6'),  # 化学物质 6 的百分比
    Integer(0, 1, name='L1'),           # L1 取值为 0 或 100
    Integer(0, 1, name='L2')            # L2 取值为 0 或 100
]

def objective_function(params):
    # 提取输入值
    chemicals = np.array(params[:6])  # 六种化学物质的百分比
    L1 = params[6] * 100              # 将 0/1 映射到 0/100
    L2 = params[7] * 100              # 将 0/1 映射到 0/100

    # 归一化六种化学物质的百分比
    chemicals_normalized = chemicals / np.sum(chemicals)

    # 组合输入值
    input_data = np.concatenate([chemicals_normalized, [L1, L2]])

    # 使用高斯过程模型预测输出值
    return -model.predict([input_data])[0]  # 负号是因为优化器默认最小化


if __name__ == '__main__':

    result = gp_minimize(
        objective_function,
        space,
        n_calls=30,          # 最大实验次数
        random_state=525,     # 随机种子
        n_initial_points=15  # 初始随机采样次数
    )

    # 处理最优输入值
    best_input = result.x
    best_chemicals = np.array(best_input[:6])
    best_chemicals_normalized = best_chemicals / np.sum(best_chemicals)
    best_L1 = best_input[6] * 100
    best_L2 = best_input[7] * 100

    # 输出最优结果
    print("最优化学物质配比 (归一化):", best_chemicals_normalized)
    print("最优 L1:", best_L1)
    print("最优 L2:", best_L2)
    print("最优输出值 (预测 k 值):", -result.fun)  # 取负号恢复原始值


    #y_pred = model.predict(X)
    #print(y_pred)
    