import pandas as pd
import numpy as np
from skopt import Optimizer
from skopt.space import Real, Integer
from sklearn.utils import shuffle
from copy import deepcopy

def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath, skiprows=1)
    input_cols = ['Bi(%)', 'Fe(%)', 'Co(%)', 'Cu1(%)', 'Ni(%)', 'Mn(%)', 'L1(%)', 'L2(%)']
    target_col = 'K'
    X_all = df[input_cols].values
    y_all = df[target_col].values
    Min,Max = np.min(y_all),np.max(y_all)
    new_Min,new_Max = 0.2,0.6
    y_all = new_Min + (new_Max-new_Min)*(y_all-Min)/(Max-Min)
    X_all = X_all*0.01
    X_all, y_all = shuffle(X_all, y_all)
    return X_all, y_all

def create_optimizer():
    return Optimizer(
        dimensions=[
            Real(0.7, 1.0), Real(0.0, 0.3), Real(0.0, 0.3), Real(0.0, 0.3),
            Real(0.0, 0.3), Real(0.0, 0.3), Integer(0, 1), Integer(0, 1)
        ],
        base_estimator="GP",
        acq_func="EI",
        acq_optimizer="sampling",
        random_state=0
    )

def init_client(X_pool, y_pool, init_num):
    indices = np.random.choice(len(X_pool), init_num, replace=False)
    X_init = X_pool[indices]
    y_init = y_pool[indices]
    mask = np.ones(len(X_pool), dtype=bool)
    mask[indices] = False
    return X_init.tolist(), y_init.tolist(), X_pool[mask], y_pool[mask]


def propose_candidate(optimizer, X_pool):
    next_x = optimizer.ask()
    dists = np.linalg.norm(X_pool - np.array(next_x), axis=1)
    idx = np.argmin(dists)
    return X_pool[idx], idx


def aggregate_candidates(candidates):
    # 选择所有客户端中预测值最高的那个点（基于模型预期）
    #1.每一轮每一个客户都要做不同的实验，每个人提出点都要用上，重复提议的处理。
    #2.用BNN网络进行聚合，得到全局BNN进行最优点推荐（多个），再分别实验。
    best = max(candidates, key=lambda tup: tup[1])
    return best[0]


def main():
    client_num = 3
    init_num = 5
    rounds = 400

    filepath = "/home/user/wyn/Fedbay/data/experimentdata.xlsx"
    X_all, y_all = load_and_preprocess_data(filepath)

    clients = []
    global_X_pool = deepcopy(X_all)
    global_y_pool = deepcopy(y_all)

    for _ in range(client_num):
        X_init, y_init, global_X_pool, global_y_pool = init_client(global_X_pool, global_y_pool, init_num)
        opt = create_optimizer()
        opt.tell(X_init, [-y for y in y_init])
        clients.append({"opt": opt, "X": X_init, "y": y_init})

    target = max(global_y_pool)
    for r in range(rounds):
        candidates = []
        for client in clients:
            x_cand, idx = propose_candidate(client["opt"], global_X_pool)
            # 模拟模型预测值为目标函数的当前最大值（即：先不看真实结果）
            pred_y = -client["opt"].base_estimator_.predict([x_cand])[0]
            candidates.append((x_cand, pred_y, idx))

        # 聚合策略：选择所有客户端中预测值最高的候选点
        best_x = aggregate_candidates(candidates)

        # 查询该点的真实实验值（模拟实验）
        dists = np.linalg.norm(global_X_pool - best_x, axis=1)
        idx = np.argmin(dists)
        true_x = global_X_pool[idx]
        true_y = global_y_pool[idx]

        print(f"Round {r + 1}: Selected x {true_x} with true y = {true_y:.4f}")

        if true_y == target:
            print("Global optimum found!")
            break

        # 所有客户端都使用此真实点进行更新
        for client in clients:
            client["opt"].tell([true_x.tolist()], [-true_y])
            client["X"].append(true_x.tolist())
            client["y"].append(true_y)

        # 从全局池中移除该点
        global_X_pool = np.delete(global_X_pool, idx, axis=0)
        global_y_pool = np.delete(global_y_pool, idx, axis=0)

    print("联邦贝叶斯优化完成！")


if __name__ == "__main__":
    main()
