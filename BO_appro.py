import pandas as pd
import numpy as np
from skopt import Optimizer
from sklearn.utils import shuffle
from skopt.space import Real, Integer

# ---------- 数据加载与预处理 ----------
def load_and_preprocess(filepath):
    df = pd.read_excel(filepath, skiprows=1)
    input_cols = ['Bi(%)', 'Fe(%)', 'Co(%)', 'Cu1(%)', 'Ni(%)', 'Mn(%)', 'L1(%)']#, 'L2(%)']
    target_col = 'K'

    X_all = df[input_cols].values
    y_all = df[target_col].values

    # 标准化
    y_min, y_max = np.min(y_all), np.max(y_all)
    y_all = 0.2 + (0.6 - 0.2) * (y_all - y_min) / (y_max - y_min)
    X_all = X_all * 0.01

    X_all, y_all = shuffle(X_all, y_all)
    return X_all, y_all, np.max(y_all)

# ---------- 初始化优化器 ----------
def create_optimizer():
    return Optimizer(
        dimensions=[
            Real(0.7, 1.0),
            Real(0.0, 0.3), Real(0.0, 0.3), Real(0.0, 0.3),
            Real(0.0, 0.3), Real(0.0, 0.3),
            Integer(0, 1),# Integer(0, 1)
        ],
        base_estimator="GP",
        acq_func="EI",
        acq_optimizer="sampling",
        random_state=None
    )

def Save_data_to_exc(num,data,endstep,initX,initY):
    # 将 initX 和 initY 转换为 DataFrame 并命名列
    initX_col = pd.DataFrame(initX).stack().reset_index(drop=True) 
    initY_col = pd.DataFrame(initY).stack().reset_index(drop=True) 
    endstep_col = pd.DataFrame([endstep], columns=["endstep"])
    # 将 data 字典转换为 DataFrame
    data_df = pd.DataFrame(data)

    # 合并所有数据
    # 确保数据对齐（如果 initX 或 initY 的长度小于其他数据，缺失的值填充为 NaN）
    full_data = pd.concat([initX_col, initY_col, endstep_col, data_df], axis=1)

    file_name = f"Single/data_randomopt{num}.xlsx"

    # 将数据保存到 Excel 文件中
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        full_data.to_excel(writer, sheet_name='Sheet1', index=False)
 
# ---------- 执行一次优化过程 ----------
def run_single_bo(num, X_all, y_all, target, init_num=5, max_rounds=250):
    #init_num += num%10
    X_train = X_all[:init_num].tolist()
    y_train = y_all[:init_num].tolist()
    X_pool = X_all[init_num:]
    y_pool = y_all[init_num:]
    data = {
        "Round":[],
        "X":[],
        "y":[],
    }

    opt = create_optimizer()
    opt.tell(X_train, [-y for y in y_train])
    endstep = -1
    for step in range(max_rounds):
        next_x = opt.ask()
        
        dists = np.linalg.norm(X_pool - np.array(next_x), axis=1)
        idx = np.argmin(dists)
        if step == 0 :
            print("Real",idx,next_x)
        true_x = X_pool[idx]
        true_y = y_pool[idx]
        data["Round"].append(step+1)
        part_x = np.round(true_x*100,1).tolist()
        data["X"].append(part_x + [100-part_x[-1]])
        data["y"].append(true_y)
        X_train.append(true_x.tolist())
        y_train.append(true_y)
        opt.tell([true_x.tolist()], [-true_y])
        print(step,true_x,true_y)
        X_pool = np.delete(X_pool, idx, axis=0)
        y_pool = np.delete(y_pool, idx, axis=0)

        if true_y == target:
            endstep = step+1
        if endstep!=-1 and step == endstep + 10:
            break    
    Save_data_to_exc(num,data,endstep,[X_all[:init_num].tolist()],[y_all[:init_num].tolist()])
    return endstep, len(X_train)  # 若未找到最优点

# ---------- 主函数：重复执行 N 轮 ----------
def main(filepath, trials=30):
    results = []
    for i in range(30,trials+30):
        X_all, y_all, target = load_and_preprocess(filepath)
        rounds, total_used = run_single_bo(i,X_all, y_all, target)
        results.append((rounds, total_used))

    #print(results)

if __name__ == '__main__':
    main('/home/user/wyn/Fedbay/data/experimentdata.xlsx')