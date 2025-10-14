#y值的最大最小不用真实值，改为指定预估值
import pandas as pd
import numpy as np
from skopt import Optimizer
from sklearn.utils import shuffle
from skopt.space import Real, Integer
import os
import json
# ---------- 数据加载与预处理 ----------
def load_and_preprocess(filepath,init_datanum):
    df = pd.read_excel(filepath, skiprows=1)
    input_cols = ['Bi(%)', 'Fe(%)', 'Co(%)', 'Cu1(%)', 'Ni(%)', 'Mn(%)', 'L1(%)']#, 'L2(%)']
    target_col = 'K'

    X_all = df[input_cols].values
    y_all = df[target_col].values

    # 标准化
    y_min, y_max = 0,0.35
    print("y_min:{},y_max:{}".format(y_min,y_max))
    y_all = 0.2 + (0.6 - 0.2) * (y_all - y_min) / (y_max - y_min)
    X_all = X_all * 0.01

    X_all, y_all = shuffle(X_all, y_all)
    #之前好像忘记排除最大值在初始节点的情况了
    target = np.max(y_all[init_datanum:])
    print("Target of this round: ",target)
    return X_all, y_all, target,input_cols,target_col
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

def run_single_bo(X_train, y_train, X_pool, y_pool, target, max_datanum):
    data = {
        "Round":[],
        "X":[],
        "y":[],
    }
    opt = create_optimizer()
    opt.tell(X_train, [-y for y in y_train])
    endstep = -1
    for step in range(max_datanum):
        next_x = opt.ask()
        
        dists = np.linalg.norm(X_pool - np.array(next_x), axis=1)
        idx = np.argmin(dists)
        true_x = X_pool[idx]
        true_y = y_pool[idx]
        data["Round"].append(step+1)
        part_x = np.round(true_x*100,1).tolist()
        data["X"].append(part_x + [100-part_x[-1]])
        data["y"].append(true_y)
        X_train.append(true_x.tolist())
        y_train.append(true_y)
        opt.tell([true_x.tolist()], [-true_y])
        X_pool = np.delete(X_pool, idx, axis=0)
        y_pool = np.delete(y_pool, idx, axis=0)

        if true_y == target:
            endstep = step+1
        if endstep!=-1 and step == endstep + 10:
            break    
    return endstep, len(X_train),data  # 若未找到最优点

def Save_result(num,colname,initdata,ret,Avgend):
    # [trials_num,endstop,totused,data]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. 构建保存目录
    save_dir = os.path.join(current_dir, "SingleResult", f"data{num}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. 数据准备
    X = initdata["X"]
    y = initdata["y"]
    # 检查数据维度匹配
    if X.shape[0] != len(y):
        raise ValueError("X 的行数与 y 的长度不一致！")
    if X.shape[1] != len(colname[0]):
        raise ValueError("X 的列数与 x 数据列名称数目不一致！")
    
    # 4. 构建 DataFrame
    data = pd.DataFrame(X, columns=colname[0])
    data[colname[1]] = y
    data.insert(0, "序号", np.arange(1, len(y) + 1))
    
    # 5. 保存为 Excel
    save_path = os.path.join(save_dir, "init_data.xlsx")
    data.to_excel(save_path, index=False)
    print(f"初始数据已保存至: {save_path}")
    


    if ret is not None:
        y_dict = {}
        max_len = 0
        for onetrial in ret:
            exp_id = onetrial[0]  # 实验编号
            y_vals = onetrial[1]["y"]
            y_dict[str(exp_id)] = y_vals
            max_len = max(max_len, len(y_vals))
        
        # 对齐长度，构建 DataFrame
        aligned_data = {}
        for exp_id, y_vals in y_dict.items():
            padded = list(y_vals) + [np.nan] * (max_len - len(y_vals))
            aligned_data[exp_id] = padded
        
        df_ret_y = pd.DataFrame(aligned_data)
        
        save_path_ret = os.path.join(save_dir, "ret_y.xlsx")
        df_ret_y.to_excel(save_path_ret, index=False)
        print(f"y值的实验过程已保存至: {save_path_ret}")

     # ---------------- 保存 ret.json ----------------
    if ret is not None:
        # 递归转换 numpy 类型 -> Python 内置类型
        def to_builtin(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: to_builtin(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_builtin(v) for v in obj]
            else:
                return obj
        
        ret_builtin = to_builtin(ret)
        
        save_path_ret = os.path.join(save_dir, "ret.json")
        with open(save_path_ret, "w", encoding="utf-8") as f:
            json.dump(ret_builtin, f, ensure_ascii=False, indent=4)
            json.dump({"Average_endstep":Avgend}, f, ensure_ascii=False, indent=4)
        print(f"全部实验结果已保存至: {save_path_ret}")

def main(filepath,num,trials,init_datanum,max_datanum):
    # num是取初始数据的轮数，trials是同一组初始数据重复实验的轮数
    
    X_all, y_all, target, inputcols, targetcol= load_and_preprocess(filepath,init_datanum)
    ret = []
    Avgend = 0
    for j in range(trials):
        X_init = X_all[:init_datanum].tolist()
        y_init = y_all[:init_datanum].tolist()
        X_pool = X_all[init_datanum:]
        y_pool = y_all[init_datanum:]
        print("INIT_Y:",y_init)
        endstep,tot_used,data = run_single_bo(X_init, y_init, X_pool,y_pool, target, max_datanum)
        ret.append([endstep,data])
        Avgend += endstep
        print("num:{},j:{},endstep:{}".format(num,j,endstep))
    #把同样起始点的所有实验结果都存了
    Save_result(num,[inputcols,targetcol],{"X":X_all[:init_datanum],'y':y_all[:init_datanum]},ret,Avgend/trials)
       
        
 
 
if __name__ == '__main__':
    datafile = './data/experimentdata.xlsx'
    totrnd = 3
    trials = 100
    init_datanum = 5
    max_datanum = 250
    for rnd in range(totrnd):
        main(datafile,rnd,trials,init_datanum,max_datanum)
    #main(datafile,1,trials,init_datanum,max_datanum)