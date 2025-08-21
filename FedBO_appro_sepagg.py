import pandas as pd
import numpy as np
from skopt import Optimizer
from skopt.space import Real, Integer
from sklearn.utils import shuffle
from copy import deepcopy
import itertools
import random
def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath, skiprows=1)
    input_cols = ['Bi(%)', 'Fe(%)', 'Co(%)', 'Cu1(%)', 'Ni(%)', 'Mn(%)', 'L1(%)']#, 'L2(%)']
    target_col = 'K'
    X_all = df[input_cols].values
    y_all = df[target_col].values
    Min,Max = np.min(y_all),np.max(y_all)
    new_Min,new_Max = 0.2,0.6
    y_all = new_Min + (new_Max-new_Min)*(y_all-Min)/(Max-Min)
    X_all = X_all*0.01
    X_all, y_all = shuffle(X_all, y_all)
    X_all = np.round(X_all,3)
    y_all = np.round(y_all,6)
    return X_all, y_all

def create_optimizer():
    return Optimizer(
        dimensions=[
            Real(0.7, 1.0), Real(0.0, 0.3), Real(0.0, 0.3), Real(0.0, 0.3),
            Real(0.0, 0.3), Real(0.0, 0.3), Integer(0, 1)#, Integer(0, 1)
        ],
        base_estimator="GP",
        acq_func="EI",
        acq_optimizer="sampling",
        random_state=None
    )

def init_client(X_pool, y_pool, init_num):
    indices = np.random.choice(len(X_pool), init_num, replace=False)
    X_init = X_pool[indices]
    y_init = y_pool[indices]
    mask = np.ones(len(X_pool), dtype=bool)
    mask[indices] = False
    return X_init.tolist(), y_init.tolist(), X_pool[mask], y_pool[mask],indices


def propose_candidate(optimizer, X_pool):
    next_x = optimizer.ask()
    dists = np.linalg.norm(X_pool - np.array(next_x), axis=1)
    idx = np.argmin(dists)
    return X_pool[idx], idx

def propose_multiple_candidates(optimizer, X_pool, n_points):
    next_xs = optimizer.ask(n_points)
    selected = []
    for next_x in next_xs:
        dists = np.linalg.norm(X_pool - np.array(next_x), axis=1)
        idx = np.argmin(dists)
        selected.append((X_pool[idx], idx))
    return selected  # list of (x, idx)

def sample_unique_assignment(client_num, self_prob=0.6):
    P = np.full((client_num, client_num), (1 - self_prob) / (client_num - 1))
    np.fill_diagonal(P, self_prob)
    perms = list(itertools.permutations(range(client_num)))
    weights = []
    for perm in perms:
        prob = 1.0
        for i in range(client_num):
            prob *= P[i][perm[i]]
        weights.append(prob)
    weights = np.array(weights)
    weights /= weights.sum()
    selected_perm = perms[np.random.choice(len(perms), p=weights)]
    return list(selected_perm)

def Single_per_client(clients,global_X_pool):
    candidates = []
    for client in clients:
        x_cand, idx = propose_candidate(client["opt"], global_X_pool)
        # 模拟模型预测值为目标函数的当前最大值（先不看真实结果）
        pred_y = -client["opt"].base_estimator_.predict([x_cand])[0]
        candidates.append((x_cand, pred_y, idx))
    return candidates

def Multi_per_client(clients,global_X_pool,num):
    candidates = [] 
    for client in clients:
        cand_list = propose_multiple_candidates(client["opt"], global_X_pool,num)
        client_cands = []
        for x_cand, idx in cand_list:
            pred_y = -client["opt"].base_estimator_.predict([x_cand])[0]
            client_cands.append((x_cand, pred_y, idx))
        candidates.append(client_cands)
    return candidates

def aggregate_candidates11(candidates, self_prob=0.5):
    """
    聚合策略：以0.5概率取自己推荐的，重复输入点就各自进行实验（会出现多组重复实验）。
    20组测试：
    轮数：53, 27, 47, 31, 18, 16, 49, 7, 51, 6, 53, 41, 57, 6, 51, 17, 6, 57, 39, 35
    实际使用的实验数据量：156, 81, 141, 92, 53, 47, 144, 21, 152, 18, 159, 119, 170, 18, 151, 51, 18, 169, 116, 104
    平均： 30.9 ,99.4
    """
    client_num = len(candidates)
    assignments = sample_unique_assignment(client_num, self_prob)
    assigned_candidates = [candidates[i] for i in assignments]
    return assignments, assigned_candidates

def aggregate_candidates12(candidates,client_data_counts):
    """
    聚合策略：重复输入点只保留一个，分配给历史数据较少的客户端。轮数会多一点，模拟速度快
    53, 55, 53, 58, 22, 25, 67, 69, 40, 55, 21, 55, 55, 57, 28, 74, 69, 28, 46, 75, 34
    141, 142, 140, 157, 49, 58, 184, 188, 103, 147, 46, 144, 148, 153, 67, 204, 187, 66, 121, 207, 85
    平均： 轮次 49.48, 使用的节点 130.33
    """
    client_num = len(candidates)
    assignments = [-1] * client_num
    assigned_candidates = [None] * client_num

    point_to_clients = {}
    for i, (x, _, _) in enumerate(candidates):
        key = tuple(np.round(x, 6))  # 消除浮点误差
        point_to_clients.setdefault(key, []).append(i)

    assigned = set()

    for key, client_indices in point_to_clients.items():
        # 找出推荐该点的客户端中，数据量最少的那个,这里不考虑用别人的推荐点
        chosen_client = min(client_indices, key=lambda idx: client_data_counts[idx])
        assignments[chosen_client] = chosen_client
        assigned_candidates[chosen_client] = candidates[chosen_client]
        assigned.add(chosen_client)

    return assignments, assigned_candidates

def aggregate_candidates13(candidates,client_data_counts):
    """
    聚合策略：重复输入点只保留一个，分配给历史数据较多的客户端。
    37, 12, 29, 23, 29, 59, 50, 30, 38, 60, 16, 38, 29, 62, 43, 28, 53, 65, 44, 83
    106, 31, 82, 64, 82, 170, 144, 84, 108, 175, 43, 109, 82, 179, 122, 78, 152, 186, 124, 239
    平均： 47.6, 112.3
    """
    client_num = len(candidates)
    assignments = [-1] * client_num
    assigned_candidates = [None] * client_num

    point_to_clients = {}
    for i, (x, _, _) in enumerate(candidates):
        key = tuple(np.round(x, 6))  # 消除浮点误差
        point_to_clients.setdefault(key, []).append(i)

    assigned = set()

    for key, client_indices in point_to_clients.items():
        # 找出推荐该点的客户端中，数据量最多的那个
        chosen_client = max(client_indices, key=lambda idx: client_data_counts[idx])
        assignments[chosen_client] = chosen_client
        assigned_candidates[chosen_client] = candidates[chosen_client]
        assigned.add(chosen_client)

    return assignments, assigned_candidates

def aggregate_candidates14(candidates, client_data_counts): 
    """
    聚合策略:效果最好的优先给所有客户中已有节点数量最少的
    52, 55, 21, 46, 42, 44, 95, 46, 43, 33, 21, 33, 61, 40, 33, 47, 51, 42, 62, 28
    137, 147, 46, 116, 108, 115, 261, 120, 110, 80, 46, 81, 165, 102, 82, 123, 134, 107, 167, 66
    平均：46.3, 119.3
    """
    client_num = len(candidates)
    assignments = [-1] * client_num
    assigned_candidates = [None] * client_num

    chosen_clients = [idx for idx,val in sorted(enumerate(client_data_counts),key=lambda x:x[1])]
    sorted_cand = sorted(candidates, key=lambda x: x[1], reverse=True)
    
    seen = set()
    deduplicated_candidates = []
    
    for candidate in sorted_cand:
        first_value = np.round(candidate[0], 6)  # 将第一元近似到6位小数
        if tuple(first_value.flatten()) not in seen:  # 使用flatten后转为tuple以便存储
            seen.add(tuple(first_value.flatten()))
            deduplicated_candidates.append(candidate)
    
    for idx,cand in zip(chosen_clients,deduplicated_candidates):
        assignments[idx] = 1
        assigned_candidates[idx] = cand
    return assignments,assigned_candidates

def aggregate_candidates15(candidates, client_data_counts):
    """
    聚合策略:效果最好的优先给所有客户中已有节点数量最少的
    53, 72, 44, 22, 40, 53, 48, 8, 32, 58, 49, 41, 44, 57, 28, 45, 16, 10, 69, 28
    152, 206, 125, 60, 115, 152, 136, 22, 91, 169, 142, 118, 125, 165, 79, 130, 43, 26, 201, 77
    平均：42.6, 113.3
    """
    client_num = len(candidates)
    assignments = [-1] * client_num
    assigned_candidates = [None] * client_num

    chosen_clients = [idx for idx,val in sorted(enumerate(client_data_counts),key=lambda x:x[1],reverse = True)]
    sorted_cand = sorted(candidates, key=lambda x: x[1], reverse=True)
    
    seen = set()
    deduplicated_candidates = []
    
    for candidate in sorted_cand:
        first_value = np.round(candidate[0], 6)  # 将第一元近似到6位小数
        if tuple(first_value.flatten()) not in seen:  # 使用flatten后转为tuple以便存储
            seen.add(tuple(first_value.flatten()))
            deduplicated_candidates.append(candidate)
    
    for idx,cand in zip(chosen_clients,deduplicated_candidates):
        assignments[idx] = 1
        assigned_candidates[idx] = cand
    return assignments,assigned_candidates

def aggregate_candidates21(all_candidates):
    """
    聚合策略：每个客户端提供多个候选点，从中选出每个客户端一个唯一的输入点。
    40, 11, 1, 21, 32, 8, 30, 23, 16, 33, 1, 50, 11, 35, 40, 19, 40, 1, 1, 46
    130, 43, 13, 73, 106, 34, 100, 79, 58, 109, 13, 160, 43, 115, 130, 67, 130, 13, 13, 148
    
    平均：22.8，79.8有三组很极端
    """
    client_num = len(all_candidates)
    assignments = [-1] * client_num
    assigned_candidates = [None] * client_num

    # 扁平化所有候选点，附带(client_id, local_idx)
    flat_candidates = []
    for client_id, candidates in enumerate(all_candidates):
        for local_idx, cand in enumerate(candidates):
            x, pred_y, pool_idx = cand
            key = tuple(np.round(x, 6))  # 对齐浮点误差
            flat_candidates.append({
                "x": x, "pred_y": pred_y, "pool_idx": pool_idx,
                "client_id": client_id, "local_idx": local_idx,
                "key": key
            })

    # 按预测值从高到低排序（优先分配潜在效果好的点）
    flat_candidates.sort(key=lambda d: -d["pred_y"])

    used_keys = set()
    assigned_clients = set()

    for cand in flat_candidates:
        key = cand["key"]
        client_id = cand["client_id"]
        local_idx = cand["local_idx"]

        if key in used_keys or client_id in assigned_clients:
            continue  # 跳过已用点或已分配的客户端

        # 分配这个点
        assignments[client_id] = local_idx
        assigned_candidates[client_id] = all_candidates[client_id][local_idx]
        used_keys.add(key)
        assigned_clients.add(client_id)

        if len(assigned_clients) == client_num:
            break

    return assignments, assigned_candidates

def aggregate_candidates22(all_candidates, self_prob=0.6):
    """
    聚合策略3：每个客户端推荐多个候选点（数量等于客户端数量）。
    每个客户端以self_prob概率优先选择自己推荐的点（不重复），否则考虑其他未被选中的候选点。(也可能选到自己的点)
    35, 51, 30, 14, 17, 15, 8, 15, 7, 54, 26, 62, 27, 37, 31, 21, 65, 42, 30, 34
    111, 159, 96, 46, 55, 51, 27, 48, 28, 167, 81, 190, 85, 116, 101, 68, 202, 132, 95, 102
    平均： 31.05,98
    """
    client_num = len(all_candidates)
    assignments = [-1] * client_num
    assigned_candidates = [None] * client_num

    used_keys = set()

    ''' # 顺序打乱客户端，避免固定偏好
    client_order = list(range(client_num))
    random.shuffle(client_order)
    '''
    for client_id in range(client_num):
        candidates = all_candidates[client_id]
        selected = None

        # 尝试按self_prob概率选择自己推荐的点
        if random.random() < self_prob:
            for i, (x, _, _) in enumerate(candidates):
                key = tuple(np.round(x, 6))
                if key not in used_keys:
                    selected = (client_id, key)
                    break

        # 如果自己推荐的点都被占了，或者随机失败，尝试别人的推荐点
        if selected is None:
            print("None")
            # 扁平化所有未使用的候选点，附带来源信息
            flat = []
            for other_id, cand_list in enumerate(all_candidates):
                for j, (x, pred_y, pool_idx) in enumerate(cand_list):
                    key = tuple(np.round(x, 6))
                    if key not in used_keys:
                        flat.append((pred_y, other_id, j, key))
            # 按预测值从高到低排序
            flat.sort(reverse=True, key=lambda x: x[0])
            for _, other_id, j, key in flat:
                
                if key not in used_keys:
                    print("KEY??",other_id,j,key)
                    selected = (other_id, key)
                    break

        # 确认选择并记录
        if selected and selected[0] != -1:
           #print("Why wrong",client_id,selected)
            j, key = selected
            assignments[client_id] = j
            assigned_candidates[client_id] = all_candidates[client_id][j]
            used_keys.add(key)

    return assignments, assigned_candidates


def Save_data_to_exc(data, Rnd, endstep, initX, initY):

    # 主记录表（每轮实验记录）
    df_main = pd.DataFrame(data)
    df_main["endstep"] = [endstep] + [np.nan] * (len(df_main) - 1)

    # 整合 initX 和 initY 为同一个表
    rows = len(initX[0])  # 每个客户端的初始化样本数
    combined_data = {}

    for i in range(len(initX)):
        combined_data[f"client{i}_initX"] = [str(x) for x in initX[i]]
        combined_data[f"client{i}_initY"] = [y for y in initY[i]]
    
    print(combined_data)
    df_init = pd.DataFrame(combined_data)

    # 写入 Excel 的两个 Sheet
    with pd.ExcelWriter(f"Fed/data{Rnd+20}.xlsx", engine="openpyxl") as writer:
        df_main.to_excel(writer, sheet_name="experiment_log", index=False)
        df_init.to_excel(writer, sheet_name="init_data", index=False)

def main(data,Rnd):
    client_num = 3
    init_num = 5
    rounds = 45

    filepath = "/home/user/wyn/Fedbay/data/experimentdata.xlsx"
    X_all, y_all = load_and_preprocess_data(filepath)

    clients = []
    global_X_pool = deepcopy(X_all)
    global_y_pool = deepcopy(y_all)
    initX,inity = [],[]
    for _ in range(client_num):
        X_init, y_init, global_X_pool, global_y_pool, inds = init_client(global_X_pool, global_y_pool, init_num)
        initX.append([[round(x * 100,1) for x in sublist]+[(1-sublist[-1])*100] for sublist in X_init])
        inity.append(deepcopy(y_init))
        opt = create_optimizer()
        opt.tell(X_init, [-y for y in y_init])
        
        #print("INIT",[[round(x * 100,1) for x in sublist]+[100-sublist[-1]] for sublist in X_init],y_init)
        clients.append({"opt": opt, "X": X_init, "y": y_init})

        data[f"client{_}y"] = []
        data[f"client{_}x"] = []
        #data[f"client{_}num"] = []
        data["assign"] = []
    #print("data",data)
    target = max(global_y_pool)
    endstep = -1
    for r in range(rounds):
        
        #1每次推荐单点的情况
        candidates = Single_per_client(clients,global_X_pool)
        
        #1.1 聚合策略：重新分配选择出的点，前期会重复实验，但是目前来看得到最优点用到的点数最少
        assignments, selected_candidates = aggregate_candidates11(candidates, self_prob=0.5)
        #1.2 聚合策略：重复的点优先分给推荐该点的且已有数据少的客户，不会出现重复实验，不保证每轮每个客户都有实验做
        #client_data_counts = [len(client["y"]) for client in clients]
        #assignments, selected_candidates = aggregate_candidates12(candidates, client_data_counts)
        #1.3 聚合策略：重复的点优先分给推荐该点的且已有数据多的客户，理解为有一部分主节点和一部分协同节点
        #client_data_counts = [len(client["y"]) for client in clients]
        #assignments, selected_candidates = aggregate_candidates13(candidates, client_data_counts)
        #1.4 聚合策略：预测最高的点优先分给所有节点中已有数据已有数据最少的
        #client_data_counts = [len(client["y"]) for client in clients]
        #assignments, selected_candidates = aggregate_candidates14(candidates, client_data_counts)
        #1.5 聚合策略：预测最高的优先给所有节点中已有数据最多的
        #client_data_counts = [len(client["y"]) for client in clients]
        #assignments, selected_candidates = aggregate_candidates15(candidates, client_data_counts)

        #2每次推荐多个点，保证每次每个人都有实验做
        #candidates = Multi_per_client(clients,global_X_pool,client_num)
        #2.1
        #assignments, selected_candidates = aggregate_candidates21(candidates)
        #2.2 
        #assignments, selected_candidates = aggregate_candidates22(candidates)
        
        used_indices = set()
        true_Y = []
        Maxy = 0
        sepy = 0
        
        for client_id, cand in enumerate(selected_candidates):
            
            if cand is None:
                continue
            
            x_sel, _, pool_idx = cand
            y_true = global_y_pool[pool_idx]

            true_Y.append(y_true)
            Maxy = max(Maxy,y_true)
            sepy+= y_true
            
            clients[client_id]["opt"].tell([x_sel.tolist()], [-y_true])
            clients[client_id]["X"].append(x_sel.tolist())
            clients[client_id]["y"].append(y_true)

            used_indices.add(pool_idx)
            #print(f"client_id:{client_id},x:{x_sel},y:{y_true}")
            if y_true == target:
                print("Global optimum found!")
                print(f"{len(global_X_pool)} points remains")
                endstep = r+1
            
            data[f"client{client_id}y"].append(y_true)
            part_x = np.round(x_sel*100,1).tolist()
            data[f"client{client_id}x"].append(part_x + [100-part_x[-1]])
            #print(len(global_X_pool),len(used_indices),454-len(global_X_pool)-len(used_indices))
            #data[f"client{client_id}num"].append(pool_idx+454-len(global_X_pool)-len(used_indices))
        
        for idx in sorted(used_indices, reverse=True):
            global_X_pool = np.delete(global_X_pool, idx, axis=0)
            global_y_pool = np.delete(global_y_pool, idx, axis=0)

        #print(f"Round {r+1},seprate:{true_Y[0]},{true_Y[1]},{true_Y[2]},Maxy:{Maxy},Avgy:{sepy/client_num}")
        #print(f"Round {r + 1}: Used assignments {[a for a in assignments]},Max y {Maxy}")
        data["Round"].append(r+1)
        
        data["assign"].append(assignments)
        data["Max_y"].append(Maxy)
        data["Avg_y"].append(sepy/client_num)
        
        '''if found:
            Save_data_to_exc(data,Rnd)

            return r+1,450-len(global_X_pool)'''
    #print("INIT",inity[0])

    Save_data_to_exc(data,Rnd,endstep,initX,inity)
    return r+1,450-len(global_X_pool)
    #print("联邦贝叶斯优化完成！")
    
if __name__ == "__main__":
    totround = 30
    '''with open("res.txt","w") as f:
        f.write('1.1\n')
        f.close()'''
    for i in range(totround):
        data = {
            "Round":[],
            "Max_y":[],
            "Avg_y":[],
        }
        rnd,pnum = main(data,i)
        
        '''with open("res.txt","a") as f:
            f.write(f'{rnd} {pnum}\n')
        '''

        