#全局推荐一个去替换掉在全局模型下表现最差的那个本地点
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from copy import deepcopy
import os
from skopt.space import Real, Integer
from skopt import Optimizer
import json
import itertools
import random
import GPy
from skopt.utils import cook_estimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

num_inducing = 20

def load_and_preprocess(filepath):
    df = pd.read_excel(filepath, skiprows=1)
    input_cols = ['Bi(%)', 'Fe(%)', 'Co(%)', 'Cu1(%)', 'Ni(%)', 'Mn(%)', 'L1(%)']#, 'L2(%)']
    target_col = 'K'
    X_all = df[input_cols].values
    y_all = df[target_col].values
    y_min,y_max = 0,0.35
    y_all = 0.2 + (0.6 - 0.2) * (y_all - y_min) / (y_max - y_min)
    X_all = X_all * 0.01
    #别混了初始点固定了X_all, y_all = shuffle(X_all, y_all)
    return X_all, y_all

custom_gp = GaussianProcessRegressor(
    kernel=Matern(nu=2.5, length_scale_bounds=(1e-2, 1e6)),  # 关键：上界设为 1e6
    alpha=1e-6,
    n_restarts_optimizer=10,
    random_state=42
)

def create_optimizer():
    return Optimizer(
        dimensions=[
            Real(0.7, 1.0), Real(0.0, 0.3), Real(0.0, 0.3), Real(0.0, 0.3),
            Real(0.0, 0.3), Real(0.0, 0.3), Integer(0, 1)#, Integer(0, 1)
        ],
        #base_estimator="GP",
        base_estimator = custom_gp,
        acq_func="EI",
        acq_optimizer="sampling",
        random_state=None
    )

def Load_fixed_initdata(datapath,idx,X_pool,y_pool, atol=1e-8, rtol=1e-5):
    datafile = os.path.join(datapath,"data"+str(idx),"init_data.xlsx")
    df = pd.read_excel(datafile)
    print("Load_fixed_initdata ",datafile)
    df_values = df.values.astype(float)
    df_x = df_values[:, 1:-1]
    df_y = df_values[:, -1]

    mask = np.ones(len(X_pool), dtype=bool)
    indices = []
    for i, (x_row, y_val) in enumerate(zip(df_x, df_y)):
        matches = np.all(np.isclose(X_pool, x_row, rtol=rtol, atol=atol), axis=1) & \
                  np.isclose(y_pool.reshape(-1), y_val, rtol=rtol, atol=atol)
        if np.any(matches):
            match_idx = np.where(matches)[0][0]  # 只取第一个
            mask[match_idx] = False
            indices.append(match_idx)
        else:
            print(f"Warning: 未找到匹配行 (第{i}行):", x_row, y_val)
    #print("The indices tobe checked is {}".format(indices))
    return X_pool[mask], y_pool[mask], df_x.tolist(), df_y.tolist(), indices

def Init_client(clients,num,global_X_pool,global_y_pool,datapath):
    global_X_pool, global_y_pool, X_init, y_init, inds = Load_fixed_initdata(datapath,num,global_X_pool, global_y_pool)
    #print("初始节点在所有数据中的编号：",inds)
    opt = create_optimizer()
    opt.tell(X_init, [-y for y in y_init])
    #print("INIT",[[round(x * 100,1) for x in sublist]+[100-sublist[-1]] for sublist in X_init],y_init)
    clients.append({"opt": opt, "X": X_init, "y": y_init})
    print("IN INIT_CLIENT: target_y",max(global_y_pool))
    return global_X_pool,global_y_pool,max(global_y_pool)

def propose_candidate(optimizer, X_pool):
    next_x = optimizer.ask()
    dists = np.linalg.norm(X_pool - np.array(next_x), axis=1)
    idx = np.argmin(dists)
    return X_pool[idx], idx

def Single_per_client(clients,global_X_pool):
    candidates = []
    for client in clients:
        x_cand, idx = propose_candidate(client["opt"], global_X_pool)
        # 模拟模型预测值为目标函数的当前最大值（先不看真实结果）
        estimator = cook_estimator(client["opt"].base_estimator_, client["opt"].space)
        # 手动 fit
        estimator.fit(client["opt"].Xi, client["opt"].yi)
        # 预测
        pred_y = -estimator.predict([x_cand])[0]
        #print("Snigle_per_client, pred_y:", pred_y)
        candidates.append((x_cand, pred_y, idx))
    return candidates

def inducing_proposal(indlist, xpool, pro_num,candidates):
    # 使用多个clent的inducing point和预测值，构造一个全局模型，在xpool中进行选点
    # indlist 为一个list，其中的元素为(indpoints, ypred)
    #把本地模型的推荐点搞进来评估一下
    tlxind = [ti[0] for ti in indlist]
    tly = [ti[1] for ti in indlist]
    gxind = np.concatenate(tlxind, axis = 0)
    gy = np.concatenate(tly, axis = 0)
    print("total inducing point:", gy.shape)
    # remove the duplicate ones
    values, indices, counts = np.unique(gxind, return_index=True, return_counts=True)
    gx = xpool[gxind[indices]]
    gy = gy[indices]
    print("after remove duplicate, total inducing point:",gx.shape, gy.shape)
    #print(gx,gy)
    m_full = GPy.models.GPRegression(gx, gy)
    m_full.optimize()
    y_pred, y_var = m_full.predict(xpool)
    idx = np.argmax(y_pred, axis = 0)   # 只有优化器minimize是用负值
    #print("total model predict:",y_pred[idx], max(y_pred))
    
    ret = [(xpool[idx].reshape(-1), y_pred[idx].item(), idx.item())]
    new_cand = []
    for cand in candidates:
        y_pred,y_var = m_full.predict(cand[0].reshape(1,-1))
        #print("OLD PRED {}, NEW PRED {} ".format(cand[1],y_pred[0][0]))
        new_cand.append((cand[0],y_pred[0][0],cand[2]))

    sorted_cand = sorted(new_cand, key=lambda x: x[1], reverse=True)
    for i in range(pro_num):
        ret.append(sorted_cand[i])
    return ret
    
def aggregate_candidates_with_inducing(allcandidates):
    allcan = shuffle(allcandidates)
    #print("all candidate ",allcan)
    return allcan

def Save_data_to_exc(output_path,rnd,endstep,clients,init_datanum):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = 'output'+str(rnd)+'_end'+str(endstep)+'.xlsx'
    print("SAVE to {}".format(filename))
    data = {}
    for i in range(len(clients)):
        #print("client{}:{}".format(i,clients[i]["y"]))
        data[f'client_{i+1}'] = clients[i]["y"][init_datanum:]
    
    df = pd.DataFrame(data)
    # 保存到Excel
    df.to_excel(os.path.join(output_path,filename), index=False) 
  
def One_trial(maxrounds,clients,global_X_pool,global_y_pool,target):
    endstep = -1
    init_datanum = len(clients[0]["y"])
    print("INIT_DATANUM ",init_datanum)
    for r in range(maxrounds):
        candidates = Single_per_client(clients,global_X_pool)
        #包含（推荐点，预测y值和节点编号）
        #print("In_One_Trials cand for all clents: ",candidates)
        
        indu_list=[]
        for cidx,client in enumerate(clients):
            #print("calculate sparse gp inducing points for client", cidx)
            locmodel = GPy.models.GPRegression(np.asarray(client["X"]), np.asarray(client["y"]).reshape(-1,1))
            locmodel.optimize()
            #rndidx = np.random.choice([i for i in range(len(global_X_pool))], num_inducing, replace=False)
            if len(global_X_pool) < num_inducing:
                rndidx = list(range(len(global_X_pool)))  # 选择所有元素
            else:
                rndidx = np.random.choice([i for i in range(len(global_X_pool))], num_inducing, replace=False)
            indups = global_X_pool[rndidx]
            y_pred, y_var = locmodel.predict(indups)
            #print("clip to 0,1, max", max(y_pred))
            y_pred = np.clip(y_pred, 0.0, 1.0) 
            fsensitivity =1.0
            epsilon = 2.0
            y_pred = y_pred + np.random.laplace(loc=0.4, scale=fsensitivity/epsilon, size=y_pred.shape)
            indu_list.append((np.asarray(rndidx), y_pred))
        

        #assignments, selected_candidates = aggregate_candidates(candidates, self_prob=0.5)
        #print(assignments)
        indu_prop =  inducing_proposal(indu_list, global_X_pool,len(clients)-1,candidates)
        
        
        selected_candidates = aggregate_candidates_with_inducing(indu_prop)

        used_indices = set()
        true_Y = []
        Maxy = 0
        sepy = 0
        
        for client_id, cand in enumerate(selected_candidates):
            if cand is None:
                continue 
            x_sel, _, pool_idx = cand
            y_true = global_y_pool[pool_idx]
            print("round {}, client {}, point idx{}, y_true {}.".format(r,client_id,pool_idx,y_true))
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
            
           
        for idx in sorted(used_indices, reverse=True):
            global_X_pool = np.delete(global_X_pool, idx, axis=0)
            global_y_pool = np.delete(global_y_pool, idx, axis=0)

        data["Round"].append(r+1)
        #data["assign"].append(assignments)
        data["Max_y"].append(Maxy)
        data["Avg_y"].append(sepy/len(clients))
        #if endstep != -1:
        #    break
        if len(global_X_pool) == 0:
            break
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"FedResultv6")
    Save_data_to_exc(output_path,rnd,endstep,clients,init_datanum)
    return endstep



if __name__ == '__main__':
    client_idx = [0,1,2]
    trials = 50
    filepath = "./data/experimentdata.xlsx"
    input_datapath = './SingleResult'
    
    X_all, y_all = load_and_preprocess(filepath)
    maxrounds = 200
    exprndlist = []

    for rnd in range(trials):
        clients = []
        data = {
            "Round":[],
            "Max_y":[],
            "Avg_y":[],
        }
        global_X_pool = deepcopy(X_all)
        global_y_pool = deepcopy(y_all)
        #这里的i决定调用的固定数据编号
        for i in client_idx:
            global_X_pool,global_y_pool,target = Init_client(clients,i,global_X_pool,global_y_pool,input_datapath)
            
        endstep = One_trial(maxrounds,clients,global_X_pool,global_y_pool,target)
        exprndlist.append(endstep)
    print("All endsteps:", exprndlist, sum(exprndlist)/len(exprndlist))

#012 all endstep: [10, 14, 14, 6, 25, 39, 25, 21, 38, 10, 18, 7, 15, 39, 13, 11, 30, 24, 21, 26, 14, 17, 16, 17, 7, 35, 12, 10, 29, 11, 18, 5, 32, 13, 31, 34, 10, 49, 8, 27, 33, 7, 35, 14, 10, 28, 27, 49, 8, 23] 20.7
#012 all [19, 17, 22, 6, 10, 14, 11, 10, 10, 25, 14, 23, 19, 10, 24, 31, 21, 11, 14, 6, 12, 18, 14, 17, 11, 17, 21, 38, 11, 26, 21, 11, 13, 11, 8, 15, 10, 14, 13, 10, 12, 16, 34, 9, 3, 17, 24, 5, 50, 26] 16.48
#456 all endstep: [34, 19, 8, 12, 2, 18, 18, 10, 15, 17, 13, 11, 8, 19, 15, 43, 6, 35, 25, 20, 16, 12, 10, 13, 13, 11, 8, 20, 34, 9, 47, 49, 6, 14, 26, 4, 15, 15, 8, 5, 9, 20, 28, 6, 16, 10, 12, 11, 10, 41] 16.92
