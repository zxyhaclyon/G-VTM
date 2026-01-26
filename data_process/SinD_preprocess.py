import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
import shutil
import argparse
warnings.filterwarnings("ignore")

INPUT_LENGTH = 4
PRED_HORIZON = 4
N_IN_FEATURES = 10
N_OUT_FEATURES = 10
fz = 10
DOWN_SAMPLE = 2
agent_dict = {'car':1, 'bus':2, 'truck':2, 'motorcycle':3, 'bicycle':3, 'tricycle':3, 'pedestrian':4}
normalize_dict = {'Tianjin':{'x_min':-24.1609, 'x_max':55.5728, 'y_min':-8.9886, 'y_max':40.0838}, \
                  'Changchun':{'x_min':-95.24867, 'x_max':54.0649, 'y_min':-82.8118, 'y_max':76.3586},\
                  'Xian':{'x_min':-95.8782, 'x_max':75.5598, 'y_min':-20.9430, 'y_max':75.3351}, \
                  'Chongqing':{'x_min':-45.1188, 'x_max':51.0306, 'y_min':-26.4988, 'y_max':64.2722}
}
grid_dict={
    'Tianjin':{'x_min':-33, 'x_max':63, 'y_min':-17, 'y_max':47}, 
}

def create_directories(data_folder):
    root = f'./data/{data_folder}'
    top_dirs = ['training', 'validation', 'testing']
    sub_dirs = ['observation', 'target']
    for d in top_dirs:
        top_dir = f'{root}/{d}'
        for s in sub_dirs:
            sub_dir = f'{top_dir}/{s}'
            if os.path.exists(sub_dir):  # and overwrite:
                shutil.rmtree(sub_dir)
            os.makedirs(sub_dir)
    return data_folder


def euclidian(x1, y1, x2, y2):
    from math import sqrt
    r = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return r


def maneuver_label(heading_start, heading_end):
    turn_alts = np.array([-np.pi / 2, 0, np.pi / 2, np.pi])
    tmp = heading_end - heading_start
    head_diff = turn_alts - np.radians(tmp)
    wrap_to_pi = np.arctan2(np.sin(head_diff), np.cos(head_diff))
    return np.argmin(np.abs(wrap_to_pi)), tmp


def find_neighboring_nodes(veh_df, ped_df, end_frame, id0, lx, ly, upper_limit=5):
    def filter_ids(sdist, radius=60):
        if sdist[0] < radius:
            return True
        else:
            return False
    df1 = veh_df[(veh_df.frame_id == end_frame) & (veh_df.track_id != id0)]
    df2 = ped_df[(ped_df.frame_id == end_frame)]
    all_df = pd.concat([df1, df2], axis=0)
    if all_df.empty:
        return []
    if not all_df.empty:
        dist = list(all_df.apply(lambda x: (euclidian(lx, ly, x.x, x.y), x.track_id), axis=1))
    dist = list(filter(filter_ids, dist))
    dist_sorted = sorted(dist)
    del dist_sorted[upper_limit:]
    return dist_sorted

def normalize_x(x):
    norm_dict = normalize_dict[city]
    x_min, x_max = norm_dict['x_min'], norm_dict['x_max']
    out = round(2 * (x - x_min) / (x_max - x_min) - 1, 4)
    return out

def normalize_y(y):
    norm_dict = normalize_dict[city]
    y_min, y_max = norm_dict['y_min'], norm_dict['y_max']
    out = round(2 * (y - y_min) / (y_max - y_min) - 1, 4)
    return out
    
def get_input_features(df, frame_start, frame_end):
    dfx = df[(df.frame_id >= frame_start) & (df.frame_id <= frame_end)]
    x = list(map(lambda x:round(x,3), dfx.x.values))
    y = list(map(lambda x:round(x,3), dfx.y.values))
    # x = list(map(normalize_x, x))
    # y = list(map(normalize_y, y))
    # 对xy坐标做归一化
    vx = list(map(lambda x:round(x,3), dfx.vx.values))
    vy = list(map(lambda x:round(x,3), dfx.vy.values))
    ax = list(map(lambda x:round(x,3), dfx.ax.values))
    ay = list(map(lambda x:round(x,3), dfx.ay.values))
    heading_rad = list(map(lambda x:round(x,3), dfx.heading_rad.values))
    agent_type = list(map(lambda x:agent_dict[x], dfx.agent_type.values))
    return x, y, vx, vy, ax, ay, heading_rad, agent_type

def get_grid(x, y):
    city_grid = grid_dict[city]
    # 缩放到224的网格中
    x_min = city_grid['x_min']
    y_max = city_grid['y_max']
    y_id = list(map(lambda m: int(np.ceil(y_max - m)), y)) 
    x_id = list(map(lambda m: int(np.ceil(m - x_min)), x))
    return x_id, y_id

def get_grid_features(df, f, fp, sv_id):
    dfx = df[(df.track_id == sv_id) & (df.frame_id >= f) & (df.frame_id <= fp)]
    lx = list(map(lambda x:round(x,3), dfx.x.values))
    ly = list(map(lambda x:round(x,3), dfx.y.values))
    # 确认轨迹点所在的Grid-ID
    x_id, y_id = get_grid(lx, ly)
    return x_id, y_id


def get_target_features(df, frame_start, frame_end, n_features):
    return_array = np.empty((frame_end - frame_start + 1, n_features), dtype = np.float64)
    dfx = df[(df.frame_id >= frame_start) & (df.frame_id <= frame_end)]
    
    first_frame = dfx.frame_id.values[0]
    x = list(map(lambda x:round(x,3), dfx.x.values))
    y = list(map(lambda x:round(x,3), dfx.y.values))
    x_id, y_id = get_grid(x, y)
    # x = list(map(normalize_x, x))
    # y = list(map(normalize_y, y))
    vx = list(map(lambda x:round(x,3), dfx.vx.values))
    vy = list(map(lambda x:round(x,3), dfx.vy.values))
    ax = list(map(lambda x:round(x,3), dfx.ax.values))
    ay = list(map(lambda x:round(x,3), dfx.ay.values))
    heading_rad = list(map(lambda x:round(x,3), dfx.heading_rad.values))
    agent_type = list(map(lambda x:agent_dict[x], dfx.agent_type.values))
    
    feat_stack = np.stack((x, y, x_id, y_id, vx, vy, ax, ay, heading_rad, agent_type), axis=1)
    # 待预测节点和其邻居节点的轨迹长度不一，缺失的地方用NAN代替
    return_array[0:feat_stack.shape[0], :] = feat_stack
    return return_array

def get_veh_adjusted_features(df, frame_start, frame_end, n_features, trackId = -1):
    return_array = np.full((frame_end - frame_start + 1, n_features), -1, dtype = np.float64)
    
    if trackId != -1:
        dfx = df[(df.frame_id >= frame_start) & (df.frame_id <= frame_end) & (df.track_id == trackId)]
    
    first_frame = dfx.frame_id.values[0]
    frame_offset = first_frame - frame_start
    
    x = list(map(lambda x:round(x,3), dfx.x.values))
    y = list(map(lambda x:round(x,3), dfx.y.values))
    x_id, y_id = get_grid(x, y)
    # x = list(map(normalize_x, x))
    # y = list(map(normalize_y, y))
    vx = list(map(lambda x:round(x,3), dfx.vx.values))
    vy = list(map(lambda x:round(x,3), dfx.vy.values))
    ax = list(map(lambda x:round(x,3), dfx.ax.values))
    ay = list(map(lambda x:round(x,3), dfx.ay.values))
    heading_rad = list(map(lambda x:round(x,3), dfx.heading_rad.values))
    agent_type = list(map(lambda x:agent_dict[x], dfx.agent_type.values))
    
    feat_stack = np.stack([x, y, x_id, y_id, vx, vy, ax, ay, heading_rad, agent_type], axis=1)
    return_array[-feat_stack.shape[0]:, :] = feat_stack
    
    return return_array

def get_ped_adjusted_features(df, frame_start, frame_end, n_features, trackId = -1):
    return_array = np.full((frame_end - frame_start + 1, n_features), -1, dtype = np.float64)
    
    if trackId != -1:
        dfx = df[(df.frame_id >= frame_start) & (df.frame_id <= frame_end) & (df.track_id == trackId)]
    
    first_frame = dfx.frame_id.values[0]
    frame_offset = first_frame - frame_start
    
    x = list(map(lambda x:round(x,3), dfx.x.values))
    y = list(map(lambda x:round(x,3), dfx.y.values))
    x_id, y_id = get_grid(x, y)
    vx = list(map(lambda x:round(x,3), dfx.vx.values))
    vy = list(map(lambda x:round(x,3), dfx.vy.values))
    ax = list(map(lambda x:round(x,3), dfx.ax.values))
    ay = list(map(lambda x:round(x,3), dfx.ay.values))
    heading_rad = [0 for i in range(len(x))]
    agent_type = [4 for i in range(len(x))]
    
    feat_stack = np.stack([x, y, x_id, y_id, vx, vy, ax, ay, heading_rad, agent_type], axis=1)
    return_array[-feat_stack.shape[0]:, :] = feat_stack
    
    return return_array


def get_storage_dict():
    dd = {}
    for t in ['training', 'validation', 'testing']:
        dd[t] = 0
    return dd


def euclidian_distance(x1, x2):
    # x1.shape (2, )
    # x2.shape (2, )
    return [round(np.sqrt(np.sum((x1 - x2) ** 2)), 2), np.arctan2(x2[1] - x1[1], x2[0] - x1[0])]


def euclidian_instance(inp):
    # inp.shape (n_vehicles, n_features)
    n_vehicles = inp.shape[0]
    output = []
    for v_neighbor in range(1, n_vehicles):
        if inp[v_neighbor, 0] != -1:
            d = euclidian_distance(inp[0, :2], inp[v_neighbor, :2])
            d.extend([inp[0, 0]-inp[v_neighbor, 0], inp[0, 1]-inp[v_neighbor, 1]])
            output.append(d)
        else:
            output.append([-1, -1, -1, -1])
    return output


def euclidian_sequence(inp):
    # inp.shape (n_vehicles, seq_len, n_features)
    seq_len = inp.shape[1]
    output = []
    for i in range(0,seq_len):
        output.append(euclidian_instance(inp[:, i]))
    output = np.array(output)
    return output


def get_frame_split(n_frames):
    all_frames = list(range(0, n_frames))
    # first variant 80-10-10
    tr = [0, all_frames[int(0.6 * n_frames) - 1]]
    val = [all_frames[int(0.6 * n_frames)], all_frames[int(0.8 * n_frames) - 1]]
    test = [all_frames[int(0.8 * n_frames)], all_frames[-1]]
    return tr, val, test


def which_set(v_frames, tr, val, test, time_len):
    assert v_frames[-1] > v_frames[0]
    curr = dict()
    # 如果在训练集阶段开始的frame
    if v_frames[0] >= tr[0] and v_frames[0] <= tr[-1]:
        if v_frames[-1] <= tr[-1]:
            curr['training'] = [0, v_frames[-1] - v_frames[0]]
        elif val[0] <= v_frames[-1] <= val[-1]:
            if (val[0] - v_frames[0]) >= time_len:
                curr['training'] = [0, val[0] - v_frames[0] - 1]
            if (v_frames[-1] - val[0] + 1) >= time_len:
                curr['validation'] = [val[0] - v_frames[0], v_frames[-1] - v_frames[0]]
        elif test[0] <= v_frames[-1] <= test[-1]:
            curr['validation'] = [val[0] - v_frames[0], val[-1] - v_frames[0]]
            if (val[0] - v_frames[0]) >= time_len:
                curr['training'] = [0, val[0] -v_frames[0] - 1]
            if (v_frames[-1] - test[0] + 1) >= time_len:
                curr['testing'] = [test[0] - v_frames[0], v_frames[-1] - v_frames[0]]
    # 如果在验证集阶段开始的frame
    if v_frames[0] >= val[0] and v_frames[0] <= val[-1]:
        if v_frames[-1] <= val[-1]:
            curr['validation'] = [0, v_frames[-1] - v_frames[0]]
        elif test[0] <= v_frames[-1] <= test[-1]:
            if (test[0] - v_frames[0]) >= time_len:
                curr['validation'] = [0, test[0] - v_frames[0] - 1]
            if (v_frames[-1] - test[0] + 1) >= time_len:
                curr['testing'] = [test[0] - v_frames[0], v_frames[-1] - v_frames[0]]
    # 如果在测试集阶段开始的frame
    if v_frames[0] >= test[0] and v_frames[-1] <= test[-1]:
        curr['testing'] = [0, v_frames[-1] - v_frames[0]]
    return curr

def city_traffic_light(trafficlight, column, all_timestamp, add_count):
    tra_dict = dict()
    ind = 0
    for ind in range(len(trafficlight)-1):
        trl1 = list(trafficlight.loc[ind][column])
        trl2 = list(trafficlight.loc[ind+1][column])
        if trl2[0] < 0:
            continue
        if trl1[0] < 0 and trl2[0] > 0:
            tra_dict[(0, trl2[0])] = list(trl1[1:]) * add_count
        if trl1[0] >= 0 and trl2[0] > 0:
            tra_dict[(trl1[0], trl2[0])] = list(trl1[1:]) * add_count
    # 加入最后时刻的信号灯状态
    trl1 = list(trafficlight.loc[ind+1][column])
    if all_timestamp > trl1[0]:
        tra_dict[(trl1[0], all_timestamp+1)] = list(trl1[1:]) * add_count
    return tra_dict

def create_traffic_light_dict(trafficlight, city, all_timestamp):
    feat_col = trafficlight.columns.tolist()
    column = ['timestamp(ms)']
    if city == 'Chongqing' or city == 'Tianjin':
        column.extend(list(feat_col[-8:]))
        tra_dict = city_traffic_light(trafficlight, column, all_timestamp, add_count = 1)
    elif city == 'Changchun' or city == 'Xian':
        column.extend(list(feat_col[-2:]))
        tra_dict = city_traffic_light(trafficlight, column, all_timestamp, add_count = 4)
    return tra_dict

def track_light_concat(tracks, traffic_light_dict, if_veh):
    origin_col = ['track_id','frame_id', 'timestamp_ms', 'agent_type','x','y','vx','vy','ax','ay']
    ori_tracks = tracks[origin_col]
    def match_timestamp(timestamp):
        for interval, values in traffic_light_dict.items():
            start, end = interval
            if start <= timestamp < end:
                result = values
                return result
        print(f'{timestamp}没有匹配上信号灯状态')
    # 应用函数到 DataFrame的每一行
    results = ori_tracks['timestamp_ms'].apply(match_timestamp)
    # 创建新的 8 个列
    light_columns = [f'light{i}' for i in range(1, 9)]
    ori_tracks[light_columns] = pd.DataFrame(results.tolist(), index=ori_tracks.index)
    return ori_tracks

# 查看长时间静止的车辆并去除
def check_park_agent(tracks, track_ids):
    park_agents = []
    for track_id in track_ids:
        cu_track = tracks[tracks['track_id'] == track_id]
        park_track = cu_track[(cu_track['vx'] == 0) & (cu_track['vy'] == 0)]
        ini_frame = park_track['frame_id'].min()
        end_frame = park_track['frame_id'].max()
        if end_frame - ini_frame > 1200:
            park_agents.append(track_id)
    return park_agents

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default='Tianjin', type=str, help="Tianjin")
    parser.add_argument("--data_save", default='SinD-Tianjin-new', type=str, help="[data save path->SinD-Tianjin]")
    parser.add_argument("--data_read", default='./raw_data/SinD_data/tianjin', type=str, help="data read path")
    args = parser.parse_args()
    city = args.city
    root_folder = args.data_read
    data_folder = args.data_save
    data_root = create_directories(data_folder)
    sub_folders = []
    # 使用os.walk()遍历文件夹
    for root, dirs, files in os.walk(root_folder):
        # 遍历子文件夹
        for dir in dirs:
            # 子文件夹的完整路径
            sub_folder_path = os.path.join(root_folder, dir)
            sub_folders.append(sub_folder_path)
    np.random.seed(1234)
    s_dict = get_storage_dict()
    sub_folders = sorted(sub_folders)
    for sub_folder in sub_folders:
        train_data, train_edge_feat, train_mask, train_target, train_target_grid = [], [], [], [], []
        valid_data, valid_edge_feat, valid_mask, valid_target, valid_target_grid = [], [], [], [], []
        test_data, test_edge_feat, test_mask, test_target, test_target_grid = [], [], [], [], []
        record_id = list(sub_folder.split('/'))[-1]
        print(f'Starting with recording {record_id}')
        # 遍历文件夹中的数据集
        file_name = []
        for root, dirs, files in os.walk(sub_folder):
            for file in files:
                # 检查文件扩展名是否为.csv
                if file.endswith('.csv'):
                    file_name.append(file)
        for name in file_name:
            if name.startswith('Veh_smoothed_tracks'):
                veh_tracks = pd.read_csv(f'{sub_folder}/{name}')
            if name.startswith('Ped_smoothed_tracks'):
                ped_tracks = pd.read_csv(f'{sub_folder}/{name}')
        # 过滤掉静止的轨迹数据
        veh_ids = set(veh_tracks['track_id'].values)
        park_veh = check_park_agent(veh_tracks, veh_ids)
        veh_tracks = veh_tracks[~veh_tracks['track_id'].isin(park_veh)]
        veh_ids = set(list(veh_tracks['track_id'].values))

        ped_ids = set(ped_tracks['track_id'].values)
        park_ped = check_park_agent(ped_tracks, ped_ids)
        ped_tracks = ped_tracks[~ped_tracks['track_id'].isin(park_ped)]
        # 获取最大的frame_id和最大的timestamp_ms
        all_frame = max(veh_tracks['frame_id'].values) + 1
        all_timestamp = max(veh_tracks['timestamp_ms'].values)
        # 按照8:1:1划分数据集（设置随机采样区间）
        train_frames, val_frames, test_frames = get_frame_split(all_frame) 
        his_len = 40
        future_len = 40
        time_len = his_len + future_len
        # Get data and store
        veh_ids = set(list(veh_tracks['track_id'].values))
        agent_ids = list(sorted(list(veh_ids)))
        ii = tqdm(range(0, len(agent_ids)))
        # ii = tqdm(range(0, 20))
        for i in ii:
            id0 = agent_ids[i]
            df = veh_tracks[veh_tracks.track_id == id0]
            # 获取当前车辆的type，过滤非机动车
            atype = agent_dict[df['agent_type'].values[0]]
            if atype in [3, 4]:
                continue
            frames = list(df.frame_id)
            # 不满足时间窗口长度的轨迹删除
            if len(frames) < time_len: 
                continue
            # 轨迹划分数据集(可能长轨迹会划分进多个数据集)
            curr_split = which_set(frames, train_frames, val_frames, test_frames, time_len)
            if not curr_split:
                # If a vehicle is within frames which are overlapping the sets
                continue
            for curr_set, indexl in curr_split.items():
                start_index = indexl[0]
                end_index =  indexl[-1]
                sub_frames = frames[start_index : end_index+1]
                for f in sub_frames[0:-time_len+1:15]: # 对满足长度的车辆轨迹做滑窗处理
                    fp = f + his_len - 1
                    fT = fp + future_len
                    x, y, vx, vy, ax, ay, heading_rad, agent_type = get_input_features(df, f, fp) # 历史窗口的特征作为输入
                    neighbors = find_neighboring_nodes(veh_tracks, ped_tracks, fp, id0, x[-1], y[-1]) # 在历史窗口的最后一个帧根据距离选择邻居（可改进）
                    # 如果没有邻居车辆则筛去该轨迹
                    if not neighbors:
                        continue
                    input_array = np.full((6, his_len, N_IN_FEATURES), -1, dtype = np.float64) # 输入输出向量维度确定
                    target_array = np.empty((future_len, N_OUT_FEATURES), dtype = np.float64)
                    
                    # 确认轨迹点所在的Grid-ID
                    x_id, y_id = get_grid(x, y)
                    input_array[0, :, :] = np.stack((x, y, x_id, y_id, vx, vy, ax, ay, heading_rad, agent_type), axis=1)
                    
                    target_array[:, :] = get_target_features(df, fp + 1, fT, N_OUT_FEATURES)
                    tx_id, ty_id = get_grid(target_array[:,0], target_array[:,1])

                    # 筛去历史和未来都在一个网格的轨迹
                    if (len(set(x_id)) == 1 and len(set(y_id)) == 1) and (len(set(tx_id)) == 1 and len(set(ty_id)) == 1):
                        if set(x_id) == set(tx_id) and set(y_id) == set(ty_id):
                            continue
                    
                    n_SVs = len(neighbors)
                    for j, n in enumerate(range(0, n_SVs)):
                        (dist, sv_id) = neighbors[n]
                        if type(sv_id) == str:
                            input_array[j + 1, :, :] = get_ped_adjusted_features(ped_tracks, f, fp, N_IN_FEATURES, sv_id)
                        else:
                            input_array[j + 1, :, :] = get_veh_adjusted_features(veh_tracks, f, fp, N_IN_FEATURES, sv_id)
                    
                    # Build edge feat
                    
                    input_edge_feat = euclidian_sequence(input_array) # (his_len, neigh_node, 4)

                    # Compute masks
                    input_mask = (input_array != -1).astype(int)
                    input_mask_3d = (~np.all(input_mask == 0, axis=-1)).astype(int) # (node, his_len)
                    
                    # 降采样
                    input_array = input_array[:,::DOWN_SAMPLE,:]
                    target_array = target_array[::DOWN_SAMPLE,:]
                    input_edge_feat = input_edge_feat[::DOWN_SAMPLE,:,:]
                    input_mask_3d = input_mask_3d[:,::DOWN_SAMPLE]
                    # 'training', 'validation', 'testing'
                    
                    if curr_set == 'training':
                        train_data.append(input_array)
                        train_target.append(target_array)
                        train_edge_feat.append(input_edge_feat)
                        train_mask.append(input_mask_3d)

                    elif curr_set == 'validation':
                        valid_data.append(input_array)
                        valid_target.append(target_array)
                        valid_edge_feat.append(input_edge_feat)
                        valid_mask.append(input_mask_3d)
                   
                    elif curr_set == 'testing':
                        test_data.append(input_array)
                        test_target.append(target_array)
                        test_edge_feat.append(input_edge_feat)
                        test_mask.append(input_mask_3d)
                       

                    s_dict[curr_set] += 1
        
        # 一个城市的一段采集时间代表一个文件
        if not os.path.exists(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/training/observation/{record_id}'):
            os.mkdir(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/training/observation/{record_id}')
            os.mkdir(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/training/target/{record_id}')
            os.mkdir(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/validation/observation/{record_id}')
            os.mkdir(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/validation/target/{record_id}')
            os.mkdir(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/testing/observation/{record_id}')
            os.mkdir(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/testing/target/{record_id}')
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/training/observation/{record_id}/dat.npz', input = np.stack(train_data, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/training/observation/{record_id}/mask.npz', mask = np.stack(train_mask, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/training/observation/{record_id}/edge_feat.npz', edge = np.stack(train_edge_feat, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/training/target/{record_id}/label.npz', label = np.stack(train_target, axis=0))
        
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/validation/observation/{record_id}/dat.npz', input = np.stack(valid_data, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/validation/observation/{record_id}/mask.npz', mask = np.stack(valid_mask, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/validation/observation/{record_id}/edge_feat.npz', edge = np.stack(valid_edge_feat, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/validation/target/{record_id}/label.npz', label = np.stack(valid_target, axis=0))

        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/testing/observation/{record_id}/dat.npz', input = np.stack(test_data, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/testing/observation/{record_id}/mask.npz', mask = np.stack(test_mask, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/testing/observation/{record_id}/edge_feat.npz', edge = np.stack(test_edge_feat, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/testing/target/{record_id}/label.npz', label = np.stack(test_target, axis=0))
        
        print(record_id, ' Training:', s_dict['training'], ' Validation:', s_dict['validation'], ' Testing:', s_dict['testing'])
