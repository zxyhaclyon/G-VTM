import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
import argparse
import shutil
warnings.filterwarnings("ignore")

his_len = 100
future_len = 100
time_len = his_len + future_len
N_IN_FEATURES = 10
N_OUT_FEATURES = 10
fz = 25
DOWN_SAMPLE = 5
agent_dict = {'car':1, 'truck':2, 'bus':2, 'van':2, 'trailer':2, 'bicycle':3, 'motorcycle':3, 'pedestrian':4}
normalize_dict = {'location0':{'x_min':-67.0118, 'x_max':62.2897, 'y_min':-36.3268, 'y_max':36.1300}
}
grid_dict={
    'location0':{'x_min':-68, 'x_max':64, 'y_min':-40, 'y_max':40}}

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


def find_neighboring_nodes(veh_df, end_frame, id0, lx, ly, upper_limit=5):
    def filter_ids(sdist, radius=60):
        if sdist[0] < radius:
            return True
        else:
            return False
    df2 = veh_df[(veh_df.frame == end_frame) & (veh_df.trackId != id0)]
    if df2.empty:
        return []
    dist = list(df2.apply(lambda x: (euclidian(lx, ly, x.xCenter, x.yCenter), x.trackId), axis=1))
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
    
def get_input_features(df, frame_start, frame_end, x0, y0):
    dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end)]
    x = list(map(lambda x:round(x,3), dfx.xCenter.values))[::DOWN_SAMPLE]
    y = list(map(lambda x:round(x,3), dfx.yCenter.values))[::DOWN_SAMPLE]
    # x = list(map(normalize_x, x))
    # y = list(map(normalize_y, y))
    # 对xy坐标做归一化
    vx = list(map(lambda x:round(x,3), dfx.xVelocity.values))[::DOWN_SAMPLE]
    vy = list(map(lambda x:round(x,3), dfx.yVelocity.values))[::DOWN_SAMPLE]
    ax = list(map(lambda x:round(x,3), dfx.xAcceleration.values))[::DOWN_SAMPLE]
    ay = list(map(lambda x:round(x,3), dfx.yAcceleration.values))[::DOWN_SAMPLE]
    heading_rad = list(map(lambda x:round(x,3), dfx.heading.values))[::DOWN_SAMPLE]
    
    return x, y, vx, vy, ax, ay, heading_rad

def get_grid(x, y):
    city_grid = grid_dict[city]
    x_max = city_grid['x_max']
    x_min = city_grid['x_min']
    y_max = city_grid['y_max']
    y_min = city_grid['y_min']
    x_length = x_max - x_min
    y_length = y_max - y_min
    x_id = list(map(lambda m: int(np.ceil(y_max - m)), y)) 
    y_id = list(map(lambda m: int(np.ceil(m - x_min)), x))
    return x_id, y_id

def get_grid_features(df, f, fp, sv_id):
    dfx = df[(df.trackId == sv_id) & (df.frame >= f) & (df.frame <= fp)]
    lx = list(map(lambda x:round(x,3), dfx.x.values))
    ly = list(map(lambda x:round(x,3), dfx.y.values))
    # 确认轨迹点所在的Grid-ID
    x_id, y_id = get_grid(lx, ly)
    return x_id, y_id


def get_target_features(df, frame_start, frame_end, n_features, agent_type, x0, y0):

    dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end)]
    
    x = list(map(lambda x:round(x - x0,3), dfx.xCenter.values))[::DOWN_SAMPLE]
    y = list(map(lambda x:round(x - y0,3), dfx.yCenter.values))[::DOWN_SAMPLE]
    x_id, y_id = get_grid(x, y)
    # x = list(map(normalize_x, x))
    # y = list(map(normalize_y, y))
    vx = list(map(lambda x:round(x,3), dfx.xVelocity.values))[::DOWN_SAMPLE]
    vy = list(map(lambda x:round(x,3), dfx.yVelocity.values))[::DOWN_SAMPLE]
    ax = list(map(lambda x:round(x,3), dfx.xAcceleration.values))[::DOWN_SAMPLE]
    ay = list(map(lambda x:round(x,3), dfx.yAcceleration.values))[::DOWN_SAMPLE]
    heading_rad = list(map(lambda x:round(x,3), dfx.heading.values))[::DOWN_SAMPLE]
    
    feat_stack = np.stack((x, y, x_id, y_id, vx, vy, ax, ay, heading_rad, agent_type), axis=1)
    return feat_stack

def get_adjusted_features(df, frame_start, frame_end, n_features, trackId, agent_type, x0, y0):
    return_array = np.full((frame_end - frame_start + 1, n_features), -1, dtype = np.float64)
    
    if trackId != -1:
        dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end) & (df.trackId == trackId)]
    
    x = list(map(lambda x:round(x - x0,3), dfx.xCenter.values))
    y = list(map(lambda x:round(x - y0,3), dfx.yCenter.values))
    x_id, y_id = get_grid(x, y)
    vx = list(map(lambda x:round(x,3), dfx.xVelocity.values))
    vy = list(map(lambda x:round(x,3), dfx.yVelocity.values))
    ax = list(map(lambda x:round(x,3), dfx.xAcceleration.values))
    ay = list(map(lambda x:round(x,3), dfx.yAcceleration.values))
    heading_rad = list(map(lambda x:round(x,3), dfx.heading.values))
    agent_type = [agent_type for i in range(len(x))]
    
    feat_stack = np.stack([x, y, x_id, y_id, vx, vy, ax, ay, heading_rad, agent_type], axis=1)
    return_array[-feat_stack.shape[0]:, :] = feat_stack
    return_array = return_array[::DOWN_SAMPLE, :]
    
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

# 查看长时间静止的车辆并去除
def remove_parked_vehicles(tracks, tracks_meta):
    parked_vehicles = tracks_meta[(tracks_meta.initialFrame == 0) &
                                  (tracks_meta.finalFrame == tracks_meta.finalFrame.max())]
    a = parked_vehicles.trackId.values
    tracks = tracks[~tracks['trackId'].isin(a)]
    tracks_meta = tracks_meta[~tracks_meta['trackId'].isin(a)]
    return tracks, tracks_meta

def remove_selected_vehicles(tracks, v_ids, rm=False):
    for v_id in v_ids:
        if rm:
            tracks = tracks.drop(tracks[(tracks.trackId == v_id)].index)
        else:
            tracks = tracks.drop(tracks[(tracks.trackId == v_id) &
                                        (tracks.xAcceleration == 0) &
                                        (tracks.yAcceleration == 0)].index)
    return tracks

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default='location0', type=str, help="location0")
    parser.add_argument("--data_save", default='rounD-Location0-new', type=str, help="[data save path->rounD-Location0]")
    parser.add_argument("--data_read", default='./raw_data/rounD_data/location0', type=str, help="data read path")
    args = parser.parse_args()
    city = args.city
    root_folder = args.data_read
    data_folder = args.data_save
    data_root = create_directories(data_folder)
    rec_ids = ['0' + str(f) if len(str(f)) < 2 else str(f) for f in range(2, 8 + 1)]
    window_stride = fz * 2
    
    # 指定要遍历的文件夹路径
    np.random.seed(1234)
    for r_id in rec_ids:
        s_dict = get_storage_dict()
        train_data, train_edge_feat, train_mask, train_target, train_target_grid = [], [], [], [], []
        valid_data, valid_edge_feat, valid_mask, valid_target, valid_target_grid = [], [], [], [], []
        test_data, test_edge_feat, test_mask, test_target, test_target_grid = [], [], [], [], []
        if r_id == '00':
            p0 = (115.51669730710512, -70.6429033531912)
        elif r_id == '01':
            p0 = (137.8338032894461, -61.07768929146573)
        elif r_id in ('02', '03', '04', '05', '06', '07', '08'):
            p0 = (79.0111777880229, -39.28401173837483)
        else:
            p0 = (80.97640635242064, -46.93989929086365)
        x0 = p0[0]
        y0 = p0[1]
        print(f'Starting with recording {r_id}')
    
        t_meta = pd.read_csv(f'{root_folder}/{r_id}_tracksMeta.csv')
        tracks = pd.read_csv(f'{root_folder}/{r_id}_tracks.csv', engine='pyarrow')
        # Perform some initial cleanup
        tracks, t_meta = remove_parked_vehicles(tracks, t_meta)
        if r_id == '03':
            tracks = remove_selected_vehicles(tracks, (15, 17))
        
        # 获取最大的frame_id和最大的timestamp_ms
        all_frame = max(tracks['frame'].values) + 1
       
        # 按照8:1:1划分数据集（设置随机采样区间）
        train_frames, val_frames, test_frames = get_frame_split(all_frame) 
        
        # Get data and store
        veh_ids = set(list(tracks['trackId'].values))
        agent_ids = list(sorted(list(veh_ids)))
        ii = tqdm(range(0, len(agent_ids)))
        for i in ii:
            id0 = agent_ids[i]
            df = tracks[tracks.trackId == id0]
            # 获取当前车辆的type，过滤非机动车
            atype = agent_dict[t_meta[t_meta.trackId == id0]['class'].iloc[0]]
            if atype in [3, 4]:
                continue
            frames = list(df.frame)
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
                for f in sub_frames[0:-time_len+1:window_stride]: # 对满足长度的车辆轨迹做滑窗处理
                    fp = f + his_len - 1
                    fT = fp + future_len
                    x, y, vx, vy, ax, ay, heading_rad = get_input_features(df, f, fp, x0, y0) # 历史窗口的特征作为输入
                    agent_ltype = [atype for i in range(len(x))]
                    neighbors = find_neighboring_nodes(tracks, fp, id0, x[-1], y[-1]) # 在历史窗口的最后一个帧根据距离选择邻居（可改进）
                    # 如果没有邻居车辆则筛去该轨迹
                    if not neighbors:
                        continue
                    input_array = np.full((6, 20, N_IN_FEATURES), -1, dtype = np.float64) # 输入输出向量维度确定
                    target_array = np.empty((20, N_OUT_FEATURES), dtype = np.float64)
                    # 网格ID
                    target_grid = np.full((20, 2), -1, dtype = np.int64)
                    
                    # 确认轨迹点所在的Grid-ID
                    x = [round(xx - x0, 3) for xx in x]
                    y = [round(yy - y0, 3) for yy in y]
                    x_id, y_id = get_grid(x, y)

                    input_array[0, :, :] = np.stack((x, y, x_id, y_id, vx, vy, ax, ay, heading_rad, agent_ltype), axis=1)
                    
                    target_array[:, :] = get_target_features(df, fp + 1, fT, N_OUT_FEATURES, agent_ltype, x0, y0)
                    tx_id, ty_id = get_grid(target_array[:,0], target_array[:,1])
                    target_grid = np.stack((tx_id, ty_id), axis=1)

                    # 筛去历史和未来都在一个网格的轨迹
                    if (len(set(x_id)) == 1 and len(set(y_id)) == 1) and (len(set(tx_id)) == 1 and len(set(ty_id)) == 1):
                        if set(x_id) == set(tx_id) and set(y_id) == set(ty_id):
                            continue
                    
                    n_SVs = len(neighbors)
                    for j, n in enumerate(range(0, n_SVs)):
                        (dist, sv_id) = neighbors[n]
                        natype = agent_dict[t_meta[t_meta.trackId == sv_id]['class'].iloc[0]]
                        input_array[j + 1, :, :] = get_adjusted_features(tracks, f, fp, N_IN_FEATURES, sv_id, natype, x0, y0)
                    
                    # Build edge feat
                    input_edge_feat = euclidian_sequence(input_array) # (his_len, neigh_node, 4)

                    # Compute masks
                    input_mask = (input_array != -1).astype(int)
                    input_mask_3d = (~np.all(input_mask == 0, axis=-1)).astype(int) # (node, his_len)
                    
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
        if not os.path.exists(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/training/observation/{r_id}'):
            os.mkdir(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/training/observation/{r_id}')
            os.mkdir(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/training/target/{r_id}')
            os.mkdir(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/validation/observation/{r_id}')
            os.mkdir(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/validation/target/{r_id}')
            os.mkdir(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/testing/observation/{r_id}')
            os.mkdir(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/testing/target/{r_id}')
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/training/observation/{r_id}/dat.npz', input = np.stack(train_data, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/training/observation/{r_id}/mask.npz', mask = np.stack(train_mask, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/training/observation/{r_id}/edge_feat.npz', edge = np.stack(train_edge_feat, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/training/target/{r_id}/label.npz', label = np.stack(train_target, axis=0))
        
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/validation/observation/{r_id}/dat.npz', input = np.stack(valid_data, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/validation/observation/{r_id}/mask.npz', mask = np.stack(valid_mask, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/validation/observation/{r_id}/edge_feat.npz', edge = np.stack(valid_edge_feat, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/validation/target/{r_id}/label.npz', label = np.stack(valid_target, axis=0))

        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/testing/observation/{r_id}/dat.npz', input = np.stack(test_data, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/testing/observation/{r_id}/mask.npz', mask = np.stack(test_mask, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/testing/observation/{r_id}/edge_feat.npz', edge = np.stack(test_edge_feat, axis=0))
        np.savez(f'/data/ZhangXinyue/Multi_Model/data/{data_root}/testing/target/{r_id}/label.npz', label = np.stack(test_target, axis=0))
        
        print(r_id, ' Training:', s_dict['training'], ' Validation:', s_dict['validation'], ' Testing:', s_dict['testing'])
