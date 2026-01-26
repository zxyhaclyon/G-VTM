import torch
import torch.utils.data as data
import numpy as np
import os
import cv2
import colorsys

class LitDataModule():
    def __init__(self, args):
        self.patch_size = args.vit_patch_size
        self.grid_size = args.grid_size
        self.batch_size = args.batch_size
        self.data_root = args.data_root
        self.map_root = args.map_root
        self.small_ds = args.small_ds
        self.local_image_size = args.local_image_size

    def train_dataloader(self):
        dataset = TrajectoryPredictionDataset('training',
                                              self.patch_size, self.grid_size, self.local_image_size,
                                              self.data_root, self.map_root,
                                              small=self.small_ds,
                                              few_port = 1)
        print('训练集的batch数量:', int(dataset.data_num / self.batch_size))
        return torch.utils.data.DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          drop_last = True,
                          collate_fn=collate_session_based)

    def val_dataloader(self):
        dataset = TrajectoryPredictionDataset('validation',
                                              self.patch_size, self.grid_size, self.local_image_size,
                                              self.data_root, self.map_root,
                                              small=self.small_ds,
                                              few_port = 1)
        print('验证集的batch数量:', int(dataset.data_num / self.batch_size))
        return torch.utils.data.DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last = True,
                          collate_fn=collate_session_based)

    def test_dataloader(self):
        dataset = TrajectoryPredictionDataset('testing',
                                              self.patch_size, self.grid_size, self.local_image_size,
                                              self.data_root, self.map_root,
                                              small=self.small_ds,
                                              few_port = 1)
        print('测试集的batch数量:', int(dataset.data_num / self.batch_size))
        return torch.utils.data.DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=True,
                          collate_fn=collate_session_based)


class TrajectoryPredictionDataset(data.Dataset):
    def __init__(self,
                 train_test: str,
                 patch_size, grid_size, local_image_size,
                 data_set_src: str, 
                 map_set_src: str,
                 small: bool = False,
                 few_port: float = None):
        self.mode = train_test
        self.data_root = data_set_src
        self.map_root = map_set_src
        self.map_width = 0
        self.map_height = 0
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.local_image_size = local_image_size

        # 遍历文件夹中的数据集
        obs_sub_folder = f'{self.data_root}/{self.mode}/observation'
        target_sub_folder = f'{self.data_root}/{self.mode}/target'
        input_data, edge_feat, mask, target_data = [], [], [], []
        obs_sub_folders, target_sub_folders = [], []
        for root, dirs, files in os.walk(obs_sub_folder):
            for dir in dirs:
                sub_folder_path = os.path.join(obs_sub_folder, dir)
                obs_sub_folders.append(sub_folder_path)
        for root, dirs, files in os.walk(target_sub_folder):
            for dir in dirs:
                sub_folder_path = os.path.join(target_sub_folder, dir)
                target_sub_folders.append(sub_folder_path)
        
        for name in obs_sub_folders:
            inp = np.load(f'{name}/dat.npz')['input']
            input_data.append(inp)
            his_mask = np.load(f'{name}/mask.npz')['mask']
            mask.append(his_mask)
            edge = np.load(f'{name}/edge_feat.npz')['edge']  
            edge_feat.append(edge)

        for name in target_sub_folders:
            label = np.load(f'{name}/label.npz')['label']
            target_data.append(label)
        
        # read static map
        static_map = cv2.imread(self.map_root)
        self.map_image = cv2.cvtColor(static_map, cv2.COLOR_BGR2RGB)
        
        self.input_data = np.concatenate(input_data, axis = 0)[::few_port]
        self.mask = np.concatenate(mask, axis = 0)[::few_port]
        self.neighbor_dis = np.concatenate(edge_feat, axis = 0)[::few_port]
        self.target_data = np.concatenate(target_data, axis = 0)[::few_port]
        self.his_time = self.input_data.shape[2]
        self.data_num = self.input_data.shape[0]
        if small:
            # Smaller version for dry runs
            self.data_num = 32 * 3

    def __len__(self):
        return self.data_num
    
    def st_trans(self, traj_grid):
        agent_dict = dict()
        count = 0
        first_grid, second_grid = 0, 0
        for j in range(traj_grid.shape[0]):
            if traj_grid[j,0] == -1 or traj_grid[j,1] == -1:
                continue
            if (traj_grid[j,0],traj_grid[j,1]) not in agent_dict:
                count += 1
                if count == 1:
                    first_grid = (traj_grid[j,0],traj_grid[j,1])
                if count == 2:
                    second_grid = (traj_grid[j,0],traj_grid[j,1])
                agent_dict[(traj_grid[j,0],traj_grid[j,1])] = np.array([0,0,0])
                dx, dy = 1, 0
                if j != 0:
                    dx = traj_grid[j,0] - traj_grid[j-1,0]
                    dy = traj_grid[j,1] - traj_grid[j-1,1]
                    if dx != 0 and abs(dx) != 1:
                        dx = int(dx/abs(dx))
                    if dy != 0 and abs(dy) != 1:
                        dy = int(dy/abs(dy))
                agent_dict[(traj_grid[j,0],traj_grid[j,1])][0] = dx
                agent_dict[(traj_grid[j,0],traj_grid[j,1])][1] = dy
                agent_dict[(traj_grid[j,0],traj_grid[j,1])][2] = j
        if second_grid:
            agent_dict[first_grid][:2] = agent_dict[second_grid][:2]
        return agent_dict

    def rgb_to_hsl(self, rgb):
        r, g, b = rgb[0]/255, rgb[1]/255, rgb[2]/255
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return (h, s, l)
    def hsl_to_rgb(self, h, s, l):
        r, g, b = colorsys.hls_to_rgb(h, l, s)  # 注意参数顺序！
        return (int(r*255), int(g*255), int(b*255))
    
    def trans_to_image(self, agent_dict, image, grid_size):
        # 不同转移方向不同色相，停留时间控制亮度
        hues = {
        (-1,0): (255, 0, 0),    # 红 - 上
        (-1,1): (255, 191, 0),    # 橙黄 - 右上
        (0,1): (128, 255, 0),    # 黄绿 - 右
        (1,1): (0, 255, 64),     # 春绿 - 右下
        (1,0): (0, 255, 255),       # 青 - 下
        (1,-1): (0, 64, 255),       # 靛蓝 - 左下
        (0,-1): (128, 0, 255),     # 紫罗兰 - 左
        (-1,-1): (255, 0, 191),     # 品红 - 左上
        }
        transport = list(agent_dict.keys())
        time = np.array([v[-1] for k,v in agent_dict.items()])
        norm_time = time / self.his_time
        ind = 0
        for x,y in transport:
            x, y = int(x), int(y)
            dx, dy = agent_dict[(x,y)][0], agent_dict[(x,y)][1]
            base_color = hues[(dx, dy)]
            h,s,_ = self.rgb_to_hsl(base_color)
            brightness = 0.3 + 0.7 * norm_time[ind]
            color = self.hsl_to_rgb(h,s,brightness)
            image[x*grid_size:(x+1)*grid_size, y*grid_size:(y+1)*grid_size] = np.array(color).astype(np.uint8)
            ind += 1
        return image
    
    def pad_local_image(self, local_image):
        pad_local_iamge = np.full((self.local_image_size, self.local_image_size, 3), 255, dtype = np.uint8)
        H, W = local_image.shape[0], local_image.shape[1]
        if H != self.local_image_size or W != self.local_image_size:
            pad_h = self.local_image_size - H
            pad_w = self.local_image_size - W
            pad_top = pad_h // 2
            pad_bottom = self.local_image_size - (pad_h - pad_top)
            pad_left = pad_w // 2
            pad_right = self.local_image_size - (pad_w - pad_left)
            pad_local_iamge[pad_top:pad_bottom, pad_left:pad_right, :] = local_image
            return pad_local_iamge
        else:
            return local_image
            
    
    def create_local_trajectory(self, traj_grid, traj_image, max_patch_xid, max_patch_yid, patch_grid_num):
        def map_patch_idx(center_grid):
            grid_xid, grid_yid = center_grid[0], center_grid[1]
            return (int(grid_xid/patch_grid_num), int(grid_yid/patch_grid_num))

        traj_grid = traj_grid[traj_grid[:,0] != -1]
        center_grid = traj_grid[traj_grid.shape[0]//2]
        center_patch_idx = map_patch_idx(center_grid)
        center_xpid, center_ypid = center_patch_idx[0], center_patch_idx[-1]
        local_patch_xids = [max(center_xpid - 6, 0), min(center_xpid + 7, max_patch_xid)]
        local_patch_yids = [max(center_ypid - 7, 0), min(center_ypid + 6, max_patch_yid)]

        local_grid_xids = [patch_grid_num * local_patch_xids[0], patch_grid_num * (local_patch_xids[1] + 1)]
        local_grid_yids = [patch_grid_num * local_patch_yids[0], patch_grid_num * (local_patch_yids[1] + 1)]
        local_traj_image = traj_image[self.grid_size * local_grid_xids[0] : self.grid_size * (local_grid_xids[1]), \
                                        self.grid_size * local_grid_yids[0] : self.grid_size * (local_grid_yids[1])]
        return self.pad_local_image(local_traj_image), local_patch_xids, local_patch_yids
    
    def vision_augmented_learner(self, input_grid, mask):
        '''
        input_grid: (node_num, inp_seq_len, 2)
        mask: (node_num, )
        '''
        trajectory_images, local_patch_xids, local_patch_yids = [], [], []
    
        self.map_width = self.map_image.shape[1]
        self.map_height = self.map_image.shape[0]
        # create target-agent trajectory image
        target_trans = self.st_trans(input_grid[0])
        target_image = self.trans_to_image(target_trans, self.map_image.copy(), self.grid_size)
        max_patch_xid = int(self.map_height / self.patch_size) - 1
        max_patch_yid = int(self.map_width / self.patch_size) - 1
        local_target_image, local_patch_xid, local_patch_yid = self.create_local_trajectory(input_grid[0], target_image, max_patch_xid, max_patch_yid, int(self.patch_size / self.grid_size))
        trajectory_images.append(local_target_image)
        local_patch_xids.append(local_patch_xid)
        local_patch_yids.append(local_patch_yid)
        
        # create neighbor-agent trajectory image
        for i in range(1, input_grid.shape[0]):
            if mask[i] == 0:
                local_neighbor_image = np.full((self.local_image_size, self.local_image_size, 3), 0, dtype = np.uint8)
                local_patch_xid = [-1, -1]
                local_patch_yid = [-1, -1]
            else:
                neigh_trans = self.st_trans(input_grid[i])
                neighbor_image = self.trans_to_image(neigh_trans, self.map_image.copy(), self.grid_size)
                local_neighbor_image, local_patch_xid, local_patch_yid = self.create_local_trajectory(input_grid[i], neighbor_image, max_patch_xid, max_patch_yid, int(self.patch_size / self.grid_size))
            trajectory_images.append(local_neighbor_image)
            local_patch_xids.append(local_patch_xid)
            local_patch_yids.append(local_patch_yid)
        # from PIL import Image
        # image1 = Image.fromarray(target_rimage, 'RGB')
        # image1.save(f'test1.png')
        trajectory_images = np.stack(trajectory_images, axis=0)
        local_patch_xids = np.array(local_patch_xids)
        local_patch_yids = np.array(local_patch_yids)
        return trajectory_images, local_patch_xids, local_patch_yids
    
    def target_vision_augmented_learner(self, input_grid):
        '''
        input_grid: (inp_seq_len, 2)
        '''
        # create target-agent label trajectory image
        target_trans = self.st_trans(input_grid)
        target_image = self.trans_to_image(target_trans, self.map_image.copy(), self.grid_size)
        max_patch_xid = int(self.map_height / self.patch_size) - 1
        max_patch_yid = int(self.map_width / self.patch_size) - 1
        local_target_image, _, _ = self.create_local_trajectory(input_grid, target_image, max_patch_xid, max_patch_yid, int(self.patch_size / self.grid_size))
        return local_target_image

    def __getitem__(self, idx):
        """
        sample_input.shape (node_num, inp_seq_len, node_feats)
        sample_mask.shape (node_num, inp_seq_len)
        sample_target.shape (tar_seq_len, node_feats)
        sample_node_dis.shape (inp_seq_len, node_num-1, 4)
        """
        #  Model inputs
        sample_input = self.input_data[idx]
        sample_mask = self.mask[idx]
        sample_neighbor_dis = self.neighbor_dis[idx]

        #  Model targets
        sample_target = self.target_data[idx]
    
        # create input image
        sample_input_grid = sample_input[:,:,2:4]
        trajectory_images, local_patch_xids, local_patch_yids = self.vision_augmented_learner(sample_input_grid, sample_mask[:,-1])
        # create label image
        sample_target_grid = sample_target[:,2:4]
        label_trajectory_images = self.target_vision_augmented_learner(sample_target_grid)

        return sample_input, sample_neighbor_dis, sample_mask, sample_target, \
                trajectory_images, local_patch_xids, local_patch_yids, self.map_image, label_trajectory_images


def collate_session_based(batch):
    '''
    batch_x_data.shape (batch_size, node_num, inp_seq_len, feat_num)
    batch_neighbor_dis.shape (batch_size, inp_seq_len, node_num, 4)
    batch_mask.shape (batch_size, node_num, inp_seq_len)
    batch_target.shape (batch_size,tar_seq_len, node_feats)
    batch_trajectory_images.shape (batch_size, node_num, image_size, image_size, 3)
    batch_local_patch_xids.shape (batch_size, node_num, 2)
    batch_local_patch_yids.shape (batch_size, node_num, 2)
    batch_map_image.shape (1, map_height, map_width, 3)
    batch_label_trajectory_images.shape (batch_size, image_size, image_size)
    '''
    batch_x_data = np.array([sample[0] for sample in batch])
    batch_neighbor_dis = np.array([sample[1] for sample in batch])
    batch_mask = np.array([sample[2] for sample in batch])
    batch_target = np.array([sample[3] for sample in batch])
    batch_trajectory_images = np.array([sample[4] for sample in batch])
    batch_local_patch_xids = np.array([sample[5] for sample in batch])
    batch_local_patch_yids = np.array([sample[6] for sample in batch])
    batch_map_image = np.array([sample[7] for sample in batch])[0]
    batch_label_trajectory_images = np.array([sample[8] for sample in batch])

    return sample_data(batch_x_data, batch_neighbor_dis, batch_mask, batch_target, batch_trajectory_images, \
                       batch_local_patch_xids, batch_local_patch_yids, batch_map_image, batch_label_trajectory_images)


class sample_data():
    def __init__(self, batch_x_data, batch_neighbor_dis, batch_mask, batch_target, batch_trajectory_images, \
                 batch_local_patch_xids, batch_local_patch_yids, batch_map_image, batch_label_trajectory_images):
        self.x_data = torch.from_numpy(batch_x_data).to(torch.float32)
        self.x_neighbor_dis = torch.from_numpy(batch_neighbor_dis).to(torch.float32)
        self.mask = torch.from_numpy(batch_mask).to(torch.float32)
        self.target = torch.from_numpy(batch_target).to(torch.float32)
        self.trajectory_images = torch.from_numpy(batch_trajectory_images).to(torch.float32)
        self.local_patch_xids = torch.from_numpy(batch_local_patch_xids).to(torch.long)
        self.local_patch_yids = torch.from_numpy(batch_local_patch_yids).to(torch.long)
        self.map_image = torch.from_numpy(batch_map_image).unsqueeze(0).to(torch.float32)
        self.target_label_iamge = torch.from_numpy(batch_label_trajectory_images).to(torch.float32)
        # self.texture = torch.from_numpy(batch_texture).to(torch.float32)
        # self.hue_vec = torch.from_numpy(batch_hue_vec).to(torch.float32)
        # self.brightness = torch.from_numpy(batch_brightness).to(torch.float32)


