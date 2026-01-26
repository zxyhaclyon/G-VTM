import numpy as np
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from utils import *
import random
import string
import warnings
warnings.filterwarnings("ignore")
random_str = lambda : ''.join(random.sample(string.ascii_letters + string.digits, 6))

# 设置随机种子
# randomSeed = 2046
# torch.manual_seed(randomSeed)
# torch.cuda.manual_seed(randomSeed)
# torch.cuda.manual_seed_all(randomSeed) 
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(randomSeed)

#-------------------------------------------------------------------------------------------------
def TrainTrajEpoch(device, loader, vision_traj_model, optim):
    vision_traj_model.train()   
    
    loss_item = 0
    count = 0
    for train_data in loader:
        x_data = train_data.x_data.to(device)
        x_neighbor_dis = train_data.x_neighbor_dis.to(device)
        x_mask = train_data.mask.to(device)
        target = train_data.target.to(device)
        map_image = train_data.map_image.to(device)
        trajectory_images = train_data.trajectory_images.to(device)
        local_patch_xids = train_data.local_patch_xids.to(device)
        local_patch_yids = train_data.local_patch_yids.to(device)
        target_label_iamge = train_data.target_label_iamge.to(device)

        # predict future trajectory
        loss, pred, _ = vision_traj_model(x_data, x_neighbor_dis, x_mask, target, map_image, trajectory_images, local_patch_xids, local_patch_yids, target_label_iamge)
        loss_item += loss.item()
        count += 1

        optim.zero_grad()
        loss.backward()
        optim.step()
  
    # 计算每个batch的平均损失
    loss_item /= count
    return loss_item

def ValidTrajEpoch(device, loader, vision_traj_model):
    with torch.no_grad():
        vision_traj_model.eval()
        
        loss_item = 0
        count = 0
        for val_data in loader:
            x_data = val_data.x_data.to(device)
            x_neighbor_dis = val_data.x_neighbor_dis.to(device)
            x_mask = val_data.mask.to(device)
            target = val_data.target.to(device)
            map_image = val_data.map_image.to(device)
            trajectory_images = val_data.trajectory_images.to(device)
            local_patch_xids = val_data.local_patch_xids.to(device)
            local_patch_yids = val_data.local_patch_yids.to(device)
            target_label_iamge = val_data.target_label_iamge.to(device)
        
            # predict future trajectory
            loss, pred, _ = vision_traj_model(x_data, x_neighbor_dis, x_mask, target, map_image, trajectory_images, local_patch_xids, local_patch_yids, target_label_iamge)
            loss_item += loss.item()
            count += 1
        # 计算每个batch的平均损失
        loss_item /= count
        return loss_item

def TestTrajEpoch(device, params_path, loader, vision_traj_model, epoch, save):
    with torch.no_grad():
        vision_traj_model.eval()
        targets = []
        pred_ade_trajs, pred_fde_trajs = [], []
        best_pis = []

        for test_data in loader:
            x_data = test_data.x_data.to(device)
            x_neighbor_dis = test_data.x_neighbor_dis.to(device)
            x_mask = test_data.mask.to(device)
            target = test_data.target.to(device)
            map_image = test_data.map_image.to(device)
            trajectory_images = test_data.trajectory_images.to(device)
            local_patch_xids = test_data.local_patch_xids.to(device)
            local_patch_yids = test_data.local_patch_yids.to(device)
            target_label_iamge = test_data.target_label_iamge.to(device)

            # predict future trajectory
            loss, pred, best_pi = vision_traj_model(x_data, x_neighbor_dis, x_mask, target, map_image, trajectory_images, local_patch_xids, local_patch_yids, target_label_iamge)
            pred1 = pred[0] + x_data[:,0,-1,:2].unsqueeze(1)
            pred2 = pred[1] + x_data[:,0,-1,:2].unsqueeze(1)

            pred_ade_trajs.append(list(pred1.detach().cpu().numpy()))
            pred_fde_trajs.append(list(pred2.detach().cpu().numpy()))
            best_pis.append(list(best_pi.detach().cpu().numpy()))
            targets.append(list(target[:,:,:2].detach().cpu().numpy()))

        targets = np.concatenate(targets, axis = 0) # (batch_num * bs, 16, 2)
        pred_ade_trajs = np.concatenate(pred_ade_trajs, axis = 0)
        pred_fde_trajs = np.concatenate(pred_fde_trajs, axis = 0)
        best_pis = np.concatenate(best_pis, axis = 0)
 
    ade, fde, adpe, mr, nll = calculate_test_indicator(targets, pred_ade_trajs, pred_fde_trajs, best_pis)

    if save:
        check_dir(params_path,mkdir=True)
        np.savez(os.path.join(params_path,f'test_traj_epoch{epoch}.npz'), targets = targets, pred_ade_trajs = pred_ade_trajs, pred_fde_trajs = pred_fde_trajs) 

    return ade, fde, adpe, mr, nll

def calculate_test_indicator(targets, pred_ade_trajs, pred_fde_trajs, best_pis):

    B, T, _ = targets.shape
    fde_targets = targets[:,-1,:] # (batch_num * bs, 2)
    fde_predicts = pred_fde_trajs[:,-1,:]
    # calculate ADE
    ade_distances = np.linalg.norm(pred_ade_trajs - targets, axis=-1)
    ade_per_sample = np.mean(ade_distances, axis=-1)
    final_ade = np.mean(ade_per_sample)
    # calculate FDE
    fde_distances = np.linalg.norm(fde_predicts - fde_targets, axis=-1)
    final_fde = np.mean(fde_distances)
    # calculate NLL
    nll = -np.log(best_pis).mean()
    
    # calculate APDE
    apde_sum = 0
    for k in range(T):
        # seek the nearest point in the true trajectory
        distances = np.linalg.norm(targets - pred_ade_trajs[:, k:k+1], axis=-1)
        k_star = np.argmin(distances, axis=1)
        x_k_star = targets[np.arange(B), k_star]
        apde_sum += np.mean(np.linalg.norm(pred_ade_trajs[:, k] - x_k_star, axis=-1))

    apde = apde_sum / T

    # calculate MR
    final_true = targets[:, -1]
    final_pred = pred_fde_trajs[:, -1]
    distances = np.linalg.norm(final_true - final_pred, axis=-1)
    mr = np.mean(distances > 2.0)

    return round(final_ade, 3), round(final_fde, 3), round(apde, 3), round(mr, 3), round(nll, 3)


    



    
    