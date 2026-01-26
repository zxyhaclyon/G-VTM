import os
import argparse
import configparser
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from datamodule import LitDataModule
from utils import *
from vision_traj_vlm_modal_pi import Vision_Trajectory_Model
from train_traj_old import *
import copy
from transformers import get_cosine_schedule_with_warmup
import time
import nni
import random
import string
import warnings
warnings.filterwarnings("ignore")
random_str = lambda : ''.join(random.sample(string.ascii_letters + string.digits, 6))

#-------------------------------------------------------------------------------------------------
def Train_traj(args, vision_traj_model, device):

    patience_count = 0

    max_epoch = args.epoch

    lr = args.lr
    val_epoch = args.val_epoch
    test_epoch = args.test_epoch

    optim = torch.optim.AdamW(params=filter(lambda x : x.requires_grad, vision_traj_model.parameters()),lr=lr,weight_decay=args.weight_decay)
    # scheduler = ExponentialLR(optimizer=optim, gamma=args.lr_decay)
    num_training_steps = max_epoch
    num_warmup_steps = int(0.05 * num_training_steps)  # 前10%步数用于warmup

    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_loss, best_ade = 1e9, 1e9
    best_model = copy.deepcopy(vision_traj_model.grad_state_dict())

    for epoch in range(max_epoch):
        st_time = time.time()
        train_loss = TrainTrajEpoch(device, train_loader, vision_traj_model, optim)
        ed_time = time.time()

        print(f"epoch {epoch} train_loss:{train_loss} train_epoch_time:{ed_time - st_time}")

        if epoch % val_epoch == 0:
            st_time = time.time()
            val_loss = ValidTrajEpoch(device, valid_loader, vision_traj_model)
            ed_time = time.time()
            
            if args.nni:
                nni.report_intermediate_result(val_loss)
            print(f"[Validation] epoch {epoch} val_loss:{val_loss} valid_epoch_time:{ed_time - st_time}")

            if val_loss < best_loss :
                patience_count = 0
                best_loss = val_loss
            else :
                patience_count += 1

        if epoch % test_epoch == 0:
            save = True if epoch % 10 == 0 else False
            ade, fde, adpe, mr, nll = TestTrajEpoch(device, params_path, test_loader, vision_traj_model, epoch, False)
            if ade < best_ade:
                best_ade = ade
                best_model = copy.deepcopy(vision_traj_model.grad_state_dict())

            print(f"[Test][prediction] epoch {epoch} ade:{ade} fde:{fde} adpe:{adpe} mr:{mr}")

        if patience_count >= args.patience:
            print('early stop')
            break

        scheduler.step()
        print(f"[Scheduler] epoch {epoch} lr:{optim.param_groups[0]['lr']}")

    # best_model = model.grad_state_dict()
    vision_traj_model.load_state_dict(best_model,strict=False)

    ade, fde, adpe, mr, nll = TestTrajEpoch(device, params_path, test_loader, vision_traj_model, epoch, save=args.save_result)
    if args.nni:
        nni.report_final_result(ade)
    print(f"[Test][prediction] best model ade:{ade} fde:{fde} adpe:{adpe} mr:{mr} nll:{nll}")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./config/VTM_inD_Location1.conf', type=str, help="configuration file path")
    args = parser.parse_args()
    config_file = args.config
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % (args.config.split('/')[-1]))
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Train']
    vision_model_config = config['Vision_Model']
    traj_model_config = config['Trajecotry_Model']


    # Data config
    args.data_root = str(data_config['data_root'])
    args.map_root = str(data_config['map_root'])
    args.dataset = str(data_config['dataset'])
    args.his_len = int(data_config['his_len'])
    args.pred_len = int(data_config['pred_len'])
    args.small_ds = bool(int(data_config['small_ds']))
    args.node_num = int(data_config['node_num'])
    args.embed_scale = float(data_config['embed_scale'])

    # vision model config
    args.vit_attention_probs_dropout_prob = float(vision_model_config['vit_attention_probs_dropout_prob'])
    args.vit_initializer_range = float(vision_model_config['vit_initializer_range'])
    args.vit_hidden_dropout_prob = float(vision_model_config['vit_hidden_dropout_prob'])
    args.vit_hidden_size = int(vision_model_config['vit_hidden_size'])
    args.vit_intermediate_size = int(vision_model_config['vit_intermediate_size'])
    args.vit_max_num_patches = int(vision_model_config['vit_max_num_patches'])
    args.vit_layer_norm_eps = float(vision_model_config['vit_layer_norm_eps'])
    args.vit_num_attention_heads = int(vision_model_config['vit_num_attention_heads'])
    args.vit_num_channels = int(vision_model_config['vit_num_channels'])
    args.vit_num_hidden_layers = int(vision_model_config['vit_num_hidden_layers'])
    args.vit_patch_size = int(vision_model_config['vit_patch_size'])
    args.vit_qkv_bias = bool(int(vision_model_config['vit_qkv_bias']))
    args.vit_origin_patch_num = int(vision_model_config['vit_origin_patch_num'])
    args.map_x_patch_num = int(vision_model_config['map_x_patch_num'])
    args.map_y_patch_num = int(vision_model_config['map_y_patch_num'])
    args.vtm_patch_size = int(vision_model_config['vtm_patch_size'])
    args.patch_embed_size = int(vision_model_config['patch_embed_size'])
    args.grid_size = int(vision_model_config['grid_size'])
    args.in_channels = int(vision_model_config['in_channels'])
    args.local_image_size = int(vision_model_config['local_image_size'])
    args.num_stages = int(vision_model_config['num_stages'])
    args.local_patch_num = int(vision_model_config['local_patch_num'])
    args.sample_query_len = int(vision_model_config['sample_query_len'])
    args.pretrain_vit_path = str(vision_model_config['pretrain_vit_path'])

    # traj model config
    args.x_grid_num = int(traj_model_config['x_grid_num'])
    args.y_grid_num = int(traj_model_config['y_grid_num'])
    args.road_user_types = int(traj_model_config['road_user_types'])
    args.encoder_dim = int(traj_model_config['encoder_dim'])
    args.gat_hid_dim = int(traj_model_config['gat_hid_dim'])
    args.gat_nhead = int(traj_model_config['gat_nhead'])
    args.num_relations = int(traj_model_config['num_relations'])
    args.top_k = int(traj_model_config['top_k'])
    args.num_shared_experts = int(traj_model_config['num_shared_experts'])
    args.num_independent_experts = int(traj_model_config['num_independent_experts'])
    args.vtm_hidden_size = int(traj_model_config['vtm_hidden_size'])
    args.vtm_num_attention_heads = int(traj_model_config['vtm_num_attention_heads'])
    args.vtm_qkv_bias = bool(int(traj_model_config['vtm_qkv_bias']))
    args.vtm_attention_dropout_prob = float(traj_model_config['vtm_attention_dropout_prob'])
    args.vtm_hidden_dropout_prob = float(traj_model_config['vtm_hidden_dropout_prob'])
    args.vtm_intermediate_size = int(traj_model_config['vtm_intermediate_size'])
    args.vtm_layer_norm_eps = float(traj_model_config['vtm_layer_norm_eps'])
    args.vtm_num_hidden_layers = int(traj_model_config['vtm_num_hidden_layers'])
    args.vtm_modality_type_vocab_size = int(traj_model_config['vtm_modality_type_vocab_size'])
    args.traj_dec_hidden_size = int(traj_model_config['traj_dec_hidden_size'])
    args.traj_modal_num = int(traj_model_config['traj_modal_num'])
    
    # train config
    args.lr = float(training_config['lr'])
    args.lr_decay = float(training_config['lr_decay'])
    args.weight_decay = float(training_config['weight_decay'])
    args.batch_size = int(training_config['batch_size'])
    args.epoch = int(training_config['epoch'])
    args.val_epoch = int(training_config['val_epoch'])
    args.test_epoch = int(training_config['test_epoch'])
    args.patience = int(training_config['patience'])
    args.model_root = str(training_config['model_root'])
    args.gpu = int(training_config['gpu'])
    args.nni = bool(int(training_config['nni']))
    if args.nni:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.save_result = int(training_config['save_result'])
    args.train_model = int(training_config['train_model'])
    args.load_model = bool(int(training_config['load_model']))
    args.fine_tune = bool(int(training_config['fine_tune']))
    args.model_root = str(training_config['model_root'])
    args.load_model_path = str(training_config['load_model_path'])
    
    # print(args)
    
    # nni设置
    if args.nni:
        params = nni.get_next_parameter()
        args.vit_num_hidden_layers = int(params['vit_num_hidden_layers'])
        args.vtm_num_hidden_layers = int(params['vtm_num_hidden_layers'])
        args.top_k = int(params['top_k'])
        embed_scale = params['embed_scale']
        args.encoder_dim = int(embed_scale * args.encoder_dim)
        args.gat_hid_dim = int(embed_scale * args.gat_hid_dim)
    else:
        args.encoder_dim = int(args.embed_scale * args.encoder_dim)
        args.gat_hid_dim = int(args.embed_scale * args.gat_hid_dim)

    
    # set save path
    modelpath = ''
    if args.nni:
        params_path = args.model_root
        exp_id = nni.get_experiment_id()
        trail_id = nni.get_trial_id()
        param_path = str(exp_id) + '_' + str(trail_id)
        model_path = os.path.join(params_path, f'{param_path}_model.pth')
    else: 
        time_str = get_time_str()
        params_path = os.path.join(args.model_root,f'{args.dataset}_mode{args.traj_modal_num}')
        print(time_str)
        model_path = os.path.join(params_path,f'model.pth')

    # model setting
    vision_traj_model = Vision_Trajectory_Model(args).to(args.device)
    if args.load_model:      
        vision_traj_model.load(args.load_model_path)
        if not args.fine_tune:
            data_module = LitDataModule(args)
            test_loader = data_module.test_dataloader()
            ade, fde, adpe, mr, nll = TestTrajEpoch(args.device, params_path, test_loader, vision_traj_model, 'zero_shot', save=args.save_result)
            print(f"[Test][prediction] best model ade:{ade} fde:{fde} adpe:{adpe} mr:{mr} nll:{nll}")
        else:
            # load data
            print('start load few shot data')
            data_module = LitDataModule(args)
            train_loader = data_module.train_dataloader()
            valid_loader = data_module.val_dataloader()
            test_loader = data_module.test_dataloader()
            print('End load data')
            Train_traj(args, vision_traj_model, args.device)
    if args.train_model:
        # load data
        print('start load data')
        data_module = LitDataModule(args)
        train_loader = data_module.train_dataloader()
        valid_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        print('End load data')
        # train model
        total_params, total_trainable_params = vision_traj_model.params_num()
        print('start training multi-model')
        print(f'total_params:{total_params}    total_trainable_params:{total_trainable_params}')
        Train_traj(args, vision_traj_model, args.device)
        #保存模型
        check_dir(params_path,mkdir=True)
        vision_traj_model.save(model_path)




    



    
    