import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from multi_Transformer import net_multi_Transformer
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer_multi_model import trainer_dataset
import os
from networks.vit_seg_modeling_L2HNet import L2HNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Chesapeake', help='experiment_name')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=10, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--CNN_width', type=int, default=64, help='L2HNet_width_size, default is 64: light mode. Set to 128: normal mode')
parser.add_argument('--savepath', type=str,default="./log_dir_multi_model")
parser.add_argument('--gpu', type=str, default="0,1",help='Select GPU number to train' )
parser.add_argument('--NUM_CHIPS_PER_TILE',type=int,default=500,help='How many chips will be sampled from one large-scale tile')
args = parser.parse_args()

if __name__ == "__main__":
    vit_patches_size=16
    img_size=224
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Chesapeake': {
            'list_dir': './dataset/chy_dataset/chy_dataset_singal.csv',  # The path of the *.csv file
            'num_classes': 5
        }
    }
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    snapshot_path = args.savepath
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg["ViT-B_16"]
    config_vit.n_classes = args.num_classes
    config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

    backbone1 = L2HNet(width=args.CNN_width,image_band=4)
    backbone2 = L2HNet(width=args.CNN_width,image_band=6)
    net = net_multi_Transformer(config_vit, backbone1=backbone1,backbone2=backbone2, img_size=img_size,
                  num_classes=config_vit.n_classes).cuda()
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    trainer_dataset(args, net, snapshot_path)
