import argparse
import os
import random
import numpy as np
import torch
from data import GeneData, get_data_file_path, prepare_fold_data
from GNAS_core.model.inference_test import inference_scratch_train


def config():
    parser = argparse.ArgumentParser("AutoGRN.")
    parser.add_argument("--data_name", default="mESC_1", type=str, help="The dataset name",
                        choices=['bonemarrow', 'mESC_1', 'mESC_2', 'mHSC_E', 'mHSC_GM', 'mHSC_L'])
    parser.add_argument("--device", default="0", type=int, help="Running device. E.g `--device 0`, if using cpu, set `--device -1`")
    parser.add_argument("--seed", default=114514, type=int)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--epochs_scratch", type=int, default=200)

    args = parser.parse_args()
    args.device = (torch.device(args.device) if args.device >= 0 else torch.device("cpu"))
    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_test_architecture(data_name):
    if data_name == 'bonemarrow':
        target_architecture = ['GATv2Conv', 'BiGCNConv', 'leaky_relu', 512, 'concat']
    elif data_name == 'mESC_1':
        target_architecture = ['SAGEConv', 'BiNoneConv', 'relu', 256, 'concat']
    elif data_name == 'mESC_2':
        target_architecture = ['TAGConv', 'BiGraphConv', 'leaky_relu', 256, 'sum']
    elif data_name == 'mHSC_E':
        target_architecture = ['SAGEConv', 'BiSAGEConv', 'leaky_relu', 256, 'abs_difference']
    elif data_name == 'mHSC_GM':
        target_architecture = ['GATv2Conv', 'BiSAGEConv', 'relu', 1024, 'max']
    elif data_name == 'mHSC_L':
        target_architecture = ['SAGEConv', 'BiSAGEConv', 'leaky_relu', 256, 'concat']
    else:
        raise Exception('Wrong dataset name!')
    return target_architecture


if __name__ == "__main__":
    args = config()


    set_seed(args.seed)

    args.store_path = './data_evaluation'
    args = get_data_file_path(args)

    data_e = GeneData(args.rpkm_path,
                 args.label_path,
                 args.divide_path,
                 TF_num=args.TF_num,
                 gene_emb_path=args.gene_emb_path,
                 cell_emb_path=args.cell_emb_path,
                 istime=args.is_time, gene_list_path=args.gene_list_path,
                 data_name=args.data_name, TF_random=args.TF_random, ish5=args.is_h5)

    data_e = prepare_fold_data(data_e, args)

    args.learning_type = 'inference'
    args.data_save_name = args.data_name + '_' + args.learning_type

    target_architecture = get_test_architecture(args.data_name)
    print(35 * "=" + " the testing start " + 35 * "=")
    print("dataset:", args.data_name)
    print("test gnn architecture", target_architecture)

    ## train from scratch
    val_auc, val_ap, test_auc, test_ap = inference_scratch_train(target_architecture, data_e=data_e, args=args)
    # print('Cross-Validation, Val AUC: {:.4f}, Val AP:{:.4f}'.format(val_auc, val_ap))
    print('Cross-Validation, Test AUC: {:.4f}, Test AP:{:.4f}'.format(test_auc, test_ap))
    print('=' * 70)
    print(35 * "=" + " the testing ending " + 35 * "=")
