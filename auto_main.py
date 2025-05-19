import argparse
import os
import random
import numpy as np
import torch
from data import GeneData, get_data_file_path, prepare_fold_data
from GNAS_core.auto_model import AutoModel


def config():
    parser = argparse.ArgumentParser("AutoGRN.")
    parser.add_argument("--data_name", default="mESC_1", type=str, help="The dataset name",
                        choices=['bonemarrow', 'mESC_1', 'mESC_2', 'mHSC_E', 'mHSC_GM', 'mHSC_L'])
    parser.add_argument("--device", default="0", type=int, help="Running device. E.g `--device 0`, if using cpu, set `--device -1`")
    parser.add_argument("--seed", default=114514, type=int)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--epochs_scratch", type=int, default=200)

    parser.add_argument('--initial_num', type=int, default=100)
    parser.add_argument('--search_epoch', type=int, default=6)
    parser.add_argument('--sharing_num', type=int, default=10)
    parser.add_argument('--train_epoch', type=int, default=200, help='the number of train epoch for sampled model')
    parser.add_argument('--return_top_k', type=int, default=5, help='the number of top model for testing')

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

    AutoModel(data_e, args)
