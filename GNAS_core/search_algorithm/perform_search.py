import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_recall_curve, auc
from GNAS_core.model.logger import gnn_architecture_performance_save
from GNAS_core.model.gnn_model import GNN_Model
from GNAS_core.model.inference_test import scratch_train_each_epoch, eval_dataset


def estimation(gnn_architecture_list, args, graph_data):
    performance = []

    for gnn_architecture in gnn_architecture_list:
        res = search_train(data_e=graph_data,
                           gnn_architecture=gnn_architecture,
                           args=args)
        performance.append(res)

        gnn_architecture_performance_save(gnn_architecture, res, args.data_save_name)

    return performance


def search_train(data_e, gnn_architecture, args):
    pre_all_val = []
    label_all_val = []

    epochs = args.train_epoch
    for fold in range(1, 4):
        graph = data_e.graph
        bipartite_graph = data_e.bipartite_graph
        train_samples = data_e.train_samples_all[fold - 1]
        train_labels = data_e.train_labels_all[fold - 1]
        val_samples = data_e.val_samples_all[fold - 1]
        val_labels = data_e.val_labels_all[fold - 1]

        gnn_model = GNN_Model(gnn_architecture, in_dim=data_e.gene_emb.shape[1]).to(args.device)
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        val_acc_best = 0.0
        early_stop = 0
        stop_num = 10

        for epoch in range(epochs):
            train_acc, train_loss = scratch_train_each_epoch(gnn_model, optimizer, criterion, graph, bipartite_graph, train_samples, train_labels)

            val_acc, val_auc, val_ap, val_loss, val_preds = eval_dataset(gnn_model, criterion, graph, bipartite_graph, val_samples, val_labels)

            if val_acc > val_acc_best:
                val_acc_best = val_acc
                early_stop = 0
            else:
                early_stop += 1

            if early_stop > stop_num:
                break

        # eval
        val_acc, val_auc, val_ap, val_loss, val_preds = eval_dataset(gnn_model, criterion, graph, bipartite_graph, val_samples, val_labels)

        pre_all_val.extend(val_preds.cpu().numpy())
        label_all_val.extend(val_labels.cpu().numpy())

    final_auc_val = roc_auc_score(label_all_val, pre_all_val)
    precision, recall, thresholds = precision_recall_curve(label_all_val, pre_all_val, pos_label=1)
    final_ap_val = auc(recall, precision)
    pre_all_val = np.where(np.asarray(pre_all_val) > 0.5, 1, 0)
    final_acc_val = accuracy_score(label_all_val, pre_all_val)

    return_result_val = final_auc_val
    return return_result_val
