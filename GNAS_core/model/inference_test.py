import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_recall_curve, auc
from GNAS_core.model.gnn_model import GNN_Model


def inference_scratch_train(gnn_architecture, data_e, args):
    pre_all_val = []
    label_all_val = []
    pre_all = []
    label_all = []
    y_test_predict = []
    y_test_true = []

    epochs = args.epochs_scratch

    for fold in range(1, 4):
        print('=' * 50)
        print('Fold:', fold)

        graph = data_e.graph
        bipartite_graph = data_e.bipartite_graph
        train_samples = data_e.train_samples_all[fold - 1]
        train_labels = data_e.train_labels_all[fold - 1]
        val_samples = data_e.val_samples_all[fold - 1]
        val_labels = data_e.val_labels_all[fold - 1]
        test_samples = data_e.test_samples_all[fold - 1]
        test_labels = data_e.test_labels_all[fold - 1]

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

            print('Epoch: {}, Train Loss: {:.4f},Train Acc: {:.4f}, '
                  'Val Loss: {:.4f}, Val Acc: {:.4f}, Val AUC: {:.4f}, Val AP: {:.4f}'.format(
                epoch, train_loss, train_acc, val_loss, val_acc, val_auc, val_ap))
            if early_stop > stop_num:
                break

        # val
        val_acc, val_auc, val_ap, val_loss, val_preds = eval_dataset(gnn_model, criterion, graph, bipartite_graph, val_samples, val_labels)
        pre_all_val.extend(val_preds.cpu().numpy())
        label_all_val.extend(val_labels.cpu().numpy())

        # test
        test_acc, test_auc, test_ap, test_loss, test_preds = eval_dataset(gnn_model, criterion, graph, bipartite_graph, test_samples, test_labels)
        pre_all.extend(test_preds.cpu().numpy())
        label_all.extend(test_labels.cpu().numpy())
        y_test_predict.append(test_preds.cpu().numpy())
        y_test_true.append(test_labels.cpu().numpy())

    # val results
    final_auc_val = roc_auc_score(label_all_val, pre_all_val)
    precision, recall, thresholds = precision_recall_curve(label_all_val, pre_all_val, pos_label=1)
    final_ap_val = auc(recall, precision)
    pre_all_val = np.where(np.asarray(pre_all_val) > 0.5, 1, 0)
    final_acc_val = accuracy_score(label_all_val, pre_all_val)

    # test results
    final_auc = roc_auc_score(label_all, pre_all)
    precision, recall, thresholds = precision_recall_curve(label_all, pre_all, pos_label=1)
    final_ap = auc(recall, precision)
    pre_all = np.where(np.asarray(pre_all) > 0.5, 1, 0)
    final_acc = accuracy_score(label_all, pre_all)

    z_all = data_e.z_all
    for fold in range(len(y_test_predict)):
        print('=' * 50)
        test_predict = y_test_predict[fold]
        test_true = y_test_true[fold]
        z = z_all[fold]
        for i in range(len(z) - 1):
            test_predict_i = test_predict[z[i]:z[i + 1]]
            test_true_i = test_true[z[i]:z[i + 1]]
            test_auc_i = roc_auc_score(test_true_i, test_predict_i)
            precision, recall, thresholds = precision_recall_curve(test_true_i, test_predict_i, pos_label=1)
            test_ap_i = auc(recall, precision)
            test_predict_i = np.where(np.asarray(test_predict_i) > 0.5, 1, 0)
            test_acc_i = accuracy_score(test_true_i, test_predict_i)
            print('\tindex {} TF, Test Acc: {:.4f}, Test AUC: {:.4f}, Test AP: {:.4f}'.format(i, test_acc_i, test_auc_i, test_ap_i))

    return final_auc_val, final_ap_val, final_auc, final_ap


def scratch_train_each_epoch(gnn_model, optimizer, criterion, graph, bipartite_graph, train_samples, train_labels):
    gnn_model.train()
    optimizer.zero_grad()
    train_acc_sum = 0.0

    train_preds = gnn_model(graph, bipartite_graph, train_samples)
    loss = criterion(train_preds, train_labels)
    train_loss = loss.item()

    loss.backward()
    optimizer.step()

    pred = torch.where(train_preds > 0.5, torch.ones_like(train_preds), torch.zeros_like(train_preds))
    train_acc_sum += (pred == train_labels).sum().item()
    train_acc = train_acc_sum / train_preds.shape[0]
    return train_acc, train_loss

def eval_dataset(gnn_model, criterion, graph, bipartite_graph, eval_samples, eval_labels):
    gnn_model.eval()
    eval_acc_sum = 0.0
    with torch.no_grad():
        eval_preds = gnn_model(graph, bipartite_graph, eval_samples)
        loss = criterion(eval_preds, eval_labels)
        eval_loss = loss.item()

        pred = torch.where(eval_preds > 0.5, torch.ones_like(eval_preds), torch.zeros_like(eval_preds))
        eval_acc_sum += (pred == eval_labels).sum().item()
        eval_acc = eval_acc_sum / eval_preds.shape[0]

    eval_auc = roc_auc_score(eval_labels.cpu().numpy(), eval_preds.cpu().numpy())
    precision, recall, thresholds = precision_recall_curve(eval_labels.cpu().numpy(), eval_preds.cpu().numpy(), pos_label=1)
    eval_ap = auc(recall, precision)
    return eval_acc, eval_auc, eval_ap, eval_loss, eval_preds
