import dgl
import numpy as np
import os
import pandas as pd
import torch
import collections
import math


class GeneData:
    def __init__(self, rpkm_path, label_path, divide_path, TF_num, gene_emb_path, cell_emb_path, istime, ish5=False,
                 gene_list_path=None, data_name=None, TF_random=False, save=False):
        self.gene_cell_src = None
        self.gene_cell_dst = None
        self.istime = istime
        self.data_name = data_name
        self.TF_random = TF_random
        self.save = save

        if not istime:
            if not ish5:
                self.df = pd.read_csv(rpkm_path, header='infer', index_col=0)
            else:
                self.df = pd.read_hdf(rpkm_path, key='/RPKMs').T

        else:
            time_h5 = []
            files = os.listdir(rpkm_path)
            for i in range(len(files)):
                if self.data_name.lower() == 'mesc1':
                    time_pd = pd.read_hdf(rpkm_path + 'RPM_' + str(i) + '.h5', key='/RPKM')
                else:
                    time_pd = pd.read_hdf(rpkm_path + 'RPKM_' + str(i) + '.h5', key='/RPKMs')
                time_h5.append(time_pd)
            train_data = pd.concat(time_h5, axis=0, ignore_index=True)
            self.df = train_data.T
        self.origin_data = self.df.values

        self.df.columns = self.df.columns.astype(str)
        self.df.index = self.df.index.astype(str)
        self.df.columns = self.df.columns.str.upper()
        self.df.index = self.df.index.str.upper()
        self.cell_to_idx = dict(zip(self.df.columns.astype(str), range(len(self.df.columns))))
        self.gene_to_idx = dict(zip(self.df.index.astype(str), range(len(self.df.index))))

        self.gene_to_name = {}
        if gene_list_path:
            gene_list = pd.read_csv(gene_list_path, header=None, sep='\s+')
            gene_list[0] = gene_list[0].astype(str)
            gene_list[1] = gene_list[1].astype(str)
            gene_list[0] = gene_list[0].str.upper()
            gene_list[1] = gene_list[1].str.upper()
            self.gene_to_name = dict(zip(gene_list[0].astype(str), gene_list[1].astype(str)))

        self.start_index = []
        self.end_index = []
        self.gene_emb = np.load(gene_emb_path)
        self.cell_emb = np.load(cell_emb_path)

        self.all_emb = np.concatenate((self.gene_emb, self.cell_emb), axis=0)

        self.key_list = []
        self.gold_standard = {}
        self.datas = []
        self.gene_key_datas = []

        self.cell_datas = []
        self.labels = []
        self.idx = []

        self.geneHaveCell = collections.defaultdict(list)
        self.node_src = []
        self.node_dst = []

        self.getStartEndIndex(divide_path)
        self.getLabel(label_path)
        self.getGeneCell(self.df)

        self.getTrainTest(TF_num)

        self.graph = self._generate_graph(self.gene_to_name, self.gene_to_idx, self.gene_key_datas, self.gene_emb)
        self.bipartite_graph = self._generate_bipartite_graph(self.geneHaveCell, self.gene_emb, self.cell_emb)

    def getStartEndIndex(self, divide_path):
        tmp = []
        with open(divide_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                tmp.append(int(line[0]))
        self.start_index = tmp[:-1]
        self.end_index = tmp[1:]

    def getLabel(self, label_path):
        s = open(label_path, 'r')
        for line in s:
            line = line.split()
            gene1 = line[0]
            gene2 = line[1]
            label = line[2]

            key = str(gene1) + "," + str(gene2)
            if key not in self.gold_standard.keys():
                self.gold_standard[key] = int(label)
                self.key_list.append(key)
            else:
                if label == 2:
                    # print(label)
                    pass
                else:
                    self.gold_standard[key] = int(label)
                self.key_list.append(key)
        s.close()
        print(len(self.key_list))
        print(len(self.gold_standard.keys()))
        # exit()

    def getTrainTest(self, TF_num):

        TF_order = list(range(0, len(self.start_index)))
        if self.TF_random:
            np.random.seed(42)
            np.random.shuffle(TF_order)
        print("TF_order", TF_order)
        index_start_list = np.asarray(self.start_index)
        index_end_list = np.asarray(self.end_index)
        index_start_list = index_start_list[TF_order]
        index_end_list = index_end_list[TF_order]

        print(index_start_list)
        print(index_end_list)
        # s = open(self.data_name + '_representation/gene_pairs.txt', 'w')
        # ss = open(self.data_name + '_representation/divide_pos.txt', 'w')
        pos_len = 0
        # ss.write(str(0) + '\n')
        for i in range(TF_num):
            name = self.data_name + '_representation/'
            # if os.path.exists(self.data_name + '_representation/' + str(i) + '_xdata.npy'):
            if self.save:
                x_data = np.load(name + str(i) + '_xdata.npy')
                h_data = np.load(name + str(i) + '_hdata.npy')
                y_data = np.load(name + str(i) + '_ydata.npy')
                gene_key_data = np.load(name + str(i) + '_gene_key_data.npy')

                self.datas.append(x_data)
                self.labels.append(y_data)
                self.gene_key_datas.append(gene_key_data)
                self.h_datas.append(h_data)
                continue

            start_idx = index_start_list[i]
            end_idx = index_end_list[i]

            print(i)
            print(start_idx, end_idx)

            this_datas = []
            this_key_datas = []
            this_labels = []

            for line in self.key_list[start_idx:end_idx]:

                label = self.gold_standard[line]
                gene1, gene2 = line.split(',')
                gene1 = gene1.upper()
                gene2 = gene2.upper()
                if int(label) != 2:
                    this_key_datas.append([gene1.lower(), gene2.lower(), label])
                    # s.write(gene1 + '\t' + gene2 + '\t' + str(label) + '\n')
                    if not self.gene_to_name:
                        gene1_idx = self.gene_to_idx[gene1]
                        gene2_idx = self.gene_to_idx[gene2]
                    else:
                        gene1_index = self.gene_to_name[gene1]
                        gene2_index = self.gene_to_name[gene2]
                        gene1_idx = self.gene_to_idx[gene1_index]
                        gene2_idx = self.gene_to_idx[gene2_index]

                    gene1_emb = self.gene_emb[gene1_idx]
                    gene2_emb = self.gene_emb[gene2_idx]

                    gene1_emb = np.expand_dims(gene1_emb, axis=0)
                    gene2_emb = np.expand_dims(gene2_emb, axis=0)

                    gene1_cells = self.geneHaveCell[gene1_idx]
                    if len(gene1_cells) == 0:
                        gene1_cells_emb = np.zeros(256)
                    else:
                        gene1_cells_emb = self.cell_emb[gene1_cells]
                        gene1_cells_emb = np.mean(gene1_cells_emb, axis=0)

                    gene2_cells = self.geneHaveCell[gene2_idx]
                    if len(gene2_cells) == 0:
                        gene2_cells_emb = np.zeros(256)
                    else:
                        gene2_cells_emb = self.cell_emb[gene2_cells]
                        gene2_cells_emb = np.mean(gene2_cells_emb, axis=0)

                    gene1_cells_emb = np.expand_dims(gene1_cells_emb, axis=0)
                    gene2_cells_emb = np.expand_dims(gene2_cells_emb, axis=0)

                    gene_emb = np.concatenate((gene1_emb, gene2_emb, gene1_cells_emb, gene2_cells_emb), axis=0)
                    this_datas.append(gene_emb)

                    this_labels.append(label)
            pos_len += len(this_datas)
            # ss.write(str(pos_len) + '\n')

            this_datas = np.asarray(this_datas)

            print(this_datas.shape)

            this_labels = np.asarray(this_labels)
            this_key_datas = np.asarray(this_key_datas)
            if self.save:
                if not os.path.exists(name):
                    os.mkdir(name)
                np.save(name + str(i) + '_xdata.npy', this_datas)
                np.save(name + str(i) + '_ydata.npy', this_labels)
                np.save(name + str(i) + '_gene_key_data.npy', this_key_datas)
                print(this_datas.shape, this_labels.shape)

            self.datas.append(this_datas)
            self.labels.append(this_labels)
            self.gene_key_datas.append(this_key_datas)
        # s.close()
        # ss.close()

    def getGeneCell(self, df):
        for i in range(df.shape[0]):
            j_nonzero = np.nonzero(df.iloc[i, :].values)[0]
            if len(j_nonzero) == 0:
                continue
            self.geneHaveCell[i].extend(j_nonzero)

    def _generate_graph(self, gene_to_name, gene_to_idx, gene_key_datas, gene_emb):
        row = []
        col = []

        for tf in gene_key_datas:
            for edge in tf:
                gene1_index = gene_to_name[edge[0].upper()]
                gene2_index = gene_to_name[edge[1].upper()]
                gene1_idx = gene_to_idx[gene1_index]
                gene2_idx = gene_to_idx[gene2_index]
                row.append(gene1_idx)
                col.append(gene2_idx)

        src = torch.tensor(row)
        dst = torch.tensor(col)
        g = dgl.graph((src, dst), num_nodes=gene_emb.shape[0])

        g.ndata['feat'] = torch.tensor(gene_emb)
        return g

    def _generate_bipartite_graph(self, geneHaveCell, gene_emb, cell_emb):
        gene_num = gene_emb.shape[0]
        cell_num = cell_emb.shape[0]

        rating_pairs = [(i, j) for i in range(gene_num) for j in geneHaveCell[i]]
        src_gene = torch.tensor(rating_pairs)[:,0]
        dst_cell = torch.tensor(rating_pairs)[:,1]

        data_dict = {
            ('gene', 'have', 'cell'): (src_gene, dst_cell),
            ('cell', 'have', 'gene'): (dst_cell, src_gene),
        }

        bipartite_graph = dgl.heterograph(data_dict,
                          num_nodes_dict={"gene": gene_num, "cell": cell_num})

        bipartite_graph.nodes['gene'].data['feat'] = torch.tensor(gene_emb)
        bipartite_graph.nodes['cell'].data['feat'] = torch.tensor(cell_emb)
        return bipartite_graph



def get_data_file_path(args):
    
    if args.data_name == 'bonemarrow':
        args.rpkm_path = args.store_path+'/bonemarrow/bone_marrow_cell.h5'
        args.label_path = args.store_path+'/bonemarrow/gold_standard_for_TFdivide'
        args.divide_path = args.store_path+'/bonemarrow/whole_gold_split_pos'
        args.gene_list_path = args.store_path+'/bonemarrow/sc_gene_list.txt'
        args.TF_num = 13
        args.is_h5 = True
        args.TF_random = False
        args.is_time = False

    elif args.data_name == 'mESC_1':
        args.rpkm_path = args.store_path+'/mesc/mesc_cell.h5'
        args.label_path = args.store_path+'/mesc/gold_standard_mesc_whole.txt'
        args.divide_path = args.store_path+'/mesc/mesc_divideTF_pos.txt'
        args.gene_list_path = args.store_path+'/mesc/mesc_sc_gene_list.txt'
        args.TF_num = 38
        args.is_h5 = True
        args.TF_random = False
        args.is_time = False

    elif args.data_name == 'mESC_2':
        args.rpkm_path = args.store_path+'/single_cell_type/mESC/ExpressionData.csv'
        args.label_path = args.store_path+'/single_cell_type/training_pairsmESC.txt'
        args.divide_path = args.store_path+'/single_cell_type/training_pairsmESC.txtTF_divide_pos.txt'
        args.gene_list_path = args.store_path+'/single_cell_type/mESC_geneName_map.txt'
        args.TF_num = 18
        args.is_h5 = False
        args.TF_random = True
        args.is_time = False
    elif args.data_name == 'mHSC_E':
        args.rpkm_path = args.store_path+'/single_cell_type/mHSC-E/ExpressionData.csv'
        args.label_path = args.store_path+'/single_cell_type/training_pairsmHSC_E.txt'
        args.divide_path = args.store_path+'/single_cell_type/training_pairsmHSC_E.txtTF_divide_pos.txt'
        args.gene_list_path = args.store_path+'/single_cell_type/mHSC_E_geneName_map.txt'
        args.TF_num = 18
        args.is_h5 = False
        args.TF_random = True
        args.is_time = False
    elif args.data_name == 'mHSC_GM':
        args.rpkm_path = args.store_path+'/single_cell_type/mHSC-GM/ExpressionData.csv'
        args.label_path = args.store_path+'/single_cell_type/training_pairsmHSC_GM.txt'
        args.divide_path = args.store_path+'/single_cell_type/training_pairsmHSC_GM.txtTF_divide_pos.txt'
        args.gene_list_path = args.store_path+'/single_cell_type/mHSC_GM_geneName_map.txt'
        args.TF_num = 18
        args.is_h5 = False
        args.TF_random = True
        args.is_time = False
    elif args.data_name == 'mHSC_L':
        args.rpkm_path = args.store_path+'/single_cell_type/mHSC-L/ExpressionData.csv'
        args.label_path = args.store_path+'/single_cell_type/training_pairsmHSC_L.txt'
        args.divide_path = args.store_path+'/single_cell_type/training_pairsmHSC_L.txtTF_divide_pos.txt'
        args.gene_list_path = args.store_path+'/single_cell_type/mHSC_L_geneName_map.txt'
        args.TF_num = 18
        args.is_h5 = False
        args.TF_random = True
        args.is_time = False

    else:
        raise Exception('Wrong dataset name!')

    args.gene_emb_path = args.store_path + '/embeddings/' + args.data_name + '/gene_embedding.npy'
    args.cell_emb_path = args.store_path + '/embeddings/' + args.data_name + '/cell_embedding.npy'
    return args


def prepare_fold_data(data_e, args):
    TF_num = args.TF_num
    data_name = args.data_name

    cross_file_path = args.store_path+'/Time_data/DB_pairs_TF_gene/' + data_name.lower() + '_cross_validation_fold_divide.txt'
    cross_index = []
    if args.is_time:
        with open(cross_file_path, 'r') as f:
            for line in f:
                cross_index.append([int(i) for i in line.strip().split(',')])

    data_e.graph = data_e.graph.to(args.device)
    data_e.bipartite_graph = data_e.bipartite_graph.to(args.device)

    # three-fold cross validation
    z_all = []
    train_samples_all = []
    train_labels_all = []
    val_samples_all = []
    val_labels_all = []
    test_samples_all = []
    test_labels_all = []
    for fold in range(1, 4):
        print('=' * 50)
        print('Fold:', fold)

        test_TF = [i for i in range(int(np.ceil((fold - 1) * 0.333333 * TF_num)),
                                    int(np.ceil(fold * 0.333333 * TF_num)))]
        if args.is_time:
            test_TF = cross_index[fold - 1]

        train_TF = [j for j in range(TF_num) if j not in test_TF]
        print("test_TF:", test_TF)

        labels = []
        for tf in data_e.labels:
            labels += tf.tolist()
        labels = torch.tensor(labels)

        train_index = torch.zeros_like(labels, dtype=torch.bool)
        val_index = torch.zeros_like(labels, dtype=torch.bool)
        test_index = torch.zeros_like(labels, dtype=torch.bool)

        for j in train_TF:
            start_sample = sum([len(tf) for tf in data_e.labels[:j]])
            end_sample = start_sample + len(data_e.labels[j])
            train_index[start_sample:end_sample] = True

        true_indices = torch.nonzero(train_index, as_tuple=True)[0]
        train_num = len(true_indices)
        torch.manual_seed(42)
        val_idx_tmp = true_indices[torch.randperm(train_num)[:math.ceil(0.2 * train_num)]]
        val_index[val_idx_tmp] = True
        train_index[val_idx_tmp] = False

        z = [0]
        z_len = 0
        for j in test_TF:
            start_sample = sum([len(tf) for tf in data_e.labels[:j]])
            end_sample = start_sample + len(data_e.labels[j])
            test_index[start_sample:end_sample] = True
            z_len += len(data_e.datas[j])
            z.append(z_len)
        z_all.append(z)

        print('train:', len(torch.nonzero(train_index, as_tuple=True)[0]))
        print('val:', len(torch.nonzero(val_index, as_tuple=True)[0]))
        print('test:', len(torch.nonzero(test_index, as_tuple=True)[0]))

        all_samples_name = []
        for tf in data_e.gene_key_datas:
            all_samples_name += tf.tolist()

        all_samples_idx = [[], []]
        for sample in all_samples_name:
            src = sample[0].upper()
            dst = sample[1].upper()
            src_index = data_e.gene_to_name[src]
            dst_index = data_e.gene_to_name[dst]
            src_idx = data_e.gene_to_idx[src_index]
            dst_idx = data_e.gene_to_idx[dst_index]
            all_samples_idx[0].append(src_idx)
            all_samples_idx[1].append(dst_idx)

        all_samples_idx = torch.tensor(all_samples_idx).T
        train_samples = all_samples_idx[train_index]
        train_labels = labels[train_index].reshape(-1, 1).to(args.device).float()
        val_samples = all_samples_idx[val_index]
        val_labels = labels[val_index].reshape(-1, 1).to(args.device).float()
        test_samples = all_samples_idx[test_index]
        test_labels = labels[test_index].reshape(-1, 1).to(args.device).float()

        train_samples_all.append(train_samples)
        train_labels_all.append(train_labels)
        val_samples_all.append(val_samples)
        val_labels_all.append(val_labels)
        test_samples_all.append(test_samples)
        test_labels_all.append(test_labels)

    data_e.z_all = z_all
    data_e.train_samples_all = train_samples_all
    data_e.train_labels_all = train_labels_all
    data_e.val_samples_all = val_samples_all
    data_e.val_labels_all = val_labels_all
    data_e.test_samples_all = test_samples_all
    data_e.test_labels_all = test_labels_all

    return data_e

