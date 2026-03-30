import argparse
import multiprocessing
from collections import defaultdict
from operator import index

import numpy as np
from six import iteritems
from sklearn.metrics import (auc, f1_score, precision_recall_curve,confusion_matrix,
                             roc_auc_score,recall_score,accuracy_score,precision_score)
from tqdm import tqdm

from walk import RWGraph

import copy
import random

class Vocab(object):

    def __init__(self, count, index):
        self.count = count
        self.index = index


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='data/amazon',
                        help='Input dataset path')

    parser.add_argument('--features', type=str, default=None,
                        help='Input node features')

    parser.add_argument('--walk-file', type=str, default=None,
                        help='Input random walks')

    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch. Default is 100.')

    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Number of batch_size. Default is 64.')

    parser.add_argument('--eval-type', type=str, default='all',
                        help='The edge type(s) for evaluation.')

    parser.add_argument('--schema', type=str, default=None,
                        help='The metapath schema (e.g., U-I-U,I-U-I).')

    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 200.')

    parser.add_argument('--edge-dim', type=int, default=10,
                        help='Number of edge embedding dimensions. Default is 10.')

    parser.add_argument('--att-dim', type=int, default=20,
                        help='Number of attention dimensions. Default is 20.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 20.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--negative-samples', type=int, default=20,
                        help='Negative samples for optimization. Default is 5.')

    parser.add_argument('--neighbor-samples', type=int, default=10,
                        help='Neighbor samples for aggregation. Default is 10.')

    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience. Default is 5.')

    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of workers for generating random walks. Default is 16.')

    parser.add_argument('--kfold', type=int, default=5,
                        help='Number of kfold. Default is 5.')

    parser.add_argument('--att-head', type=int, default=1,
                    help='Number of attention head. Default is 1.')
    parser.add_argument('--log-store', type=str, default="/mnt/5280d/twang/xyf/runs/",
                        help='log store address.')
    parser.add_argument('--model-dir', type=str, default="/mnt/5280d/twang/xyf/models/",
                        help='model store address.')
    return parser.parse_args()

def get_G_from_edges(edges):
    edge_dict = defaultdict(set)#工厂函数用set初始化后，在字典中查找，如果不存在，返回的是set()，而不是keyError
    for edge in edges:
        u, v = str(edge[0]), str(edge[1])
        edge_dict[u].add(v)
        edge_dict[v].add(u)
    return edge_dict


def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def merge(dict1, dict2):
    print(type(dict1))
    print(type(dict2))
    d = dict1.copy()
    return copy.deepcopy(d.update(dict2))

def load_dict(index,data):
    edge_data = dict()
    edge_data["1"]= list()
    for i in index:
        a = tuple(data[i])
        edge_data["1"].append(a)
    return edge_data

def load_data(f_name):
    print('We are loading data from:', f_name)
    edge_data =[]
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = words[1], words[2]
            a=[x,y]
            edge_data.append(a)
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    print('Total training nodes: ' + str(len(all_nodes)))

    return edge_data

def load_training_data(f_name):
    print('We are loading data from:', f_name)
    edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            if words[0]=="":
                return None
            else:
                if words[0] not in edge_data_by_type:
                    edge_data_by_type[words[0]] = list()
                x, y = words[1], words[2]
                edge_data_by_type[words[0]].append((x, y))
                all_nodes.append(x)
                all_nodes.append(y)

    all_nodes = list(set(all_nodes))#去重
    print('Total training nodes: ' + str(len(all_nodes)))
    return edge_data_by_type


def load_testing_data(f_name):
    print('We are loading data from:', f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = words[1], words[2]
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    return true_edge_data_by_type, false_edge_data_by_type

def load_node_type(f_name):
    print('We are loading node type from:', f_name)
    node_type = {}
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type

def load_feature_data(f_name):
    feature_dic = {}
    with open(f_name, 'r') as f:
        first = True
        for line in f:#跳过第一行不读
            if first:
                first = False
                continue
            items = line.strip().split()
            feature_dic[items[0]] = items[1:]
    return feature_dic

def generate_walks(network_data, num_walks, walk_length, schema, file_name, num_workers):
    if schema is not None:
        node_type = load_node_type(file_name + '/node_type.txt')
    else:
        node_type = None

    all_walks = []
    for layer_id, layer_name in enumerate(network_data):#layer_name即为边类型名，layer_id是从0开始的序号编码
        #分层随机游走奇奇怪怪，要稳定实现这段代码原本的功能得基本每一个节点拥有几乎所有类型的边
        #并不适合目前的任务，可改进
        tmp_data = network_data[layer_name]
        # start to do the random walk on a layer

        layer_walker = RWGraph(get_G_from_edges(tmp_data), node_type, num_workers)
        print('Generating random walks for layer', layer_id)
        layer_walks = layer_walker.simulate_walks(num_walks, walk_length, schema=schema)

        all_walks.append(layer_walks)

    print('Finish generating the walks')

    return all_walks

def generate_pairs(all_walks, vocab, window_size, num_workers):
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        print('Generating training pairs for layer', layer_id)
        for walk in tqdm(walks):
            for i in range(len(walk)):
                for j in range(1, skip_window + 1):
                    if i - j >= 0:
                        pairs.append((vocab[walk[i]].index, vocab[walk[i - j]].index, layer_id))
                    if i + j < len(walk):
                        pairs.append((vocab[walk[i]].index, vocab[walk[i + j]].index, layer_id))
    return pairs

def generate_vocab(all_walks):
    index2word = []
    raw_vocab = defaultdict(int)

    for layer_id, walks in enumerate(all_walks):
        print('Counting vocab for layer', layer_id)
        for walk in tqdm(walks):
            for word in walk:
                raw_vocab[word] += 1

    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)

    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i

    return vocab, index2word

def load_walks(walk_file):
    print('Loading walks')
    all_walks = []
    with open(walk_file, 'r') as f:
        for line in f:
            content = line.strip().split()
            layer_id = int(content[0])
            if layer_id >= len(all_walks):
                all_walks.append([])
            all_walks[layer_id].append(content[1:])
    return all_walks

def save_walks(walk_file, all_walks):
    with open(walk_file, 'w') as f:
        for layer_id, walks in enumerate(all_walks):
            print('Saving walks for layer', layer_id)
            for walk in tqdm(walks):
                f.write(' '.join([str(layer_id)] + [str(x) for x in walk]) + '\n')

def generate(network_data, num_walks, walk_length, schema, file_name, window_size, num_workers, walk_file):
    if walk_file is not None:
        all_walks = load_walks(walk_file)
    else:
        all_walks = generate_walks(network_data, num_walks, walk_length, schema, file_name, num_workers)
        save_walks(file_name + '/walks.txt', all_walks)
    vocab, index2word = generate_vocab(all_walks)
    train_pairs = generate_pairs(all_walks, vocab, window_size, num_workers)

    return vocab, index2word, train_pairs

def generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples):
    edge_type_count = len(edge_types)
    neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]
    for r in range(edge_type_count):
        print('Generating neighbors for layer', r)
        g = network_data[edge_types[r]]
        for (x, y) in tqdm(g):
            ix = vocab[x].index
            iy = vocab[y].index
            neighbors[ix][r].append(iy)
            neighbors[iy][r].append(ix)
        for i in range(num_nodes):#如果邻居数量不够，补全邻居为自己，邻居数量太多，则从其中抽取足量个
            if len(neighbors[i][r]) == 0:
                neighbors[i][r] = [i] * neighbor_samples
            elif len(neighbors[i][r]) < neighbor_samples:
                neighbors[i][r].extend(list(np.random.choice(neighbors[i][r], size=neighbor_samples-len(neighbors[i][r]))))
            elif len(neighbors[i][r]) > neighbor_samples:
                neighbors[i][r] = list(np.random.choice(neighbors[i][r], size=neighbor_samples))
    return neighbors

def get_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        if np.linalg.norm(vector1) == np.linalg.norm(vector2)== 0:
            return 1.0
        elif np.linalg.norm(vector1) is None or np.linalg.norm(vector2) is None:
            print("一者不在训练图中")
        elif np.linalg.norm(vector1) == 0:
            return vector2 / np.linalg.norm(vector2)
        elif np.linalg.norm(vector2) == 0:
            return vector1 / np.linalg.norm(vector1)
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))#夹角余弦距离，其中一个出现0值或None会报错
    except Exception as e:
        pass


def evaluate(model, true_edges, false_edges):
    #f = open('true_edges.txt', 'w', encoding='utf-8')
    #f.write(str(true_edges))
    #f = open('false_edges.txt', 'w', encoding='utf-8')
    #f.write(str(false_edges))
    true_list = list()
    prediction_list = list()
    true_num = 0
    for edge in true_edges:#得到真实标签
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        #f = open('true_edge.txt', 'w', encoding='utf-8')
        #f.write(str(edge))
        #print(tmp_score)
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1#因为xiao本质上还是个排序为主的推荐算法，所以只要正样本score排在最前面，就认为预测结果正确，并不需要卡死阈值

    print(true_num)
    for edge in false_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        #f = open('false_edge.txt', 'w', encoding='utf-8')
        #f.write(str(edge))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):#得到模型预测的结果
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)#precision, recall, thresholds

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    recall=recall_score(y_true, y_pred, pos_label=1)
    precision=precision_score(y_true, y_pred, pos_label=1)
    #f = open('y_true.txt', 'w', encoding='utf-8')
    #f.write(str(y_true))
    #f = open('y_scores.txt', 'w', encoding='utf-8')
    #f.write(str(y_scores))
    #f = open('y_pred.txt', 'w', encoding='utf-8')
    #f.write(str(y_pred))
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps),accuracy_score(y_true, y_pred), specificity, recall, precision
