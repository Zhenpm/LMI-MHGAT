import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
#from numpy import random

from utils import *

from sklearn.model_selection import KFold
from tensorflow.contrib.tensorboard.plugins import projector

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# tf1.x version
# 自适应显存占用
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def get_batches(pairs, neighbors, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    for idx in range(n_batches):
        x, y, t, neigh = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])#layer_id
            neigh.append(neighbors[pairs[index][0]])
        yield (np.array(x).astype(np.int32), np.array(y).reshape(-1, 1).astype(np.int32), np.array(t).astype(np.int32), np.array(neigh).astype(np.int32))

def train_model(network_data, feature_dic, log_name):
    vocab, index2word, train_pairs = generate(network_data, args.num_walks, args.walk_length, args.schema, file_name, args.window_size, args.num_workers, args.walk_file)

    edge_types = list(network_data.keys())

    num_nodes = len(index2word)
    edge_type_count = len(edge_types)
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions # Dimension of the embedding vector.
    embedding_u_size = args.edge_dim
    u_num = edge_type_count
    num_sampled = args.negative_samples # Number of negative examples to sample.
    dim_a = args.att_dim
    att_head = args.att_head
    neighbor_samples = args.neighbor_samples
    log_store=args.log_store
    model_dir=args.model_dir

    neighbors = generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples)

    graph = tf.Graph()

    if feature_dic is not None:
        feature_dim = len(list(feature_dic.values())[0])
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for key, value in feature_dic.items():#不在feature文件中的节点feature为0
            if key in vocab:
                features[vocab[key].index, :] = np.array(value)

    with graph.as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.name_scope('Parameters'):
            if feature_dic is not None:
                node_features = tf.Variable(features, name='node_features', trainable=False)
                feature_weights = tf.Variable(tf.truncated_normal([feature_dim, embedding_size], stddev=1.0))

                embed_trans = tf.Variable(tf.truncated_normal([feature_dim, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
                u_embed_trans = tf.Variable(tf.truncated_normal([edge_type_count, feature_dim, embedding_u_size], stddev=1.0 / math.sqrt(embedding_size)))
            else:
                node_embeddings = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0), name='node_embeddings')
                node_type_embeddings = tf.Variable(tf.random_uniform([num_nodes, u_num, embedding_u_size], -1.0, 1.0), name='node_type_embeddings')

            trans_weights = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, embedding_size ], stddev=1.0 / math.sqrt(embedding_size))
                                        , name='trans_weights')# embedding_size // att_head这里如果除不尽，后头就会出问题，报错,我去掉试试
            trans_weights_s1 = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, dim_a], stddev=1.0 / math.sqrt(embedding_size))
                                        , name='trans_weights_s1')
            trans_weights_s2 = tf.Variable(tf.truncated_normal([edge_type_count, dim_a, att_head], stddev=1.0 / math.sqrt(embedding_size))
                                        , name='trans_weights_s2')
            trans_weights_l = tf.Variable(tf.truncated_normal([edge_type_count, att_head, 1], stddev=1.0 / math.sqrt(embedding_size))
                                        , name='trans_weights_l')
            trans_weights_f = tf.Variable(tf.truncated_normal([embedding_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size))
                                        , name='trans_weights_f')
            trans_weights_a = tf.Variable(tf.truncated_normal([edge_type_count, 1, edge_type_count],
                                                        stddev=1.0 / math.sqrt(embedding_size)), name='trans_weights_a')
            #trans_weights_a = tf.Variable(tf.truncated_normal([edge_type_count, att_head, edge_type_count],
            #                                                  stddev=1.0 / math.sqrt(embedding_size)),
            #                              name='trans_weights_a')
            '''
            trans_weights = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, embedding_size ], stddev=1.0 / math.sqrt(embedding_size))
                                        , name='trans_weights')# embedding_size // att_head这里如果除不尽，后头就会出问题，报错,我去掉试试
            trans_weights_s1 = tf.Variable(tf.truncated_normal([edge_type_count, att_head, embedding_u_size, dim_a], stddev=1.0 / math.sqrt(embedding_size))
                                        , name='trans_weights_s1')
            trans_weights_s2 = tf.Variable(tf.truncated_normal([edge_type_count, att_head, embedding_u_size, dim_a], stddev=1.0 / math.sqrt(embedding_size))
                                        , name='trans_weights_s2')
            trans_weights_s3 = tf.Variable(tf.truncated_normal([edge_type_count, att_head, embedding_u_size, embedding_u_size], stddev=1.0 / math.sqrt(embedding_size))
                                        , name='trans_weights_s3')
            trans_weights_l = tf.Variable(tf.truncated_normal([edge_type_count, 1, edge_type_count*att_head], stddev=1.0 / math.sqrt(embedding_size))
                                        , name='trans_weights_l')
            '''
            #trans_weights_s1 = tf.ones(shape=[edge_type_count, embedding_u_size, dim_a])
            #trans_weights_s2 = tf.ones(shape=[edge_type_count, dim_a, att_head])
            nce_weights = tf.Variable(tf.truncated_normal([num_nodes, embedding_size], stddev=1.0 / math.sqrt(embedding_size))
                                      , name='nce_weights')
            nce_biases = tf.Variable(tf.zeros([num_nodes]), name='nce_biases')

        # Input data
        with tf.name_scope('Input'):
            train_inputs = tf.placeholder(tf.int32, shape=[None],name='train_inputs')#batch_size
            train_labels = tf.placeholder(tf.int32, shape=[None, 1],name='train_labels')
            train_types = tf.placeholder(tf.int32, shape=[None],name='train_types')
            node_neigh = tf.placeholder(tf.int32, shape=[None, edge_type_count, neighbor_samples],name='node_neigh')

        # Look up embeddings for nodes
        with tf.name_scope('LookingEmbedding'):
            if feature_dic is not None:
                node_embed = tf.nn.embedding_lookup(node_features, train_inputs)
                node_embed = tf.matmul(node_embed, embed_trans)
            else:
                node_embed = tf.nn.embedding_lookup(node_embeddings, train_inputs,name="node_embed_lookup")
                #(batch_size,embedding_size),此处node_num不是num_node，是查询的节点数
        with tf.name_scope('Edge'):
            #按子图聚合邻居信息，若没有邻居那邻居就是自己
            if feature_dic is not None:
                node_embed_neighbors = tf.nn.embedding_lookup(node_features, node_neigh)
                node_embed_tmp = tf.concat([tf.matmul(tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, 0], [-1, 1, -1, -1]),
                       [-1, feature_dim]), tf.reshape(tf.slice(u_embed_trans, [i, 0, 0], [1, -1, -1]),
                       [feature_dim, embedding_u_size])) for i in range(edge_type_count)], axis=0)
                node_type_embed = tf.transpose(tf.reduce_mean(tf.reshape(node_embed_tmp,
                                 [edge_type_count, -1, neighbor_samples, embedding_u_size]), axis=2), perm=[1,0,2])
            else:
                node_embed_neighbors = tf.nn.embedding_lookup(node_type_embeddings, node_neigh,name="node_embed_neighbors_lookup")
                #(batch_size,edge_type_count,neighbor_samples,edge_type_count,embedding_u_size)
                #tf.slice(inputs, begin, size, name)
                node_embed_tmp = tf.concat([tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, i, 0], [-1, 1, -1, 1, -1],name="node_embed_neighbors_slice"),
                                 #(batch_size,1,neighbor_samples,1,embedding_u_size)
                                 [1, -1, neighbor_samples, embedding_u_size],name="node_embed_neighbors_reshape")
                                 #(1,batch_size,neighbor_samples,embedding_u_size)
                                 for i in range(edge_type_count)], axis=0,name="node_embed_tmp")
                                 #(edge_type_count,batch_size,neighbor_samples,embedding_u_size)

                node_type_embed = tf.transpose(tf.reduce_mean(node_embed_tmp, axis=2), perm=[1,0,2],name="node_type_embed")#利用tf.reduce_mean做平均池化
                #(batch_size,edge_type_count,embedding_u_size)
        with tf.name_scope('Attention'):
            '''
            #linear
            trans_w = tf.nn.embedding_lookup(trans_weights, train_types, name="trans_w_lookup")
            attention = tf.nn.embedding_lookup(trans_weights_a, train_types, name="attention")
            '''
            #structal attention
            #MLP聚合子图信息，linear调整，softmax
            trans_w = tf.nn.embedding_lookup(trans_weights, train_types,name="trans_w_lookup")
            #(batch_size, embedding_u_size, embedding_size)
            trans_w_s1 = tf.nn.embedding_lookup(trans_weights_s1, train_types,name="trans_w_s1_lookup")
            #(batch_size, embedding_u_size, dim_a)
            trans_w_s2 = tf.nn.embedding_lookup(trans_weights_s2, train_types,name="trans_w_s2_lookup")
            #(batch_size, dim_a, att_head)
            trans_w_l = tf.nn.embedding_lookup(trans_weights_l, train_types,name="trans_w_l_lookup")

            attention = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(tf.matmul(node_type_embed, trans_w_s1,name="att_matmul_1"),
                                                                   #(batch_size,edge_type_count,dim_a)
                                                                   name="att_tanh"),
                                                                   trans_w_s2,name="att_matmul_2"),
                                                                   #(batch_size,edge_type_count,att_head)
                                                                   [-1, u_num],name="att_reshape_1"),
                                                                   #(batch_size*att_head,edge_type_count)
                                                                   name="att_softmax"),
                                                                   [-1, att_head, u_num],name="att_reshape_2")
                                                                   #(batch_size,att_head,edge_type_count)

            '''
            #muti-head-attention
            trans_w = tf.nn.embedding_lookup(trans_weights, train_types,name="trans_w_lookup")
            #(batch_size, embedding_u_size, embedding_size)
            trans_w_s1 = tf.nn.embedding_lookup(trans_weights_s1, train_types,name="trans_w_s1_lookup")
            #(batch_size, att_head, embedding_u_size, dim_a)
            trans_w_s2 = tf.nn.embedding_lookup(trans_weights_s2, train_types,name="trans_w_s2_lookup")
            #(batch_size, att_head, embedding_u_size, dim_a)
            trans_w_s3 = tf.nn.embedding_lookup(trans_weights_s3, train_types,name="trans_w_s3_lookup")
            #(batch_size, att_head, embedding_u_size, embedding_u_size)
            trans_w_l = tf.nn.embedding_lookup(trans_weights_l, train_types,name="trans_w_l_lookup")
            node_type_embed =tf.reshape(node_type_embed,[-1,1,edge_type_count,embedding_u_size])

            q=tf.matmul(node_type_embed, trans_w_s1,name='q')#(node_num,att_head,edge_type_count,dim_a)
            k=tf.matmul(node_type_embed, trans_w_s2,name="k")#(node_num,att_head,edge_type_count,dim_a)
            matmul_qk = tf.matmul(q, k, transpose_b=True)#(node_num,att_head,edge_type_count,edge_type_count)
            dk = tf.cast(tf.shape(node_type_embed)[-1], tf.float32,name="dk")
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
            attention = tf.nn.softmax(scaled_attention_logits, axis=-1,name="attention")
            '''
            #attention = tf.reshape(tf.reshape(tf.matmul(tf.matmul(node_type_embed, trans_w_s1), trans_w_s2), [-1, u_num]), [-1, att_head, u_num])
        with tf.name_scope('Embedding'):
            '''
            ###平均池化
            node_type_embed = tf.matmul(attention, node_type_embed,name="att_node_type_embed_matmul")
            #(batch_size,att_head,embedding_u_size)
            node_embed = node_embed + tf.reshape(tf.reduce_mean(tf.matmul(node_type_embed, trans_w,name="node_type_embed_matmul"),
            #(batch_size,att_head,embedding_size)
            axis=1, keep_dims=True,name="embedding_pool"),
            [-1, embedding_size],name="embedding_reshape")
            #(batch_size,embedding_size)
            #node_embed = tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size])
            '''

            ###最大池化
            # (batch_size,att_head,edge_type_count)*(batch_size,edge_type_count,embedding_u_size)
            node_type_embed = tf.matmul(attention, node_type_embed,name="att_node_type_embed_matmul")
            #(batch_size,att_head,embedding_u_size)

            #
            node_embed = node_embed + tf.reshape(tf.nn.max_pool(tf.reshape(tf.matmul(node_type_embed, trans_w,name="node_type_embed_matmul"),
            [-1, att_head,1,embedding_size],name="pool_reshape"),
            ksize=[1,att_head,1,1],strides=[1,1,1,1],padding="VALID",name="max_pool"),
            #(batch_size,1,embedding_size)
            [-1, embedding_size],name="embedding_reshape")
            #(batch_size,embedding_size)
            #node_embed = tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size])
            '''
            #linear fusion
            #node_embed = tf.matmul(node_embed, trans_weights_f ,name="linear_fusion")

            node_type_embed = tf.matmul(attention, node_type_embed, name="att_node_type_embed_matmul")
            # (batch_size,1,embedding_u_size)
            node_embed = node_embed + tf.reshape(node_type_embed,
                                                 # (node_num,1,embedding_size)
                                                 [-1, embedding_size], name="embedding_reshape")
            '''
            '''
            #特征交叉池化
            node_type_embed = tf.matmul(attention, node_type_embed,name="att_node_type_embed_matmul")
            #(batch_size,att_head,embedding_u_size)
            node_type_embed = tf.matmul(node_type_embed, trans_w,name="node_type_embed_matmul")
            #(batch_size,att_head,embedding_size)
            square_of_sum = tf.square(tf.reduce_sum(node_type_embed, axis=1, keepdims=True))
            sum_of_square = tf.reduce_sum(node_type_embed * node_type_embed, axis=1, keepdims=True)
            cross_term = 0.5 * (square_of_sum - sum_of_square)
            #(batch_size,1,embedding_size)
            node_embed = node_embed + tf.reshape(cross_term, [-1, embedding_size])
            '''
            '''
            #structal-attention-linear
            node_type_embed = tf.matmul(attention, node_type_embed,name="att_node_type_embed_matmul")
            #(batch_size,att_head,embedding_u_size)
            node_embed = node_embed + tf.reshape(tf.matmul(tf.matmul(node_type_embed, trans_w,name="node_type_embed_matmul"),
            #(batch_size,att_head,embedding_size)
            trans_w_l, transpose_a=True,name="node_type_embed_linear"),
            [-1, embedding_size],name="embedding_reshape")
            #(batch_size,embedding_size)
            #node_embed = tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size])
            '''
            '''
            #muti-attention-linear
            v=tf.matmul(node_type_embed, trans_w_s3,name="v")#(node_num,att_head,edge_type_count,embedding_u_size)
            node_type_embed = tf.matmul(attention, v,name="att_node_type_embed_matmul")#(node_num,att_head,edge_type_count,embedding_u_size)
            node_type_embed =tf.reshape(node_type_embed, [-1,att_head*edge_type_count,embedding_u_size])
            node_type_embed = tf.matmul(node_type_embed, trans_w,name="att_node_type_linear")
            #(batch_size,att_head*edge_type_count,embedding_size)
            node_embed = node_embed + tf.reshape(tf.matmul(trans_w_l,node_type_embed), [-1, embedding_size])
            '''
            if feature_dic is not None:
                node_feat = tf.nn.embedding_lookup(node_features, train_inputs)
                node_embed = node_embed + tf.matmul(node_feat, feature_weights)

            last_node_embed = tf.nn.l2_normalize(node_embed, axis=1)#在数据张量的第2个维度上进行norm
        with tf.name_scope('Loss'):
            #loss = tf.reduce_mean(
            #    tf.nn.nce_loss(
            #        weights=nce_weights,
            #        biases=nce_biases,
            #        labels=train_labels,
            #        inputs=last_node_embed,
            #        num_sampled=num_sampled,
            #        num_classes=num_nodes))
            loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=last_node_embed,
                    num_sampled=num_sampled,
                    num_classes=num_nodes,
                    name='sampled_softmax_loss'))
            plot_loss = tf.summary.scalar("loss", loss)

        # Optimizer.

        with tf.name_scope('Train'):
            optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=20)

        merged = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        # Initializing the variables
        with tf.name_scope('Init'):
            init = tf.global_variables_initializer()


    # Launch the graph
    print("Optimizing")

    with tf.Session(graph=graph, config=config) as sess:
        writer = tf.summary.FileWriter( log_store+ log_name, sess.graph) # tensorboard --logdir=./runs
        #draw_writer = tf.summary.FileWriter( log_store+ log_name+'_draw')
        sess.run(init)

        print('Training')
        g_iter = 0
        best_score = 0
        test_score = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        patience = 0
        for epoch in range(epochs):
            #print("attention.type:",str(type(attention)))
            #print("attention:",str(attention.shape))
            #print("node_type_embed:",str(node_type_embed.shape))
            np.random.shuffle(train_pairs)
            batches = get_batches(train_pairs, neighbors, batch_size)

            data_iter = tqdm(batches,
                            desc="epoch %d" % (epoch),
                            total=(len(train_pairs) + (batch_size - 1)) // batch_size,
                            bar_format="{l_bar}{r_bar}")
            avg_loss = 0.0

            for i, data in enumerate(data_iter):
                feed_dict = {train_inputs: data[0], train_labels: data[1], train_types: data[2], node_neigh: data[3]}#train_labels即为pair中的y
                _, loss_value, summary_str = sess.run([optimizer, loss, merged], feed_dict)
                writer.add_summary(summary_str, g_iter)

                g_iter += 1

                avg_loss += loss_value

                if i % 5000 == 0:
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss_value
                    }
                    data_iter.write(str(post_fix))

            final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
            for i in range(edge_type_count):
                for j in range(num_nodes):
                    final_model[edge_types[i]][index2word[j]] = np.array(sess.run(last_node_embed, {train_inputs: [j], train_types: [i], node_neigh: [neighbors[j]]})[0])
            valid_aucs, valid_f1s, valid_prs,valid_accuracy,valid_specificity,valid_recall,valid_precision = [], [], [],[], [], [], []
            test_aucs, test_f1s, test_prs,test_accuracy,test_specificity,test_recall,test_precision = [], [], [],[], [], [], []
            for i in range(edge_type_count):
                if args.eval_type == 'all' or edge_types[i] in args.eval_type.split(','):
                    tmp_auc, tmp_f1, tmp_pr,accuracy,specificity,recall,precision = evaluate(final_model[edge_types[i]], valid_true_data_by_edge[edge_types[i]], valid_false_data_by_edge[edge_types[i]])
                    valid_aucs.append(tmp_auc)
                    valid_f1s.append(tmp_f1)
                    valid_prs.append(tmp_pr)
                    valid_accuracy.append(accuracy)
                    valid_specificity.append(specificity)
                    valid_recall.append(recall)
                    valid_precision.append(precision)

                    tmp_auc, tmp_f1, tmp_pr,accuracy,specificity,recall,precision = evaluate(final_model[edge_types[i]], testing_true_data_by_edge[edge_types[i]], testing_false_data_by_edge[edge_types[i]])
                    test_aucs.append(tmp_auc)
                    test_f1s.append(tmp_f1)
                    test_prs.append(tmp_pr)
                    test_accuracy.append(accuracy)
                    test_specificity.append(specificity)
                    test_recall.append(recall)
                    test_precision.append(precision)
            print('valid auc:', np.mean(valid_aucs))
            print('valid pr:', np.mean(valid_prs))
            print('valid f1:', np.mean(valid_f1s))
            print('valid accuracy:', np.mean(valid_accuracy))
            print('valid specificity:', np.mean(valid_specificity))
            print('valid recall:', np.mean(valid_recall))
            print('valid precision:', np.mean(valid_precision))
            #auc_summary =tf.summary.scalar('ROC-AUC', np.mean(valid_aucs))
            #f1_summary =tf.summary.scalar('F1', np.mean(valid_f1s))
            #pr_summary =tf.summary.scalar('PR-AUC', np.mean(valid_prs))
            #acc_summary =tf.summary.scalar('accuracy', np.mean(valid_accuracy))
            #spec_summary =tf.summary.scalar('specificity', np.mean(valid_specificity))
            #rec_summary =tf.summary.scalar('recall', np.mean(valid_recall))
            #pre_summary =tf.summary.scalar('precision', np.mean(valid_precision))
            #draw_summary = tf.summary.merge([auc_summary,f1_summary,pr_summary,acc_summary,spec_summary,rec_summary,pre_summary])
            #writer.add_summary(draw_summary.eval(), epoch)

            average_auc = np.mean(test_aucs)
            average_f1 = np.mean(test_f1s)
            average_pr = np.mean(test_prs)
            average_accuracy = np.mean(test_accuracy)
            average_specificity = np.mean(test_specificity)
            average_recall = np.mean(test_recall)
            average_precision = np.mean(test_precision)

            cur_score = np.mean(valid_aucs)
            if cur_score > best_score:
                best_score = cur_score
                test_score = (average_auc, average_f1, average_pr,average_accuracy,average_specificity,average_recall,average_precision)
                patience = 0
                saver.save(sess, model_dir)
                #pconfig = projector.ProjectorConfig()
                #embedding = pconfig.embeddings.add()
                #embedding.name=final_model[edge_types[1]]
            else:
                patience += 1
                if patience > args.patience:#早停法
                    print('Early Stopping')
                    break
    #f = open('result.txt','w',encoding='utf-8')
    #f.write(str(test_score))

    return test_score


if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    print(args)
    if args.features is not None:
        feature_dic = load_feature_data(args.features)
    else:
        feature_dic = None

    log_name = file_name.split('/')[-1] + f'_neighbor_{args.neighbor_samples}_atthead_{args.att_head}_attdim_{args.att_dim}'

    # training_data_by_type = load_training_data(file_name + '/train.txt')
    # valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + '/valid.txt')
    #testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + '/test.txt')
    main_data=load_data(file_name + '/main.txt')
    extra_data = load_training_data(file_name + '/extra.txt')
    _, testing_false_data_by_edge=load_testing_data(file_name + '/testing_false_data_by_edge.txt')
    _, valid_false_data_by_edge=load_testing_data(file_name + '/valid_false_data_by_edge.txt')
    if args.kfold==0:
        test_data, train_data = data_split(main_data, ratio=0.2, shuffle=True)
        testing_true_data_by_edge= load_dict(range(len(test_data)), train_data)
        training_main_data_by_type= load_dict(range(len(train_data)), train_data)
        valid_data, train_data = data_split(training_main_data_by_type["1"], ratio=0.1, shuffle=True)
        valid_true_data_by_edge = load_dict(range(len(valid_data)),valid_data)
        training_main_data_by_type= load_dict(range(len(train_data)), train_data)
        network=training_main_data_by_type.copy()#浅复制
        if extra_data is not None:
            network.update(extra_data)
        average_auc, average_f1, average_pr,average_accuracy,average_specificity,average_recall,average_precision = train_model(network, feature_dic,
                                                          log_name + '_' + time.strftime('%Y-%m-%d %H-%M-%S',
                                                                                         time.localtime(time.time())))
        print('Overall ROC-AUC:', average_auc)
        print('Overall PR-AUC', average_pr)
        print('Overall F1:', average_f1)
        print('Overall accuracy:', average_accuracy)
        print('Overall specificity:', average_specificity)
        print('Overall recall', average_recall)
        print('Overall precision:', average_precision)

    else:

        kf = KFold(n_splits=args.kfold, shuffle=True,random_state=1)  # 初始化KFold
        KFold_auc=[]
        KFold_pr=[]
        KFold_f1=[]
        KFold_accuracy = []
        KFold_specificity = []
        KFold_recall = []
        KFold_precision = []
        #f = open('valid_false_data_by_edge.txt', 'w', encoding='utf-8')
        #f.write(str(valid_false_data_by_edge))
        for training_main_index, testing_true_data_index in kf.split(main_data):  # 调用split方法切分数据
            #f = open('training_main_index.txt', 'w', encoding='utf-8')
            #f.write(str(training_main_index))
            training_main_data_by_type=load_dict(training_main_index,main_data)
            #f = open('training_main_data_by_type.txt', 'w', encoding='utf-8')
            #f.write(str(training_main_data_by_type))
            testing_true_data_by_edge=load_dict(testing_true_data_index,main_data)
            valid_data, train_data = data_split(training_main_data_by_type["1"], ratio=0.1, shuffle=True)
            valid_true_data_by_edge = load_dict(range(len(valid_data)),valid_data)
            training_main_data_by_type= load_dict(range(len(train_data)), train_data)
            #f = open('valid_true_data_by_edge.txt', 'w', encoding='utf-8')
            #f.write(str(valid_true_data_by_edge))
            #network=merge(training_main_data_by_type,extra_data)
            network=training_main_data_by_type.copy()#浅复制
            if extra_data is not None:
                network.update(extra_data)
            #f = open('extra_data.txt', 'w', encoding='utf-8')
            #f.write(str(extra_data))
            average_auc, average_f1, average_pr,average_accuracy,average_specificity,average_recall,average_precision = train_model(network, feature_dic,
                                                              log_name + '_' + time.strftime('%Y-%m-%d %H-%M-%S',
                                                                                             time.localtime(time.time())))
            KFold_auc.append(average_auc)
            KFold_pr.append(average_pr)
            KFold_f1.append(average_f1)
            KFold_accuracy.append(average_accuracy)
            KFold_specificity.append(average_specificity)
            KFold_recall.append(average_recall)
            KFold_precision.append(average_precision)
            print('Overall ROC-AUC:', average_auc)
            print('Overall PR-AUC', average_pr)
            print('Overall F1:', average_f1)
            print('Overall accuracy:', average_accuracy)
            print('Overall specificity:', average_specificity)
            print('Overall recall', average_recall)
            print('Overall precision:', average_precision)
        print('KFold ROC-AUC:', np.mean(KFold_auc))
        print('KFold ROC-AUC vearance:', np.var(KFold_auc))
        print('KFold PR-AUC', np.mean(KFold_pr))
        print('KFold PR-AUC vearance:', np.var(KFold_pr))
        print('KFold F1:', np.mean(KFold_f1))
        print('KFold F1 vearance:', np.var(KFold_f1))
        print('KFold_accuracy:', np.mean(KFold_accuracy))
        print('KFold_accuracy vearance:', np.var(KFold_accuracy))
        print('KFold_specificity:', np.mean(KFold_specificity))
        print('KFold_specificity vearance:', np.var(KFold_specificity))
        print('KFold_recall:', np.mean(KFold_recall))
        print('KFold_recall vearance:', np.var(KFold_recall))
        print('KFold_precision:', np.mean(KFold_precision))
        print('KFold_precision vearance:', np.var(KFold_precision))
