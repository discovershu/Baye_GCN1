from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from gcn_gae_opinion.metrics import masked_mae_np
from gcn_gae_opinion.optimizer import OptimizerAE, OptimizerDir, OptimizerDir2
from gcn_gae_opinion.input_data import load_data_sub
from gcn_gae_opinion.model import GCNModelAE, GCNModelDir, GCNModelDir2
from gcn_gae_opinion.preprocessing import preprocess_graph, construct_feed_dict_sub, sparse_to_tuple, mask_test_syn_sub, mask_test_syn_sub_dif, mask_test_syn_sub_dif2
from gcn_gae_opinion.metrics import masked_dir_error, masked_dir_error2

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 1, 'Number of units in hidden layer 2, P in our paper.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_Dir', 'Model string.')
flags.DEFINE_string('dataset', 'epinion', 'Dataset string.')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')

# flags.DEFINE_float('alpha_0', 3, 'prior of Beta distribution.')
# flags.DEFINE_float('beta_0', 6.9, 'prior of Beta distribution.')
flags.DEFINE_float('p_encode', 0.01, 'trade off parameter of auto_encode.')
flags.DEFINE_float('p_kl', 0.01, 'trade off parameter of KL-divergence.')
flags.DEFINE_integer('KL_m', 10, 'approximate of the infinite sum in KLD.')
flags.DEFINE_float('test_rat', 0.2, 'test number of dataset.')
flags.DEFINE_integer('T', 6, 'time winidow.')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)
print(seed)
# Load data5
# adj, features = load_data(dataset_str)
adj, features = load_data_sub()
# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()
adj_train = adj

label_1, label_2, label_3, train_mask, test_mask, uncertainty = mask_test_syn_sub_dif2(FLAGS.test_rat, T=FLAGS.T, seed=0)

if FLAGS.features == 0:
    features = sp.identity(label_1.shape[0])  # featureless

# Some preprocessing
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'labels_1': tf.placeholder(tf.float32, shape=(None, label_1.shape[1])),
    'labels_2': tf.placeholder(tf.float32, shape=(None, label_1.shape[1])),
    'omega_test': tf.placeholder(tf.float32, shape=(None, label_1.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'labels_3': tf.placeholder(tf.float32, shape=(None, label_1.shape[1])),
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_Dir':
    model = GCNModelDir2(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_Dir':
        opt = OptimizerDir2(
                           model=model,
                           label_1=placeholders['labels_1'], label_2=placeholders['labels_2'],
                           mask=placeholders['labels_mask'], label_3=placeholders['labels_3'], )

# Initialize session
sess = tf.Session()

adj_label = adj_train + sp.eye(adj_train.shape[0])
# adj_label = adj_train
adj_label = sparse_to_tuple(adj_label)

result = []
for k in range(10):
    sess.run(tf.global_variables_initializer())
    label_1, label_2, label_3, train_mask, test_mask, uncertainty = mask_test_syn_sub_dif2(FLAGS.test_rat, T=FLAGS.T, seed=k)
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict_sub(adj_norm, adj_label, features, placeholders, label_1, label_2, train_mask,
                                            uncertainty)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run training epoch
        outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

        # Compute trining loss
        train_cost = outs[1]
        # print(train_cost)
        if np.mod(epoch + 1, 100) == 0:
            # print("epoch:", epoch + 1, "Loss:", train_cost)
            feed_dict = construct_feed_dict_sub(adj_norm, adj_label, features, placeholders, label_1, label_2,
                                                test_mask,
                                                uncertainty)
            feed_dict.update({placeholders['dropout']: 0.0})
            # Run single weight update
            outs = sess.run([opt.cost, model.belief1, model.belief2, model.belief3, model.uncertainty], feed_dict=feed_dict)
            # error = masked_dir_error(outs[1], outs[2], outs[3], label_1, label_2, label_3, test_mask)
            model2_error = masked_dir_error2(outs[1], outs[2], outs[3],  outs[3], label_1, label_2, label_3, uncertainty, test_mask)
            # Compute test loss
            print("epoch:", epoch + 1, "Loss:", "{:.3f}".format(outs[0]), "error:", "{:.2f}".format(model2_error))
    # feed_dict = construct_feed_dict_sub(adj_norm, adj_label, features, placeholders, label_1, label_2,
    #                                     test_mask,
    #                                     uncertainty)
    # feed_dict.update({placeholders['dropout']: 0.0})
    # test_cost = sess.run(opt.cost, feed_dict=feed_dict)
    result.append(model2_error)
    print("Optimization Finished!")
print("final loss:", np.mean(result))
