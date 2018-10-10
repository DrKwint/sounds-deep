from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import operator
from functools import reduce
import json
import pickle

import numpy as np
import sonnet as snt
import tensorflow as tf
import scipy
import sklearn

import sounds_deep.contrib.data.data as data
import sounds_deep.contrib.util.scaling as scaling
import sounds_deep.contrib.util.util as util
import sounds_deep.contrib.models.cpvae as cpvae
import sounds_deep.contrib.models.vae as vae
import sounds_deep.contrib.parameterized_distributions.discretized_logistic as discretized_logistic
import sounds_deep.contrib.util.plot as plot
import sounds_deep.contrib.util.eval_cpvae as eval_cpvae

parser = argparse.ArgumentParser(description='Train a VAE model.')
parser.add_argument(
    '--task', type=str, default='train', choices=['train', 'eval'])

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_dimension', type=int, default=50)
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--learning_rate', type=float, default=3e-5)
parser.add_argument('--dataset', type=str, default='mnist')

parser.add_argument('--max_leaf_nodes', type=int, default=20)
parser.add_argument('--max_depth', type=int, default=10)
parser.add_argument('--update_period', type=int, default=3)
parser.add_argument('--update_samples', type=int, default=20)

parser.add_argument('--beta', type=float, default=1.)
parser.add_argument('--gamma', type=float, default=10.)
parser.add_argument('--delta', type=float, default=1.)

parser.add_argument('--output_dir', type=str, default='./')
parser.add_argument('--load', action='store_true')

# 2leaf walks from one leaf to another
# class instance generates a walk from encoded point to class
parser.add_argument('--conf_matr', action='store_true')
parser.add_argument(
    '--viz_task', type=str, choices=['2leaf', 'class_instance', 'single_dim'])
parser.add_argument('--viz_steps', type=int, default=3)
parser.add_argument('--viz_classes', nargs='*', type=int)
parser.add_argument('--viz_dimension', type=int, default=None)
parser.add_argument('--viz_dir', type=str, default='./')
args = parser.parse_args()

# enforce arg invariants
if args.viz_task == '2leaf':
    assert len(args.viz_classes) == 2

# sampled img save directory
if args.output_dir == './' and 'SLURM_JOB_ID' in os.environ.keys():
    job_id = os.environ['SLURM_JOB_ID']
    output_directory = 'cpvae_{}'.format(job_id)
    args.output_dir = output_directory
    if not args.load: os.mkdir(output_directory)
else:
    if args.output_dir == './':
        args.output_dir = './'
        output_directory = './'
    else:
        output_directory = args.output_dir
        if not args.load: os.mkdir(output_directory)

with open(os.path.join(args.output_dir, 'cmd_line_arguments.json'), 'w') as fp:
    if not args.load: json.dump(vars(args), fp)
print(vars(args))

# load the data
if args.dataset == 'cifar10':
    train_data, train_labels, test_data, test_labels = data.load_cifar10(
        './data/cifar10/')
elif args.dataset == 'mnist':
    train_data, train_labels, test_data, test_labels = data.load_mnist(
        './data/mnist/')
    train_data = np.reshape(train_data, [-1, 28, 28, 1])
    test_data = np.reshape(test_data, [-1, 28, 28, 1])
elif args.dataset == 'fmnist':
    train_data, train_labels, test_data, test_labels = data.load_fmnist(
        './data/fmnist/')
    train_data = np.reshape(train_data, [-1, 28, 28, 1])
    test_data = np.reshape(test_data, [-1, 28, 28, 1])

train_data_shape = (args.batch_size, ) + train_data.shape[1:]
test_data_shape = (args.batch_size, ) + test_data.shape[1:]
train_label_shape = (args.batch_size, ) + train_labels.shape[1:]
test_label_shape = (args.batch_size, ) + test_labels.shape[1:]

train_batches_per_epoch = train_data.shape[0] // args.batch_size
test_batches_per_epoch = test_data.shape[0] // args.batch_size
train_gen = data.parallel_data_generator([train_data, train_labels],
                                         args.batch_size)
test_gen = data.parallel_data_generator([test_data, test_labels],
                                        args.batch_size)

# build the model
if args.dataset == 'cifar10':
    encoder_module = snt.Sequential([
        snt.Conv2D(16, 3),
        snt.Residual(snt.Conv2D(16, 3)),
        snt.Residual(snt.Conv2D(16, 3)), scaling.squeeze2d,
        snt.Conv2D(64, 3),
        snt.Residual(snt.Conv2D(64, 3)),
        snt.Residual(snt.Conv2D(64, 3)), scaling.squeeze2d,
        snt.Conv2D(64, 3),
        snt.Residual(snt.Conv2D(64, 3)),
        snt.Residual(snt.Conv2D(64, 3)), scaling.squeeze2d,
        snt.Conv2D(128, 3),
        snt.Residual(snt.Conv2D(128, 3)),
        snt.Residual(snt.Conv2D(128, 3)), scaling.squeeze2d,
        snt.Conv2D(256, 3),
        snt.Residual(snt.Conv2D(256, 3)),
        snt.Residual(snt.Conv2D(256, 3)), scaling.squeeze2d,
        tf.keras.layers.Flatten(),
        snt.Linear(100)
    ])
    decoder_module = snt.Sequential([
        lambda x: tf.reshape(x, [-1, 1, 1, args.latent_dimension]),
        snt.Conv2D(32, 3),
        snt.Residual(snt.Conv2D(32, 3)),
        snt.Residual(snt.Conv2D(32, 3))
    ] + [
        scaling.unsqueeze2d,
        snt.Conv2D(32, 3),
        snt.Residual(snt.Conv2D(32, 3)),
        snt.Residual(snt.Conv2D(32, 3))
    ] * 5 + [snt.Conv2D(3, 3)])
    output_distribution_fn = discretized_logistic.DiscretizedLogistic
elif args.dataset == 'mnist' or args.dataset == 'fmnist':
    encoder_module = snt.Sequential(
        [tf.keras.layers.Flatten(),
         snt.nets.MLP([200, 200])])
    decoder_module = snt.Sequential([
        lambda x: tf.reshape(x, [-1, 1, 1, args.latent_dimension]),
        snt.Residual(snt.Conv2D(1, 1)),
        lambda x: tf.reshape(x, [-1, args.latent_dimension]),
        snt.nets.MLP([200, 200,
                      784]), lambda x: tf.reshape(x, [-1, 28, 28, 1])
    ])
    output_distribution_fn = vae.BERNOULLI_FN


def train_feed_dict_fn():
    feed_dict = dict()
    arrays = next(train_gen)
    feed_dict[data_ph] = arrays[0]
    feed_dict[label_ph] = arrays[1]
    return feed_dict


def test_feed_dict_fn():
    feed_dict = dict()
    arrays = next(test_gen)
    feed_dict[data_ph] = arrays[0]
    feed_dict[label_ph] = arrays[1]
    return feed_dict


def test_classification_rate(session):
    codes = []
    labels = []
    for _ in range(test_batches_per_epoch):
        c, l = session.run([model.latent_posterior_sample, label_ph],
                           feed_dict=test_feed_dict_fn())
        codes.append(c)
        labels.append(l)
    codes = np.squeeze(np.concatenate(codes, axis=1))
    labels = np.argmax(np.concatenate(labels), axis=1)
    return decision_tree.score(codes, labels)


decision_tree = sklearn.tree.DecisionTreeClassifier(
    max_depth=args.max_depth,
    min_weight_fraction_leaf=0.01,
    max_leaf_nodes=args.max_leaf_nodes)

model = cpvae.CPVAE(
    args.latent_dimension,
    args.max_leaf_nodes,
    10,
    decision_tree,
    encoder_module,
    decoder_module,
    beta=args.beta,
    gamma=args.gamma,
    delta=args.delta,
    output_dist_fn=output_distribution_fn)

# build model
data_ph = tf.placeholder(
    tf.float32,
    shape=(args.batch_size, ) + train_data_shape[1:],
    name='data_ph')
label_ph = tf.placeholder(
    tf.float32,
    shape=(args.batch_size, ) + train_label_shape[1:],
    name='label_ph')
objective = model(data_ph, label_ph, analytic_kl=True)
cluster_prob_ph = tf.placeholder(tf.float32, name='cluster_prob_ph')
sample = model.sample(args.batch_size, cluster_prob_ph)

optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate)
train_op = optimizer.minimize(objective)
base_epoch = tf.get_variable(
    'base_epoch', initializer=tf.zeros((), dtype=tf.int32))

verbose_ops_dict = dict()
verbose_ops_dict['distortion'] = model.distortion
verbose_ops_dict['rate'] = model.rate
verbose_ops_dict['elbo'] = model.elbo
verbose_ops_dict['iw_elbo'] = model.importance_weighted_elbo
verbose_ops_dict['posterior_logp'] = model.posterior_logp
verbose_ops_dict['classification_loss'] = model.classification_loss


def classification_rate(session, feed_dict_fn, batches):
    codes = []
    labels = []
    for _ in range(batches):
        c, l = session.run([model.latent_posterior_sample, label_ph],
                           feed_dict=feed_dict_fn())
        codes.append(c)
        labels.append(l)
    codes = np.squeeze(np.concatenate(codes, axis=1))
    labels = np.argmax(np.concatenate(labels), axis=1)
    return decision_tree.score(codes, labels)


saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
best_test_class_rate = 0.0
with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    base_epoch_val = session.run(base_epoch)
    if args.load:
        decision_tree_path = os.path.join(args.output_dir, 'decision_tree.pkl')
        if os.path.exists(decision_tree_path):
            decision_tree_path = os.path.join(args.output_dir,
                                              'decision_tree.pkl')
            with open(
                    os.path.join(args.output_dir, 'decision_tree.pkl'),
                    'rb') as dt_file:
                model._decision_tree = pickle.load(dt_file)
        saver.restore(session, os.path.join(args.output_dir, 'model_params'))
        base_epoch_val = session.run(base_epoch)

    if args.task == 'train':

        def train_setup_fn(session, epoch):
            if epoch % args.update_period == 1:
                model.update(
                    session,
                    label_ph,
                    args.update_samples * train_batches_per_epoch,
                    train_feed_dict_fn,
                    epoch,
                    output_dir=args.output_dir)

            class_rate = classification_rate(session, train_feed_dict_fn,
                                             train_batches_per_epoch)
            return {'class_rate': class_rate}

        def validate_setup_fn(session, epoch):
            class_rate = classification_rate(session, train_feed_dict_fn,
                                             train_batches_per_epoch)
            return {'class_rate': class_rate}

        train_dict = {
            'setup_fn': train_setup_fn,
            'steps_per_epoch': train_batches_per_epoch,
            'feed_dict_fn': train_feed_dict_fn
        }
        validate_dict = {
            'setup_fn': validate_setup_fn,
            'steps_per_epoch': test_batches_per_epoch,
            'feed_dict_fn': test_feed_dict_fn
        }

        def exit_fn(session, epoch, validate_dict):
            # save decoder samples of each class
            for c in range(10):
                cluster_probs = np.zeros([args.batch_size, 10], dtype=float)
                cluster_probs[:, c] = 1.
                generated_img = session.run(sample,
                                            {cluster_prob_ph: cluster_probs})
                filename = os.path.join(output_directory,
                                        'epoch{}_class{}.png'.format(epoch, c))
                plot.plot(filename, np.squeeze(generated_img), 4, 4)

            # decide whether to save model
            if not hasattr(exit_fn, 'best_class_rate'):
                exit_fn.best_class_rate = 0.0
            class_rate = validate_dict['class_rate']
            if class_rate > exit_fn.best_class_rate:
                print('Saving model parameters at {} test classification rate'.
                      format(validate_dict['class_rate']))
                session.run([tf.assign(base_epoch, epoch)])
                saver.save(session,
                           os.path.join(args.output_dir, 'model_params'))

                decision_tree_path = os.path.join(args.output_dir,
                                                  'decision_tree.pkl')
                with open(decision_tree_path, 'wb') as dt_file:
                    pickle.dump(model._decision_tree, dt_file)
                exit_fn.best_class_rate = class_rate
            # not currently doing early stopping
            return False

        util.train(session, args.epochs, train_dict, validate_dict, [train_op],
                   verbose_ops_dict, exit_fn)

    elif args.task == 'eval':
        # confusion matrix
        if args.conf_matr:
            eval_dict = util.run_epoch_ops(
                session,
                test_batches_per_epoch,
                verbose_ops_dict={
                    'labels': label_ph,
                    'codes': model.latent_posterior_sample
                },
                feed_dict_fn=test_feed_dict_fn)
            label_vals = np.stack(eval_dict['labels'])
            code_vals = np.concatenate(eval_dict['codes'], axis=0)
            prediction_vals = model._decision_tree.predict(code_vals)

            cnf_matrix = sklearn.metrics.confusion_matrix(
                np.argmax(label_vals, axis=1), prediction_vals)
            plot.plot_confusion_matrix(
                cnf_matrix,
                classes=[str(c) for c in range(10)],
                filename='test_confusion_matrix')

        # calculate mu for each node
        c_means, c_sds = model.aggregate_posterior_parameters(
            session, label_ph, train_batches_per_epoch, train_feed_dict_fn)

        # write routine to perform walks (discriminative and generative)

        # print(
        #     eval_cpvae.evaluation_spacing(
        #         np.zeros(10), np.ones(10), list(range(10))).shape)

        if args.viz_task == '2leaf':
            classes, dims = np.asarray(args.viz_classes), np.asarray(
                args.viz_dimension)
            latent_codes, filenames = eval_cpvae.two_leaf_visualization(
                c_means, c_sds, classes, dims, args.viz_steps)

            latent_code_ph = tf.placeholder(tf.float32)
            img_tensor = model.sample(
                len(latent_codes), None, latent_code=latent_code_ph)
            latent_codes = [a.astype(np.float32) for a in latent_codes]

            for latent_code, filename in zip(latent_codes, filenames):
                img_val = session.run(img_tensor,
                                      {latent_code_ph: latent_code})
                plot.plot_single(filename, img_val)
        elif args.viz_task == 'class_instance':
            pass
        elif args.viz_task == 'single_dim':
            dims = np.asarray(args.viz_dimension)
            latent_codes, filenames = eval_cpvae.mean_digit_dim_visualization(
                c_means, c_sds, dims, args.viz_steps)

            latent_code_ph = tf.placeholder(tf.float32)
            img_tensor = model.sample(
                len(latent_codes), None, latent_code=latent_code_ph)
            latent_codes = [a.astype(np.float32) for a in latent_codes]

            for latent_code, filename in zip(latent_codes, filenames):
                img_val = session.run(img_tensor,
                                      {latent_code_ph: latent_code})
                plot.plot_single(filename, img_val)
        #python sounds_deep/contrib/experiments/train_cpvae.py --task eval --output_dir cpvae_16177836/ --load --update_samples 1 --viz_task 2leaf --viz_classes 4 9 --viz_dimension 26
