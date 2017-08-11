import sonnet as snt
import tensorflow as tf
from keras.layers import Dense
from tensorflow.examples.tutorials.mnist import input_data

import sounds_deep_new as sd

# set flags
flags = tf.app.flags
flags.DEFINE_integer('latent_dim', 2, 'VAE latent dimension')

flags.DEFINE_integer('max_steps', 1000, '')
flags.DEFINE_integer('batch_size', 16, 'minibatch size')
flags.DEFINE_float('learning_rate', 1e-3, '')
FLAGS = flags.FLAGS

# build model
model_args = {
    'latent_dim': FLAGS.latent_dim,
    'encoder': Dense(2 * FLAGS.latent_dim, activation='tanh'),
    'decoder': Dense(784, activation='tanh'),
}
train_args = {
    'max_steps': FLAGS.max_steps,
    'batch_size': FLAGS.batch_size,
    'learning_rate': FLAGS.learning_rate,
}

# inject data into MemBackedDataset

# train model
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
data_dict = {'train_input_tensor': mnist.train.images} #'labels': mnist.test.labels}

for ld in [2]:
    print("!!!! LATENT DIM IS {} !!!!".format(ld))
    model_args['latent_dim'] = ld
    experiment = sd.Experiment('VAE', sd.VAE, model_args, data_dict, checkpoint_dir='./vae_checkpoint/')
    experiment.run_training(max_steps=100000, batch_size=16, learning_rate=1e-3)
    experiment.run_training(max_steps=200000, batch_size=32, learning_rate=1e-4)
