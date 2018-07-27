import sacred
import tensorflow as tf

import util

train_ingredient = sacred.Ingredient('train')


@train_ingredient.config
def cfg():
    epochs = 250


@train_ingredient.capture
def run_training(write_update_fn,
                 train_op,
                 train_gen,
                 input_ph,
                 verbose_ops_dict,
                 batches_per_epoch,
                 epochs):
    out_dicts = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            out_dict = util.run_epoch_ops(
                session,
                batches_per_epoch,
                verbose_ops_dict=verbose_ops_dict,
                silent_ops=[train_op],
                feed_dict_fn=lambda: {input_ph: next(train_gen)},
                verbose=False)
            out_dicts.append(out_dict)
            write_update_fn(epoch, out_dict)
    return out_dicts