import os
import csv

import tensorflow as tf
import sounds_deep as sd

tf.logging.set_verbosity(tf.logging.INFO)


class Experiment(object):
    """
    Object designed to handle training a model with the given model
    specification, hyperparameters, and data. Should make running experiments
    easier/faster by handling the programmatic details and by gathering the
    architecture, hyperparameters, and dataset all in one place.
    """

    def __init__(self,
                 name,
                 model_fn,
                 model_args_dict,
                 data_dict,
                 checkpoint_dir='./'):
        """
        Args:
            - model_fn: fn that accepts a session and returns a model with
                initialized graph
        """
        self.name = name
        self.model_fn = model_fn
        self.model_args_dict = model_args_dict
        self.data_dict = data_dict

        checkpoint_suffix = '-'.join(
            filter(lambda s: not s[-1] == '>',
                   map(lambda (k, v): '{}_{}'.format(k, v),
                       zip(model_args_dict.keys(), model_args_dict.values()))))
        checkpoint_suffix = self.name + '-' + checkpoint_suffix
        print(checkpoint_suffix)
        self.checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_suffix)

    def run_training(self,
                     max_steps,
                     batch_size,
                     learning_rate,
                     optimizer=tf.train.AdamOptimizer,
                     verbose=True):
        """
        Args:
            - steps (int)
        """
        dataset = sd.MemBackedDataset(self.data_dict, shuffle=True)
        steps_per_epoch = dataset.calc_batches_per_epoch(batch_size)
        input_tensors = dataset.get_batch_tensors(batch_size).values()
        model = self.model_fn(self.name, self.model_args_dict)(*input_tensors)

        global_step = tf.contrib.framework.get_or_create_global_step()
        train_op = optimizer(learning_rate).minimize(
            model.loss, global_step=global_step)

        tf.summary.scalar("loss", model.loss)
        merged_summary_op = tf.summary.merge_all()

        # METRICS
        metrics_csv_filename = os.path.join(self.checkpoint_dir, 'metrics.csv')
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.isfile(metrics_csv_filename):
            csvfile = open(metrics_csv_filename, 'a')
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['step'] + model.losses.keys())
            csvfile.close()

        with tf.name_scope('metrics') as scope:  # ignore vscode about this
            # add any metrics here, appending their update to the metrics update
            # further, remember to add the correct update and resets below
            # reset will use the calculate_batches_per_epoch of the dataset
            # https://www.tensorflow.org/api_guides/python/contrib.metrics#Metric_Ops_
            # https://github.com/tensorflow/tensorflow/issues/4814
            for name, loss in model.losses.iteritems():
                mean, update_op = tf.contrib.metrics.streaming_mean(
                    loss, name='{}_loss_metric'.format(name))
                tf.add_to_collection('metrics', mean)
                tf.add_to_collection('metric_update_ops', update_op)
        metrics_reset_op = [tf.initialize_variables(tf.local_variables())]

        server = tf.train.Server.create_local_server()
        hooks = [tf.train.StopAtStepHook(last_step=max_steps)]
        my_scaffold = tf.train.Scaffold()
        my_scaffold.saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=0.25)
        mon_sess = tf.train.MonitoredTrainingSession(
            master=server.target,
            checkpoint_dir=self.checkpoint_dir,
            scaffold=my_scaffold,
            hooks=hooks,
            save_summaries_steps=steps_per_epoch,
            log_step_count_steps=steps_per_epoch)

        csvfile = open(metrics_csv_filename, 'a')
        csvwriter = csv.writer(csvfile)
        while not mon_sess.should_stop():
            out = mon_sess.run([
                train_op, merged_summary_op,
                tf.contrib.framework.get_or_create_global_step()
            ] + tf.get_collection('metric_update_ops'))[2:]
            if out[0] % steps_per_epoch == 0:
                csvwriter.writerow(out)
                mon_sess.run(metrics_reset_op)
        csvfile.close()
        mon_sess.close()
