"""Essential utility functions"""
import tqdm
import numpy as np


def run_epoch_ops(session,
                  steps_per_epoch,
                  verbose_ops_dict=None,
                  silent_ops=None,
                  feed_dict_fn=lambda: None,
                  verbose=False):
    """
    Args:
        session (tf.Session): Session with tf.Graph containing the operations
            passed in `verbose_ops_dict` and `silent_ops`
        steps_per_epoch (int): number of times to run operations
        verbose_ops_dict (dict): strings to tf operations whose values will be
            returned
        silent_ops (list): list of tf operations to run, ignoring output
        feed_dict_fn (callable): called to retrieve the feed_dict
            (dict of tf.placeholder to np.array)
        verbose (bool): whether to use tqdm progressbar on stdout
    Return:
        dict of str to np.array parallel to the verbose_ops_dict
    """
    if verbose_ops_dict is None: verbose_ops_dict = dict()
    if silent_ops is None: silent_ops = list()
    verbose_vals = {k: [] for k, v in verbose_ops_dict.items()}
    if verbose:
        iterable = tqdm.tqdm(list(range(steps_per_epoch)))
    else:
        iterable = list(range(steps_per_epoch))

    for _ in iterable:
        out = session.run(
            [silent_ops, verbose_ops_dict], feed_dict=feed_dict_fn())[1]
        verbose_vals = {
            k: v + [np.array(out[k])]
            for k, v in verbose_vals.items()
        }

    return {k: np.stack(v) for k, v in verbose_vals.items()}
