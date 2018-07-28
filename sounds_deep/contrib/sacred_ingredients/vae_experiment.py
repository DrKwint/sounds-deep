import sacred
import numpy as np

from sounds_deep.contrib.sacred_ingredients.load_data_ingredient import load_data_ingredient, load_data
from sounds_deep.contrib.sacred_ingredients.define_vae_ingredient import define_vae_ingredient, define_model, write_verbose_ops
from sounds_deep.contrib.sacred_ingredients.train_ingredient import train_ingredient, run_training

ex = sacred.Experiment(
    'vae_experiment',
    ingredients=[
        load_data_ingredient, define_vae_ingredient, train_ingredient
    ])

ex.observers.append(sacred.observers.TinyDbObserver.create('vae_results'))


@ex.automain
def run():
    train_gen, _, batches_per_epoch, data_shape = load_data()
    _, input_ph, train_op, verbose_ops_dict = define_model(data_shape)
    output = run_training(write_verbose_ops, train_op, train_gen, input_ph,
                          verbose_ops_dict, batches_per_epoch)

    max_mean = lambda out_dict_list, metric: float(np.max([np.mean(out_dict[metric]) for out_dict in out_dict_list]))

    return {
        'best_elbo': max_mean(output, 'elbo'),
        'best_iw_elbo': max_mean(output, 'iw_elbo')
    }
