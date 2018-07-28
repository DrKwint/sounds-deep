import sacred
import sounds_deep.contrib.data.data as data

load_data_ingredient = sacred.Ingredient('dataset')


@load_data_ingredient.config
def cfg():
    dataset_name = 'mnist'
    batch_size = 32


@load_data_ingredient.capture
def load_data(dataset_name, batch_size):
    if dataset_name == 'mnist':
        train_data, _, test_data, _ = data.load_mnist('./data/')
    elif dataset_name == 'cifar10':
        train_data, _, test_data, _ = data.load_cifar10('./data/')
    else:
        assert False, "Must specify a valid dataset_name"
    data_shape = (batch_size, ) + train_data.shape[1:]
    batches_per_epoch = train_data.shape[0] // batch_size
    train_gen = data.data_generator(train_data, batch_size)
    test_gen = data.data_generator(test_data, batch_size)
    return train_gen, test_gen, batches_per_epoch, data_shape
