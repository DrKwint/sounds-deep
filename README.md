# sounds-deep

A library built on top of TensorFlow, Sonnet, and Sacred to faciliate deep learning research.

Version: 0.1.0.dev1

## Requirements

- Python 3.6
- virtualenv is strongly recommended
- CUDA 9.0 and cuDNN (if enabling GPU)


## Installation

Clone to repository, `cd` into the root directory, activate a virtual environment (optional), and run

```
pip install -r requirements.txt
pip uninstall tensorflow && pip install tensorflow-gpu # run this line if you want to enable GPU
```

Setup is more complicated than this on `Crane`, but talk to Ellie directly about it because we don't have an automated process nailed down yet.

Documentation on `master` can be found at [sounds-deep.readthedocs.io](https://sounds-deep.readthedocs.io/en/latest/) or can be built by running
```
./docs/build_scrip.sh
```
and pointing a browser at `./docs_build/index.html`

Because the use of this package is expected to stay within the lab right now, you can find me in person or on slack with any questions.

## Usage

```
import sounds_deep as sd
```

## Modules:
- `contrib.data`: Easy downloading of standard datasets and loading for TensorFlow
- `contrib.distributions`: Handles distributions in ways not done in `tf.contrib.distributions` (use `tfd` when possible)
- `contrib.experiments`: Executable files with command line interfaces which train a model
- `contrib.models`: `Sonnet` modules implementing entire model frameworks
- `contrib.parameterized_distributions`: Distributions with parameters baked in
- `contrib.sacred_ingredients`: Classes inheriting from `Sacred.Ingredient`

## Contributing

Everyone in the lab is invited to contribute code pertaining to Sonnet, TensorFlow, or deep learning/machine learning with Python.

### Author

Eleanor Quint, a Ph.D. student in the computer science and engineering department at the University of Nebraska-Lincoln
