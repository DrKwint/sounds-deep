# sounds-deep

A library built on top of TensorFlow, Sonnet, and Sacred to faciliate deep learning research.

Version: 0.1.0.dev1

## Usage

```
import sounds_deep as sd
```

## Modules:
- `contrib.data`: Easy downloading of standard datasets and loading for TensorFlow
- `contrib.distributions`: Handles distributions in ways not done in `tf.contrib.distributions` (use `tfd` when possible)
- `contrib.experiments`: Executable files with command line interfaces which train a model
- `contrib.models`: `Sonnet` modules implementing entire model frameworks
- `contrib.sacred_ingredients`: Classes inheriting from `Sacred.Ingredient`


### Author

Eleanor Quint, a Ph.D. student in the computer science and engineering department at the University of Nebraska-Lincoln
