# GORU-tensorflow

Gated Orthogonal Recurrent Unit

This model combines gating mechanism and orthogonal RNN approach. It solves forgetting problem and long-term dependency.

If you find this work useful, please cite [arXiv:1706.02761](https://arxiv.org/pdf/1706.02761.pdf).

## Installation

requires TensorFlow 1.2.0

## Usage

To use GORU in your model, simply copy [goru.py](https://github.com/jingli9111/GORU-tensorflow/blob/master/goru.py).

Then you can use GORU in the same way you use built-in LSTM:
```
from goru import GORUCell
cell = GORUCell(hidden_size, capacity, fft)
```
Args:
- `hidden_size`: `Integer`.
- `capacity`: `Optional`. `Integer`. Only works for tunable style.
- `fft`: `Optional`. `Bool`. If `True`, GORU is set to FFT style. Default is `True`.


## Example tasks for GORU
Code for copying task is here as example. The other experiment code will appear soon!

#### Copying Memory Task
```
python copying_task.py --model GORU
```
