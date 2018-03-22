# GORU-tensorflow

Gated Orthogonal Recurrent Unit

If you find this work useful, please cite [arXiv:1706.02761](https://arxiv.org/pdf/1706.02761.pdf).

## Installation

requires TensorFlow 1.2.0

## Usage

To use GORU in your model, simply copy [GORU.py](https://github.com/jingli9111/EUNN-tensorflow/blob/master/GORU.py).

Then you can use GORU in the same way you use built-in LSTM:
```
from GORU import GORUCell
cell = GORUCell(n_hidden, fft=True)
```
Args:
- `n_hidden`: `Integer`.
- `capacity`: `Optional`. `Integer`. Only works for tunable style.
- `fft`: `Optional`. `Bool`. If `True`, GORU is set to FFT style. Default is `True`.


## Example tasks for GORU
Two tasks for RNN in the paper are shown here. Use `-h` for more information

#### Copying Memory Task
```
python copying_task.py --model GORU
```

