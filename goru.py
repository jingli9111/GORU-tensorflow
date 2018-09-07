from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import numpy as np
from tensorflow.python.ops import rnn_cell_impl


def modrelu(inputs, bias):
    """
    modReLU activation function
    """

    norm = tf.abs(inputs) + 0.00001
    biased_norm = norm + bias
    magnitude = tf.nn.relu(biased_norm)
    phase = tf.sign(inputs)
       
    return phase * magnitude

def generate_index_tunable(s, L):
    """
    generate the index lists for goru to prepare orthogonal matrices 
    and perform efficient rotations
    This function works for tunable case
    """    
    ind1 = list(range(s))
    ind2 = list(range(s))

    for i in range(s):
        if i%2 == 1:
            ind1[i] = ind1[i] - 1
            if i == s -1:
                continue
            else:
                ind2[i] = ind2[i] + 1
        else:
            ind1[i] = ind1[i] + 1
            if i == 0: 
                continue
            else:
                ind2[i] = ind2[i] - 1

    ind_exe = [ind1, ind2] * int(L/2)

    ind3 = []
    ind4 = []

    for i in range(int(s/2)):
        ind3.append(i)
        ind3.append(i + int(s/2))

    ind4.append(0)
    for i in range(int(s/2) - 1):
        ind4.append(i + 1)
        ind4.append(i + int(s/2))
    ind4.append(s - 1)

    ind_param = [ind3, ind4]

    return ind_exe, ind_param


def generate_index_fft(s):
    """
    generate the index lists for goru to prepare orthogonal matrices 
    and perform efficient rotations
    This function works for fft case
    """      
    def ind_s(k):
        if k==0:
            return np.array([[1,0]])
        else:
            temp = np.array(range(2**k))
            list0 = [np.append(temp + 2**k, temp)]
            list1 = ind_s(k-1)
            for i in range(k):
                list0.append(np.append(list1[i],list1[i] + 2**k))
            return list0

    t = ind_s(int(math.log(s/2, 2)))

    ind_exe = []
    for i in range(int(math.log(s, 2))):
        ind_exe.append(tf.constant(t[i]))

    ind_param = []
    for i in range(int(math.log(s, 2))):
        ind = np.array([])
        for j in range(2**i):
            ind = np.append(ind, np.array(range(0, s, 2**i)) + j).astype(np.int32)

        ind_param.append(tf.constant(ind))
    
    return ind_exe, ind_param


def fft_param(num_units):
    
    phase_init = tf.random_uniform_initializer(-3.14, 3.14)
    capacity = int(math.log(num_units, 2))

    theta = tf.get_variable("theta", [capacity, num_units//2], 
        initializer=phase_init)
    cos_theta = tf.cos(theta)
    sin_theta = tf.sin(theta)
        
    cos_list = tf.concat([cos_theta, cos_theta], axis=1)
    sin_list = tf.concat([sin_theta, -sin_theta], axis=1)

        
    ind_exe, index_fft = generate_index_fft(num_units)

    v1 = tf.stack([tf.gather(cos_list[i,:], index_fft[i]) for i in range(capacity)])
    v2 = tf.stack([tf.gather(sin_list[i,:], index_fft[i]) for i in range(capacity)])


    return v1, v2, ind_exe

def tunable_param(num_units, capacity):

    capacity_A = int(capacity//2)
    capacity_B = capacity - capacity_A
    phase_init = tf.random_uniform_initializer(-3.14, 3.14)

    theta_A = tf.get_variable("theta_A", [capacity_A, num_units//2], 
        initializer=phase_init)
    cos_theta_A = tf.cos(theta_A)
    sin_theta_A = tf.sin(theta_A)

    cos_list_A = tf.concat([cos_theta_A, cos_theta_A], axis=1)
    sin_list_A = tf.concat([sin_theta_A, -sin_theta_A], axis=1)         


    theta_B = tf.get_variable("theta_B", [capacity_B, num_units//2 - 1], 
        initializer=phase_init)
    cos_theta_B = tf.cos(theta_B)
    sin_theta_B = tf.sin(theta_B)

    cos_list_B = tf.concat([tf.ones([capacity_B, 1]), cos_theta_B, 
        cos_theta_B, tf.ones([capacity_B, 1])], axis=1)
    sin_list_B = tf.concat([tf.zeros([capacity_B, 1]), sin_theta_B, 
        - sin_theta_B, tf.zeros([capacity_B, 1])], axis=1)


    ind_exe, [index_A, index_B] = generate_index_tunable(num_units, capacity)
 

    diag_list_A = tf.gather(cos_list_A, index_A, axis=1)
    off_list_A = tf.gather(sin_list_A, index_A, axis=1)
    diag_list_B = tf.gather(cos_list_B, index_B, axis=1)
    off_list_B = tf.gather(sin_list_B, index_B, axis=1)


    v1 = tf.reshape(tf.concat([diag_list_A, diag_list_B], axis=1), [capacity, num_units])
    v2 = tf.reshape(tf.concat([off_list_A, off_list_B], axis=1), [capacity, num_units])


    return v1, v2, ind_exe


class GORUCell(rnn_cell_impl.RNNCell):
    """Gated Orthogonal Recurrent Unit Cell
    
    The implementation is based on: 

    http://arxiv.org/abs/1706.02761.

    """

    def __init__(self, 
                num_units, 
                capacity=2, 
                fft=True, 
                activation=modrelu,
                reuse=None):
        """Initializes the GORU cell.
        Args:
          num_units: int, The number of units in the GORU cell.
          capacity: int, The capacity of the orthogonal matrix for tunable
            case.
          fft: bool, default false, whether to use fft style 
          architecture or tunable style.
        """



        super(GORUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation
        self._capacity = capacity
        self._fft = fft

        if self._capacity > self._num_units:
            raise ValueError("Do not set capacity larger than hidden size, it is redundant")

        if self._fft:
            if math.log(self._num_units, 2) % 1 != 0: 
                raise ValueError("FFT style only supports power of 2 of hidden size")
        else:
            if self._num_units % 2 != 0:
                raise ValueError("Tunable style only supports even number of hidden size")

            if self._capacity % 2 != 0:
                raise ValueError("Tunable style only supports even number of capacity")



        if self._fft:
            self._capacity = int(math.log(self._num_units, 2))
            self._v1, self._v2, self._ind = fft_param(self._num_units)
        else:
            self._v1, self._v2, self._ind = tunable_param(self._num_units, self._capacity)


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


    def loop(self, h):
        for i in range(self._capacity):
            diag = h * self._v1[i,:]
            off = h * self._v2[i,:]
            h = diag + tf.gather(off, self._ind[i], axis=1)

        return h


    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "goru_cell"):

            inputs_size = inputs.get_shape()[-1]

            input_matrix_init = tf.random_uniform_initializer(-0.01, 0.01)
            bias_init = tf.constant_initializer(2.)
            mod_bias_init = tf.constant_initializer(0.01)
            
            U = tf.get_variable("U", [inputs_size, self._num_units * 3], dtype=tf.float32, initializer = input_matrix_init)
            Ux = tf.matmul(inputs, U)
            U_cx, U_rx, U_gx = tf.split(Ux, 3, axis=1)

            W_r = tf.get_variable("W_r", [self._num_units, self._num_units], dtype=tf.float32, initializer = input_matrix_init)
            W_g = tf.get_variable("W_g", [self._num_units, self._num_units], dtype=tf.float32, initializer = input_matrix_init)
            W_rh = tf.matmul(state, W_r)
            W_gh = tf.matmul(state, W_g)

            bias_r = tf.get_variable("bias_r", [self._num_units], dtype=tf.float32, initializer = bias_init)
            bias_g = tf.get_variable("bias_g", [self._num_units], dtype=tf.float32)
            bias_c = tf.get_variable("bias_c", [self._num_units], dtype=tf.float32, initializer = mod_bias_init)
        

            r_tmp = U_rx + W_rh + bias_r
            g_tmp = U_gx + W_gh + bias_g
            r = tf.sigmoid(r_tmp)
            g = tf.sigmoid(g_tmp)

            Unitaryh = self.loop(state)
            c = self._activation(r * Unitaryh + U_cx, bias_c)
            new_state = tf.multiply(g, state) +  tf.multiply(1 - g, c)

        return new_state, new_state






