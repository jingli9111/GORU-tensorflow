from __future__ import absolute_import
from __future__ import division
from __future__	import print_function

import numpy as np
import argparse, os
import tensorflow as tf

from EURNN import EURNNCell
from GORU import GORUCell


def noise_data(T, n_data, n_sequence):
	seq = np.random.randint(1, high=10, size=(n_data, n_sequence))
	zeros1 = np.zeros((n_data, T))

	for i in range(n_data):
		ind = np.random.choice(T, n_sequence)
		ind.sort()
		zeros1[i][ind] = seq[i]

	zeros2 = np.zeros((n_data, T+1))
	marker = 10 * np.ones((n_data, 1))
	zeros3 = np.zeros((n_data, n_sequence))

	x = np.concatenate((zeros1, marker, zeros3), axis=1).astype('int32')
	y = seq


	return x, y

def main(model, T, n_iter, n_batch, n_hidden, capacity, fft):

	# --- Set data params ----------------
	n_input = 11
	n_output = 10
	n_sequence = 10
	n_train = n_iter * n_batch
	n_test = n_batch

	n_steps = T+11
	n_classes = 10





	# --- Create graph and compute gradients ----------------------
	x = tf.placeholder("int32", [None, n_steps])
	y = tf.placeholder("int64", [None, n_sequence])
	
	input_data = tf.one_hot(x, n_input, dtype=tf.float32)



	# --- Input to hidden layer ----------------------
	if model == "LSTM":
		cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
		hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "GRU":
		cell = tf.nn.rnn_cell.GRUCell(n_hidden)
		hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "EUNN":
		cell = EUNNCell(n_hidden, capacity, fft, comp)
		if comp:
			hidden_out_comp, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.complex64)
			hidden_out = tf.real(hidden_out_comp)
		else:
			hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "GORU":
		cell = GORUCell(n_hidden, capacity, fft)
		hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)

	# --- Hidden Layer to Output ----------------------

	V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_classes], dtype=tf.float32)
	V_bias = tf.get_variable("V_bias", shape=[n_classes], dtype=tf.float32)

	hidden_out_list = tf.unstack(hidden_out, axis=1)[-n_sequence:]
	temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list])
	output_data = tf.nn.bias_add(tf.transpose(temp_out, [1,0,2]), V_bias) 

	# --- evaluate process ----------------------
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_data, labels=y))
	correct_pred = tf.equal(tf.argmax(output_data, 2), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	# --- Initialization ----------------------
	optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9).minimize(cost)
	init = tf.global_variables_initializer()


	# --- Training Loop ----------------------

	step = 0
	with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)) as sess:

		sess.run(init)



		while step < n_iter:
			batch_x, batch_y = noise_data(T, n_batch, n_sequence)
			
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

			acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})

			print("Iter " + str(step) + ", Minibatch Loss= " + \
				  "{:.6f}".format(loss) + ", Training Accuracy= " + \
				  "{:.5f}".format(acc))


			step += 1


		print("Optimization Finished!")


		
		# --- test ----------------------
		test_x, test_y = noise_data(T, n_test, n_sequence)

		test_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
		test_loss = sess.run(cost, feed_dict={x: test_x, y: test_y})
		print("Test result: Loss= " + "{:.6f}".format(test_loss) + \
					", Accuracy= " + "{:.5f}".format(test_acc))


if __name__=="__main__":
	parser = argparse.ArgumentParser(
		description="Denoise Task")
	parser.add_argument("model", default='LSTM', help='Model name: LSTM, EUNN, GRU, GORU')
	parser.add_argument('-T', type=int, default=200, help='Information sequence length')
	parser.add_argument('--n_iter', '-I', type=int, default=5000, help='training iteration number')
	parser.add_argument('--n_batch', '-B', type=int, default=128, help='batch size')
	parser.add_argument('--n_hidden', '-H', type=int, default=128, help='hidden layer size')
	parser.add_argument('--capacity', '-L', type=int, default=2, help='Tunable style capacity, default value is 2')
	parser.add_argument('--comp', '-C', type=str, default="False", help='Complex domain or Real domain, only for EUNN. Default is False: complex domain')
	parser.add_argument('--fft', '-F', type=str, default="False", help='fft style, only for EUNN and GORU, default is False: tunable style')

	args = parser.parse_args()
	dict = vars(args)

	for i in dict:
		if (dict[i]=="False"):
			dict[i] = False
		elif dict[i]=="True":
			dict[i] = True
		
	kwargs = {	
				'model': dict['model'],
				'T': dict['T'],
				'n_iter': dict['n_iter'],
			  	'n_batch': dict['n_batch'],
			  	'n_hidden': dict['n_hidden'],
			  	'capacity': dict['capacity'],
			  	'fft': dict['fft'],
			}

	main(**kwargs)
