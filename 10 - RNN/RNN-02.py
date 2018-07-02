import tensorflow as tf
import numpy as np


char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

# one-hot 인코딩 사용 및 디코딩을 하기 위해 연관 배열을 만듭니다.
# {'a': 0, 'b': 1, 'c': 2, ..., 'j': 9, 'k', 10, ...}
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 다음 배열은 입력값과 출력값으로 다음처럼 사용할 것 입니다.
# wor -> X, d -> Y
# woo -> X, d -> Y
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']

def make_batch(seq_data):
    input_batch = []
    target_batch - []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[:-1]]
        target = num_dic[seq[-1]]
        input_batch.append(np.eye(dic_len)[input])
        target_batch.append([target])

    return input_batch, target_batch


learning_rate = 0.01
n_hidden = 128
total_epoch = 30

n_step = 3
n_input = n_class = dic_len

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None])

W = tf.Variable(tf.random_normal([n_hidden, n_class)])
b = tf.Variable(tf.random_normal([n_class]))

cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob = 0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype = tf.float32)

outputs = tf.transpose(outputs, [1,0,2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

cost =
tf.reduce_mean(tf.n.sparse_softmax_cross_entropy_with_logits(logits=model,
                                                             labes=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Sesion()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost], feed_dict ={X:input_batch,
                                                      Y;target_batch})

prediction = tf.cast(tf.argmax(model, 1), tf.int32)
prediction_check = tf.equl(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

