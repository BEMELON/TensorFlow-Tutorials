# 모델을 저장하고 재사용하는 방법을 익혀봅니다.

import tensorflow as tf
import numpy as np


data = np.loadtxt('./data.csv', delimiter=',',
                  unpack=True, dtype='float32')
#uppack = True  는 np.transpose 처럼 Transpose 행렬로 바꾸어준다.
# Transpose행렬은, 행렬 연산을 효율적으로 하기 위해 변환합니다.
# 털, 날개, 기타, 포유류, 조류
# x_data = 0, 1
# y_data = 2, 3, 4
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])


#########
# 신경망 모델 구성
######
# 학습에 직접적으로 사용하지 않고 학습 횟수에 따라 단순히 증가시킬 변수를 만듭니다.
global_step = tf.Variable(0, trainable=False, name='global_step')

# 편향 없이 가중치만 사용한 모델, 계층 하나 증가
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# W1, 가중치를 [2, 10] 형태로 -1 ~ 1 사이의 정규분포 값을 따른다.
# L1, 첫번째 뉴런의 학습값
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

# W2, 첫번째 뉴런의 학습값은 10열을가지게 됨으로, 10행을 설정한다.
# L2, 두번째 뉴런의 학습값
W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))

# W3, 두번째 뉴런의 학습값이 20열을 가지게 됨으로, 20행을 설정 한다.
# model, 최종 뉴런의 학습값 3열을 하지게 된다.
W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
model = tf.matmul(L2, W3)

# cost, 손실함수(비용함수)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# global_step로 넘겨준 변수를, 학습용 변수들을 최적화 할 때 마다 학습 횟수를 하나씩 증가시킵니다.
train_op = optimizer.minimize(cost, global_step=global_step)


#########
# 신경망 모델 학습
######
sess = tf.Session()
# 모델을 저장하고 불러오는 API를 초기화합니다.
# global_variables 함수를 통해 앞서 정의하였던 변수들을 저장하거나 불러올 변수들로 설정합니다.
saver = tf.train.Saver(tf.global_variables())

# ./model 폴더에 저장된 값을 가져온다.
ckpt = tf.train.get_checkpoint_state('./model')

# 만약 저장된 값이 있다면, restore 함수로 불러온다.
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

# 최적화 진행
for step in range(2):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print('Step: %d, ' % sess.run(global_step),
          'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# 최적화가 끝난 뒤, 변수를 저장합니다.
# global_step 값은 저장되는 이름에 추가로 붙게 되며, 텐서 변수 혹은 값을 넣어줄 수 있습니다.
saver.save(sess, './model/dnn.ckpt', global_step=global_step)

#########
# 결과 확인
# 0: 기타 1: 포유류, 2: 조류
######
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
