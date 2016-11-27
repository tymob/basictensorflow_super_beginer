# -*- coding: UTF-8 -*-
# 必要なモジュール
import numpy as np
import tensorflow as tf
import pandas as pd

# TensorFlow でロジスティック回帰する

# 1. 学習したいモデルを記述する
# 入力変数と出力変数のプレースホルダを生成
x = tf.placeholder(tf.float32, shape=(None, 3), name="x")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# モデルパラメータ
a = tf.Variable(-10 * tf.ones((3, 1)), name="a")
b = tf.Variable(200., name="b")
# モデル式
u = tf.matmul(x,a) + b
y = tf.sigmoid(u)

# 2. 学習やテストに必要な関数を定義する
# 誤差関数(loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(u, y_))
# 最適化手段(最急降下法)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)



df =pd.read_csv("data/x.csv", header=None)
train_x = np.array(df)

df2 =pd.read_csv("data/y.csv", header=None)
train_y = np.array(df2)
print("x=", train_x)
print("y=", train_y)



# (2) セッションを準備し，変数を初期化
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# (3) 最急勾配法でパラメータ更新 (200回更新する)
for i in range(200):
    _, l, a_, b_ = sess.run([train_step, loss, a, b], feed_dict={x: train_x, y_: train_y})
    #if (i + 1) % 100 == 0:
    print("step=%3d, a1=%6.2f, a2=%6.2f, b=%6.2f, loss=%.2f" % (i + 1, a_[0], a_[1], b_, l))

# (4) 学習結果を出力
est_a, est_b = sess.run([a, b], feed_dict={x: train_x, y_: train_y})
print("Estimated: a1=%6.2f, a2=%6.2f, b=%6.2f" % (est_a[0], est_a[1], est_b))

# 4. 新しいデータに対して予測する
# (1) 新しいデータを用意


df3 = pd.read_csv("data/x_test.csv", header=None)
new_x = np.array(df3)

df4 = pd.read_csv("data/y_test.csv",header=None)
y_real = np.array(df4)

# (2) 学習結果をつかって，予測実施
new_y = sess.run(y, feed_dict={x: new_x})
correct = 0
height,width=new_y.shape
for i in range(height):
  if y_real[i][0]-new_y[i][0]<0.01:
      correct=correct+1
print("aculacy:" + str(correct/height*100) + "%")

# 5. 後片付け
# セッションを閉じる
sess.close()
