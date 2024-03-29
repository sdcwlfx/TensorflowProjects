# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 22:59:25 2018

@author: asus
"""

import tensorflow as tf
from numpy.random import RandomState

#定义训练数据的大小
batch_size=8

#定义神经网络的参数
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#定义损失函数和反向传播的算法
y=tf.sigmoid(y)
cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))
+(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm=RandomState(1)
dataset_size=128
X=rdm.rand(dataset_size,2)

Y=[[int(x1+x2<1)] for (x1,x2) in X]

with tf.Session() as sess:
    #初始化变量
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    
    print sess.run(w1)
    print sess.run(w2)
    
    
--['    STEPS=5000
    
    