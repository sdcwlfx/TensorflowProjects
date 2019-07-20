# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 21:42:02 2018

@author: asus
"""
#通过变量实现神经网络的参数并实现前向传播
import tensorflow as tf

#biases=tf.Variable(tf.zeros([3]))
#with tf.Session() as sess:
#    print(biases)


#声明两个权重矩阵变量，通过seed设定随机种子保证每次运行得到的结果是一样的
w1=tf.Variable(tf.random_normal((2,3),stddev=1,seed=1))
w2=tf.Variable(tf.random_normal((3,1),stddev=1,seed=1))

#将输入的特征向量定义为1x2常量矩阵
x=tf.constant([[0.7,0.9]])

#利用前向传播算法获得神经网络的输出
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

sess=tf.Session()
 
#初始化权重矩阵
#sess.run(w1.initializer)
#sess.run(w2.initializer)
#一次性初始化所有变量
init_op=tf.global_variables_initializer()
sess.run(init_op)
    
print(sess.run(y))
sess.close()

