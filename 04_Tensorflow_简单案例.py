import tensorflow as tf
import numpy as np
#使用numpy生成100个随机点
x_data=np.random.rand(100)
#一条直线，斜率0.1，截距0.2
y_data=x_data*0.1+0.2

#构造一个线性模型,优化b、k使得接近于上面样本点分布，k、b初始为0.0
b=tf.Variable(0.)
k=tf.Variable(0.)
y=k*x_data+b

#二次代价函数,y_data:真实值；y:预测值，目的是求使得误差平方均值最小的y
loss=tf.reduce_mean(tf.square(y_data-y))
#定义一个梯度下降法来进行训练的优化器,学习率：0.2
optimizer=tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数,预测值y越接近于y_data，k、b值越接近已知值
train=optimizer.minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        #每20次打印次数及k、b值
        if(step%20==0):
            print(step,sess.run([k,b]))
