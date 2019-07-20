import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#生成(-0.5,0.5)之间均匀的200个点,[]使得为二维数据，200行1列
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
#生成随机值噪音，维度为和x_data一样
noise=np.random.normal(0,0.02,x_data.shape)
#大体U型
y_data=np.square(x_data)+noise

#定义两个placeholder,[None,1]:行不确定，1列
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#构建神经网络：输入层一个神经元，中间层10个神经元，输出层1个神经元
#定义神经网络中间层
#权值随机数赋值[1,10]:一行10列
Weights_L1=tf.Variable(tf.random_normal([1,10]))
#偏执值初始化为0
biases_L1=tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1
#激活函数,作为中间层的输出，同时作为输出层的输入
L1=tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络输出层
Weights_L2=tf.Variable(tf.random_normal([10,1]))
#偏执值初始化为0
biases_L2=tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2
prediction=tf.nn.tanh(Wx_plus_b_L2)

#构建代价函数(损失函数)
loss=tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法 学习率：0.1
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        #使用样本点x_data、y_data训练2000次，确定各层权重参数及偏置值
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    #获得预测值，将x_data传入并获得预测值prediction_value
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
    #画图查看预测结果
    plt.figure()
    #散点图打印样本点
    plt.scatter(x_data,y_data)
    #打印x_data及对应预测值
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()



