#保存训练好的模型参数(权重矩阵、偏置值等)到指定文件夹中
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集,one_hot:将标签转换为只有一位为1，其它为0，会自动从网上下载数据集到当前目录
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次的大小
batch_size=100
#计算一共有多少个批次（整除）
n_batch=mnist.train.num_examples//batch_size

#该神经网络只输入层和输出层，输入层包含784个神经元，输出层包含10个神经元
#定义两个placeholder,将28*28数字图片偏平为规格为784的向量
x=tf.placeholder(tf.float32,[None,784])
#标签结果
y=tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
#权值初始化为0,  784x10
W=tf.Variable(tf.zeros([784,10]))
#偏置值
b=tf.Variable(tf.zeros([10]))
#softmax将输出转化为概率值
prediction=tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数,差的平方的平均值
#loss=tf.reduce_mean(tf.square(y-prediction))
#交叉熵代价函数的平均值
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#梯度下降法
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

#tf.argmax(y,1)返回1的位置(真实值)，tf.argmax(prediction,1)(预测值)返回概率值最大的位置，比较位置是否相等，若想等返回true，不等返回false，存放在布尔列表中
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#tf.argmax()返回一维张量中最大值的位置
#求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # 对所有图片迭代11次
    for epoch in range(11):
        #对所有图片分批训练一次
        for batch in range(n_batch):
            #获取一批(100个)样本图片,batch_xs:图片信息，batch_ys:图片标签
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            #利用训练图片信息及对应标签，梯度下降法训练模型，得到权重W及b
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        #利用测试集进行测试该迭代时模型的准确率
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        #打印迭代次数及对应准确率
        print("Iter "+str(epoch)+",Testing Accuracy "+str(acc))

    #保存训练好的模型(权值、偏置值等信息)
    saver.save(sess,'net/my_net.ckpt')



