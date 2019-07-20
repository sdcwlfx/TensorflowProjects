import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集,one_hot:将标签转换为只有一位为1，其它为0，会自动从网上下载数据集到当前目录
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次的大小
batch_size=100
#计算一共有多少个批次（整除）
n_batch=mnist.train.num_examples//batch_size

#该神经网络包含输入层、中间层、输出层，输入层包含784个神经元，输出层包含10个神经元
#定义两个placeholder,将28*28数字图片偏平为规格为784的向量
x=tf.placeholder(tf.float32,[None,784])
#标签结果
y=tf.placeholder(tf.float32,[None,10])
#设置神经元保活概率-->Dropout防止过拟合
keep_prob=tf.placeholder(tf.float32)
#学习率变量,初始为0.001
lr=tf.Variable(0.001,dtype=tf.float32)

#创建一个简单的神经网络
#一般权值初始化不能为0,  784x10,输入层784个神经元
W1=tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
#偏置值初始化为0.1
b1=tf.Variable(tf.zeros([500])+0.1)
L1=tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop=tf.nn.dropout(L1,keep_prob)

#500个神经元
W2=tf.Variable(tf.truncated_normal([500,300],stddev=0.1))
b2=tf.Variable(tf.zeros([300])+0.1)
L2=tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop=tf.nn.dropout(L2,keep_prob)



#输出层10个神经元
W3=tf.Variable(tf.truncated_normal([300,10],stddev=0.1))
b3=tf.Variable(tf.zeros([10])+0.1)
#softmax将输出转化为概率值
prediction=tf.nn.softmax(tf.matmul(L2_drop,W3)+b3)


#二次代价函数,差的平方的平均值
#loss=tf.reduce_mean(tf.square(y-prediction))
#交叉熵代价函数的平均值
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#梯度下降法
#train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#Adam优化器 学习率：lr
train_step=tf.train.AdamOptimizer(lr).minimize(loss)


#初始化变量
init=tf.global_variables_initializer()

#tf.argmax(y,1)返回1的位置(真实值)，tf.argmax(prediction,1)(预测值)返回概率值最大的位置，比较位置是否相等，若想等返回true，不等返回false，存放在布尔列表中
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#tf.argmax()返回一维张量中最大值的位置
#求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    # 对所有图片迭代51次
    for epoch in range(51):
        #对所有图片分批训练一次,更改学习率(随着训练次数增加，减小学习率)
        sess.run(tf.assign(lr,0.001*(0.95**epoch)))
        for batch in range(n_batch):
            #获取一批(100个)样本图片,batch_xs:图片信息，batch_ys:图片标签
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            #利用训练图片信息及对应标签，梯度下降法训练模型，得到权重W及b，keep_prob=1.0代表训练时神经元不随机失活
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        #获取学习率到leaning_rate
        leaning_rate=sess.run(lr)
        #利用测试集进行测试该迭代时模型的准确率
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})#利用测试集测试模型
        #train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels,keep_prob:1.0})#利用训练集测试模型,准确率一定很高
        #打印迭代次数及对应准确率、学习率
        print("Iter "+str(epoch)+",Testing Accuracy "+str(test_acc)+",Leaning rate "+str(leaning_rate))




