import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集,one_hot:将标签转换为只有一位为1，其它为0，会自动从网上下载数据集到当前目录
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次的大小
batch_size=100
#计算一共有多少个批次（整除）
n_batch=mnist.train.num_examples//batch_size

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)#平均值
        tf.summary.scalar('mean',mean)#为平均值取名mean
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)#标准差
        tf.summary.scalar('max',tf.reduce_max(var))#最大值
        tf.summary.scalar('min',tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram',var)#直方图

#命名空间
with tf.name_scope('input'):
    #该神经网络只输入层和输出层，输入层包含784个神经元，输出层包含10个神经元
    #定义两个placeholder,将28*28数字图片偏平为规格为784的向量
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    #标签结果
    y=tf.placeholder(tf.float32,[None,10],name='y-input')

with tf.name_scope('layer'):
    #创建一个简单的神经网络
    with tf.name_scope('wights'):
        #权值初始化为0,  784x10
        W=tf.Variable(tf.zeros([784,10]),name='W')
        # 分析权值的平均值、标准差、最大值、最小值、直方图
        variable_summaries(W)
    with tf.name_scope('biases'):
        #偏置值
        b=tf.Variable(tf.zeros([10]),name='b')
        # 分析偏置值的平均值、标准差、最大值、最小值、直方图
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b=tf.matmul(x,W)+b
    with tf.name_scope('softmax'):
        #softmax将输出转化为概率值
        prediction=tf.nn.softmax(wx_plus_b)

#二次代价函数,差的平方的平均值
#loss=tf.reduce_mean(tf.square(y-prediction))
with tf.name_scope('loss'):
    #交叉熵代价函数的平均值
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    #梯度下降法
    train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        #tf.argmax(y,1)返回1的位置(真实值)，tf.argmax(prediction,1)(预测值)返回概率值最大的位置，比较位置是否相等，若想等返回true，不等返回false，存放在布尔列表中
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#tf.argmax()返回一维张量中最大值的位置
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy', accuracy)#监测accuracy

#合并所有的检测summary
merged=tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    #将图的结构存放在当前目录下logs文件夹中
    writer=tf.summary.FileWriter('logs/',sess.graph)
    # 对所有图片迭代51次
    for epoch in range(51):
        #对所有图片分批训练一次
        for batch in range(n_batch):
            #获取一批(100个)样本图片,batch_xs:图片信息，batch_ys:图片标签
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            #利用训练图片信息及对应标签，训练模型，得到权重W及b
            summary,_  = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})

        #将summry及epoch写入文件夹logs下文件中
        writer.add_summary(summary,epoch)
        #利用测试集进行测试该迭代时模型的准确率
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        #打印迭代次数及对应准确率
        print("Iter "+str(epoch)+",Testing Accuracy "+str(acc))




