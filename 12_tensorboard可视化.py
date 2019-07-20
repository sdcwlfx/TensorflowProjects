import tensorflow as tf
#tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

#载入数据集
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
#运行次数
max_steps=1001
#图片数量
image_num=3000
#文件路径
DIR="F:/ustcsse/TensorFlowPro"

#定义会话
sess=tf.Session()

#载入图片
embedding=tf.Variable(tf.stack(mnist.test.images[:image_num]),trainable=False,name='embedding')

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

#显示图片
with tf.name_scope('input_reshape'):
    #将x重塑形状，-1：代表不确定值(图片数量不确定) 28*28：28行28列 1：代表灰度图像(单通道)
    image_shaped_input=tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input',image_shaped_input,10)#放10张图片


with tf.name_scope('layer'):
    #创建一个简单的神经网络
    with tf.name_scope('wights'):
        #权值初始化为0,  784x10
        W=tf.Variable(tf.truncated_normal([784,10],stddev=0.1),name='W')
        # 分析权值的平均值、标准差、最大值、最小值、直方图
        variable_summaries(W)
    with tf.name_scope('biases'):
        #偏置值
        b=tf.Variable(tf.zeros([10])+0.1,name='b')
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
sess.run(init)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        #tf.argmax(y,1)返回1的位置(真实值)，tf.argmax(prediction,1)(预测值)返回概率值最大的位置，比较位置是否相等，若想等返回true，不等返回false，存放在布尔列表中
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#tf.argmax()返回一维张量中最大值的位置
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy', accuracy)#监测accuracy
#产生metadata文件
if tf.gfile.Exists(DIR+'/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR+'/projector/metadata.tsv')
with open(DIR+'/projector/metadata.tsv','w') as f:
    labels=sess.run(tf.argmax(mnist.test.labels[:],1))#求出标签最大值所在位置，保存到labels中
    for i in range(image_num):
        f.write(str(labels[i])+'\n')#每行一个标签写入metadata.tsv中

#合并所有的检测summary
merged=tf.summary.merge_all()

projector_writer=tf.summary.FileWriter(DIR+'/projector',sess.graph)
saver=tf.train.Saver()#保存网络模型
config=projector.ProjectorConfig()
embed=config.embeddings.add()
embed.tensor_name=embedding.name
embed.metadata_path=DIR+'/projector/metadata.tsv'
embed.sprite.image_path=DIR+'/projector/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28,28])
projector.visualize_embeddings(projector_writer,config)

#以供训练了max_steps*100章图片
for i in range(max_steps):
    #每个批次100个样本
    batch_xs,batch_ys= mnist.train.next_batch(100)
    #固定设置
    run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata=tf.RunMetadata()
    summary,_=sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys},options=run_options,run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata,'step%03d'%i)
    projector_writer.add_summary(summary,i)#记录参数变化

    if i%100==0:#每训练100次记录准确率
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter "+str(i)+", Testing Accuracy="+str(acc))
#保存训练后的模型
saver.save(sess,DIR+'/projector/a_model.ckpt',global_step=max_steps)
projector_writer.close()
sess.close()










