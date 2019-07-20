import tensorflow as tf
#Fetch：一个会话中执行多个op
#定义三个常量
input1=tf.constant(3.0)
input2=tf.constant(2.0)
input3=tf.constant(5.0)

add=tf.add(input2,input3)
mul=tf.multiply(input1,add)

with tf.Session() as sess:
    #在一个会话中同时执行两个op,mul和add
    result=sess.run([mul,add])
    print(result)

#Feed
#创建占位符(相当于申请了控空间但没赋值),可以在运行时将值传入
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
#定义乘法op
output=tf.multiply(input1,input2)
with tf.Session() as sess:
    #feed中数据以字典的形式传入,为input1=7.0，input2=2.0
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))

