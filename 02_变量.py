import tensorflow as tf
#定义变量,必须初始化
x=tf.Variable([1,2])
#定义常量
a=tf.constant([3,3])
#增加一个减法op
sub=tf.subtract(x,a)
#增加一个加法op
add=tf.add(x,sub)
#初始化所有变量
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))
#定义变量counter，初始化为0
state=tf.Variable(0,name='counter')
#创建一个op,作用是state++
new_value=tf.add(state,1)
#赋值op,将new_value赋值给state
update=tf.assign(state,new_value)
#初始化所有变量
init=tf.global_variables_initializer()

with tf.Session() as sess:
    #变量初始化
    sess.run(init)
    #打印state值
    print(sess.run(state))
    #循环5次
    for _ in range(5):
        #执行state的更新操作，将new_value=state+1赋值给state
        sess.run(update)
        #打印state
        print(sess.run(state))


