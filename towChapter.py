# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:13:52 2018

@author: asus
"""

import tensorflow as tf

#定义两个常亮向量
a=tf.constant([1.0,2.0],name="a")
b=tf.constant([2.0,3.0],name="b")
result=a+b

#创建会话
sess=tf.Session()
print(sess.run(result))
print(result)

#输出计算图
print(a.graph)
print(tf.get_default_graph())


