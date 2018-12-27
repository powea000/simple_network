import numpy as np
import tensorflow as tf

def get_placeholder():
    
    x = tf.placeholder(tf.float32, [None, 2], name="x")
    y = tf.placeholder(tf.int64, [None, ], name="y")
    
    return x, y    

if __name__ == "__main__":

tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
    x, y = get_placeholder()

    with tf.variable_scope("net1"):
        w1 = tf.Variable(tf.random_normal([2, 2]), name='w1')
        b1 = tf.Variable(tf.random_normal([2]), name='b1')
        
        net1 =  tf.add(tf.matmul(x, w1), b1, name='net1')
        a1 = tf.nn.sigmoid(net1, name='a1')

    with tf.variable_scope("net2"):
        w2 = tf.Variable(tf.random_normal([2, 2]), name='w2')
        b2 = tf.Variable(tf.random_normal([2]), name='b2')
        
        net2 =  tf.add(tf.matmul(a1, w2), b2, name='net2')
        z = tf.nn.softmax(net2, name='output')

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y), name='loss')
    tf.summary.scalar("loss", loss)
    
    pre = tf.argmax(z, axis=1)
    correct_predictions = tf.equal(pre, y)
    acc = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
    
    global_step = tf.Variable(0, name="global_step", trainable=False)
    add_step_op = tf.assign_add(global_step, tf.constant(1))
    optimizer = tf.train.GradientDescentOptimizer(1)
    train_op = optimizer.minimize(loss)
    
    merge_op = tf.summary.merge_all()
    
with tf.Session(graph=train_graph) as sess:
    writer = tf.summary.FileWriter('/usr/local/qoss/staff/hhjj/teaching_material/graph', sess.graph)
    sess.run(tf.global_variables_initializer())
    _x = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
    _y = np.array([1, 1, 0, 0])
    
    for i in range(1000):
            
        feed = {x:_x, y:_y}
        sess.run(train_op, feed)
        step, train_loss, merge, _ = sess.run([add_step_op, loss, merge_op, train_op], feed)
        writer.add_summary(merge, step)
        
        print(sess.run(acc, feed))
        #print(train_loss)