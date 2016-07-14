import numpy as np
import tensorflow as tf


def hist_summaries(*args):
    return tf.merge_summary([tf.histogram_summary(t.name, t) for t in args])


def fanin_init(shape, fanin=None):
    fanin = fanin or shape[0]
    v = 1 / np.sqrt(fanin)
    return tf.random_uniform(shape, minval=-v, maxval=v)


l1 = 200  # dm 200
l2 = 200  # dm 200


def theta(dimIn, dimOut, scope, v):
    with tf.variable_scope(scope):
        return [tf.Variable(fanin_init([dimIn, l1]), name='1w'),
                tf.Variable(fanin_init([l1], dimIn), name='1b'),
                tf.Variable(fanin_init([l1, l2]), name='2w'),
                tf.Variable(fanin_init([l2], l1), name='2b'),
                tf.Variable(tf.random_uniform([l2, dimOut], -v, v), name='3w'),
                tf.Variable(tf.random_uniform([dimOut], -v, v), name='3b')]

def lfunction(obs, theta_L, name="lfunction"):
    with tf.variable_op_scope([obs], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h1 = tf.nn.relu(tf.matmul(h0, theta_L[0]) + theta_L[1], name='h1')
        h2 = tf.nn.relu(tf.matmul(h1, theta_L[2]) + theta_L[3], name='h2')
        q = tf.matmul(h2, theta_L[4]) + theta_L[5]

        summary = hist_summaries(h0, h1, h2, q)
        return q, summary

def ufunction(obs, theta_U, name="ufunction"):
    with tf.variable_op_scope([obs], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h1 = tf.nn.relu(tf.matmul(h0, theta_U[0]) + theta_U[1], name='h1')
        h2 = tf.nn.relu(tf.matmul(h1, theta_U[2]) + theta_U[3], name='h2')
        act = tf.tanh(tf.matmul(h2, theta_U[4]) + theta_U[5])

        summary = hist_summaries(h0, h1, h2, act)
        return act, summary

def afunction(action, lvalue, uvalue, dimA, name="afunction"):
    with tf.variable_op_scope([action, lvalue, uvalue], name, name):
        h0a = tf.identity(action, name='h0-action')
        h0l = tf.identity(lvalue, name='h0-lvalue')
        h0u = tf.identity(uvalue, name='h0-uvalue')
        h0d = h0a - h0u

        L = tf.reshape(h0l, [-1, dimA, dimA])
        L = tf.batch_matrix_band_part(L, -1, 0) #lower triangle

        h1 = tf.reshape(h0d, [-1, 1, dimA])
        h1 = tf.batch_matmul(h1, L) # batch:1:dimA
        h1 = tf.squeeze(h1, [1]) #batch:dimA
        h2 = -0.5 * tf.reduce_sum(h1 * h1, 1) #batch

        summary = hist_summaries(h0d, L, h2)
        return h2, summary

def qfunction(obs, avalue, theta_V, name="qfunction"):
    with tf.variable_op_scope([obs, avalue], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h0a = tf.identity(avalue, name='h0-avalue')
        h1 = tf.nn.relu(tf.matmul(h0, theta_V[0]) + theta_V[1], name='h1')
        h2 = tf.nn.relu(tf.matmul(h1, theta_V[2]) + theta_V[3], name='h2')
        qs = tf.matmul(h2, theta_V[4]) + theta_V[5]
        q = tf.squeeze(qs, [1], name='h3-q') + h0a

        summary = hist_summaries(h0, h0a, h1, h2, q)
        return q, summary
