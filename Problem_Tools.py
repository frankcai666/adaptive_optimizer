# -*- coding: utf-8 -*-
# @Author   ：Frank
# @Time     ：2018/8/22 11:00

import tensorflow as tf
import numpy as np
import scipy.stats as sta
from My_Optimizer import MyAdam, MyMomentum, MyAdagrad, MyAdaptiveOptimizer, MyAdaptiveOptimizerMAA
from time import time


def com_network(output_node, dense_node, hidden_node, active_func, seed):
    com_graph = tf.Graph()
    with com_graph.as_default():
        tf.set_random_seed(seed)
        x = tf.sparse_placeholder(tf.float32, name='x_input')
        y = tf.placeholder(tf.float32, [None, output_node], name='y_input')
        w_initializer, b_initializer = tf.random_normal_initializer(0.0, 0.008), tf.constant_initializer(0.0)
        if active_func == 'elu':
            activate = tf.nn.elu

        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [dense_node, hidden_node], initializer=w_initializer)
            b1 = tf.get_variable('b1', [hidden_node], initializer=b_initializer)
            l1 = activate(tf.sparse_tensor_dense_matmul(x, w1) + b1, 'l1')

        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [hidden_node, hidden_node], initializer=w_initializer)
            b2 = tf.get_variable('b2', [hidden_node], initializer=b_initializer)
            l2 = activate(tf.matmul(l1, w2) + b2, 'l2')

        with tf.variable_scope('l3'):
            w3 = tf.get_variable('w3', [hidden_node, output_node], initializer=w_initializer)
            b3 = tf.get_variable('b3', [output_node], initializer=b_initializer)
            y_hat = tf.matmul(l2, w3) + b3

        prob = tf.nn.sigmoid(y_hat, name='prob')
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_hat, name='per_loss')
        loss = tf.reduce_mean(cross_entropy, name='loss')

        # correct_prediction = tf.equal(y, tf.cast(tf.greater_equal(prob, 0.5), tf.float32))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # auc, auc_op = tf.metrics.auc(labels=y, predictions=prob, name='auc')

    return com_graph, x, y


def one_layer_mnist(input_node, output_node, hidden_node, train_flag):
    olm_graph = tf.Graph()
    with olm_graph.as_default():
        if train_flag is True:
            tf.set_random_seed(1)
        x = tf.placeholder(tf.float32, [None, input_node], name='x')
        y = tf.placeholder(tf.float32, [None, output_node], name='y')
        w_initializer, b_initializer = tf.random_normal_initializer(0, 0.01), tf.random_normal_initializer(0, 0.01)

        with tf.variable_scope('l1'):
            h1 = tf.layers.dense(x, hidden_node, activation='relu', kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer)
        with tf.variable_scope('l2'):
            y_hat = tf.layers.dense(h1, output_node, kernel_initializer=w_initializer, bias_initializer=b_initializer)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=y_hat,
                                                                       name='per_loss')

        loss = tf.reduce_mean(cross_entropy, name='loss')

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    # [print(tensor.name) for tensor in olm_graph.as_graph_def().node]
    return olm_graph


def quadratic(batch_size, num_dims, train_flag):
    # Trainable variable.
    quadratic_graph = tf.Graph()
    with quadratic_graph.as_default():
        if train_flag is True:
            tf.set_random_seed(1)
        with tf.variable_scope('quadratic'):
            x = tf.get_variable(
                "x",
                shape=[batch_size, num_dims],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.01))

            # Non-trainable variables.
            w = tf.get_variable("w",
                                shape=[batch_size, num_dims, num_dims],
                                dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(),
                                trainable=False)
            y = tf.get_variable("y",
                                shape=[batch_size, num_dims],
                                dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(),
                                trainable=False)
        product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
        per_loss = tf.reduce_sum((product - y) ** 2, 1, name='per_loss')
        loss = tf.reduce_mean(per_loss, name='loss')
        # loss = tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1))
    return quadratic_graph


def next_batch(batch_size, count, x_set, y_set):
    data_size = x_set.shape[1]
    start = count * batch_size
    end = min(start + batch_size, data_size)
    next_x = x_set[start:end, :]
    next_y = y_set[start:end]
    if end == data_size:
        flag_batch_end = 1
    else:
        flag_batch_end = 0
    return next_x, next_y, flag_batch_end


def optimizer_construct(learning_rate, optimizer_name):
    dictionary = {'adam': tf.train.AdamOptimizer(learning_rate),
                  'myadam': MyAdam(learning_rate),
                  'myadagrad': MyAdagrad(learning_rate),
                  'mymomentum': MyMomentum(learning_rate, 0.9),
                  'rmsprop': tf.train.RMSPropOptimizer(learning_rate),
                  'adagrad': tf.train.AdagradOptimizer(learning_rate),
                  'adadelta': tf.train.AdadeltaOptimizer(learning_rate),
                  'nesterov': tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True),
                  'momentum': tf.train.MomentumOptimizer(learning_rate, momentum=0.9),
                  'gd': tf.train.GradientDescentOptimizer(learning_rate)}

    return dictionary[optimizer_name]


def smooth_kurtosis(data):
    kurtosis_per_loss = sta.kurtosis(data, axis=None) + 3
    if kurtosis_per_loss > 0:
        return np.log(kurtosis_per_loss + 1)
    else:
        return kurtosis_per_loss


def my_log(x):
    return np.log(x+1)


def cal_feature(epoch, per_loss, grads, pre_grads):
    feature_list = []
    '''
        statistic value of per_loss state[1-6]
    '''
    mean_per_loss = np.mean(per_loss)
    feature_list.append(my_log(mean_per_loss))
    # feature_list.append(np.median(per_loss))
    feature_list.append(my_log(np.std(per_loss)))
    # feature_list.append(sta.entropy(per_loss))
    # feature_list.append(smooth_kurtosis(per_loss))
    # feature_list.append(sta.skew(per_loss))
    less_mean_loss = np.mean(per_loss < mean_per_loss)
    feature_list.append(less_mean_loss)

    '''
        statistic value of grads state[7-13]
    '''
    abs_grads = np.abs(grads)
    mean_abs_grads = np.mean(abs_grads)
    feature_list.append(my_log(mean_abs_grads))
    # median_abs_grads = np.median(abs_grads)
    # feature_list.append(median_abs_grads)
    feature_list.append(my_log(np.std(abs_grads)))
    # feature_list.append(sta.entropy(abs_grads.flatten()))

    # feature_list.append(smooth_kurtosis(abs_grads))
    # feature_list.append(sta.skew(abs_grads, axis=None))

    less_mean_grad = np.mean(abs_grads < mean_abs_grads)
    feature_list.append(less_mean_grad)
    # positive_grad = grads > 0
    # feature_list.append(np.mean(positive_grad))

    '''
        statistic value of delta grads[14-18]
    '''
    if epoch == 0:
        feature_list.append(0)
        feature_list.append(0)
        # feature_list.append(0)
        feature_list.append(0.5)
        feature_list.append(1)
    else:
        abs_pre_grads = np.abs(pre_grads)
        delta_grad = np.abs((pre_grads - grads) / (abs_pre_grads + 1e-8))
        mean_delta_grad = np.mean(delta_grad)
        feature_list.append(my_log(mean_delta_grad))
        # feature_list.append(np.median(delta_grad))
        feature_list.append(my_log(np.std(delta_grad)))
        less_mean_delta = np.mean(delta_grad < mean_delta_grad)
        feature_list.append(less_mean_delta)

        positive_pre_grad = (pre_grads > 0)
        positive_grad = (grads > 0)
        feature_list.append(np.mean(positive_grad == positive_pre_grad))

    return np.array(feature_list)


def cal_feature_1(epoch, loss_graph, opt_sess, grads, pre_grads):
    feature_list = []
    graph = loss_graph.graph
    '''
        statistic value of per_loss state[1-6]
    '''
    per_loss = opt_sess.run(graph.get_tensor_by_name('per_loss:0'))
    mean_per_loss = np.mean(per_loss)
    feature_list.append(mean_per_loss)
    feature_list.append(np.median(per_loss))
    feature_list.append(np.std(per_loss))
    feature_list.append(sta.entropy(per_loss))
    feature_list.append(smooth_kurtosis(per_loss))
    # feature_list.append(sta.skew(per_loss))
    less_mean_loss = np.mean(per_loss < mean_per_loss)
    feature_list.append(less_mean_loss)

    '''
        statistic value of grads state[7-13]
    '''
    abs_grads = np.abs(grads)
    mean_abs_grads = np.mean(abs_grads)
    feature_list.append(mean_abs_grads)
    median_abs_grads = np.median(abs_grads)
    feature_list.append(median_abs_grads)
    std_abs_grads = np.std(abs_grads)
    feature_list.append(std_abs_grads)
    feature_list.append(sta.entropy(abs_grads.flatten()))

    feature_list.append(smooth_kurtosis(abs_grads))
    # feature_list.append(sta.skew(abs_grads, axis=None))

    less_mean_grad = np.mean(abs_grads < mean_abs_grads)
    feature_list.append(less_mean_grad)
    positive_grad = grads > 0
    feature_list.append(np.mean(positive_grad))

    '''
        statistic value of delta grads[14-18]
    '''
    if epoch == 0:
        feature_list.append(0)
        feature_list.append(0)
        feature_list.append(0)
        feature_list.append(less_mean_grad)
        feature_list.append(1)
    else:
        abs_pre_grads = np.abs(pre_grads)
        delta_grad = np.log(np.abs(pre_grads - grads) / (abs_pre_grads + 1e-6) + 1)
        mean_delta_grad = np.mean(delta_grad)
        feature_list.append(mean_delta_grad)
        feature_list.append(np.median(delta_grad))
        feature_list.append(np.std(delta_grad))
        less_mean_delta = np.mean(delta_grad < mean_delta_grad)
        feature_list.append(less_mean_delta)

        positive_pre_grad = (pre_grads > 0)
        positive_grad = (grads > 0)
        feature_list.append(np.mean(positive_grad == positive_pre_grad))

    return np.array(feature_list)


def cal_reward(pre_loss, current_loss):
    if pre_loss - current_loss < 0:
        return pre_loss, 0
    mag_pre_loss = np.floor(np.log10(pre_loss))
    mag_opt_loss = np.floor(np.log10(current_loss))

    if mag_opt_loss < mag_pre_loss:
        reward_value = mag_pre_loss - mag_opt_loss
        pre_loss = current_loss
    else:
        reward_value = 0

    return pre_loss, reward_value


'''
    
    RL training nn
'''


def grads_handle(grads):
    concat_grad = grads[0].reshape(-1)
    for i in range(1, len(grads)):
        concat_grad = np.hstack((concat_grad, grads[i].reshape(-1)))
    return concat_grad


def rl_training_nn(loss_graph, mnist_train, validate_x, validate_y, agent, config, epoch_max, epoch_gap,
                   initial_flag):
    x_input = loss_graph.get_tensor_by_name('x:0')
    y_input = loss_graph.get_tensor_by_name('y:0')
    loss = loss_graph.get_tensor_by_name('loss:0')
    accuracy = loss_graph.get_tensor_by_name('accuracy:0')
    per_loss = loss_graph.get_tensor_by_name('per_loss/per_loss:0')
    loss_his_validate = np.zeros(epoch_max)
    accuracy_his_validate = np.zeros(epoch_max)
    with tf.Session(graph=loss_graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # initial step
        opt_tmp = tf.train.GradientDescentOptimizer(learning_rate=1)
        grads_and_vars = opt_tmp.compute_gradients(loss)
        grads, _ = zip(
            *sess.run(grads_and_vars, feed_dict={x_input: validate_x, y_input: validate_y})
        )
        grad = grads_handle(grads)
        pre_grad = grad
        per_loss_tmp, accuracy_his_validate[0] = sess.run([per_loss, accuracy], feed_dict={
            x_input: validate_x, y_input: validate_y
        })
        loss_his_validate[0] = np.mean(per_loss_tmp)
        pre_loss = loss_his_validate[0]

        state_features = cal_feature(epoch=0, per_loss=per_loss_tmp, grads=grad, pre_grads=pre_grad)
        action_tmp, lr_tmp, w1_tmp, w2_tmp, w3_tmp = agent.choose_action(state_features, train_flag=True)
        pre_vars = set(tf.global_variables())
        beta1 = 0.9
        beta2 = 0.999
        opt_tmp = MyAdaptiveOptimizerMAA(lr_tmp, w1_tmp, w2_tmp, w3_tmp, None, None, None, None)
        train_op = opt_tmp.minimize(loss)
        sess.run(tf.variables_initializer(set(tf.global_variables()) - pre_vars))
        pre_action = action_tmp
        # training step
        for epoch in range(1, epoch_max):
            x_train, y_train = mnist_train.next_batch(batch_size=128, shuffle=True)
            sess.run(train_op, feed_dict={x_input: x_train, y_input: y_train})
            per_loss_tmp, accuracy_his_validate[epoch] = sess.run([per_loss, accuracy], feed_dict={
                x_input: validate_x, y_input: validate_y
            })
            loss_his_validate[epoch] = np.mean(per_loss_tmp)

            if (pre_loss - loss_his_validate[epoch]) / pre_loss < -0.1:
                agent.store_hist(state_features, action_tmp, 1 / loss_his_validate[epoch], state_features, 0,
                                 initial_flag)
                return agent, loss_his_validate, accuracy_his_validate

            # checking state and change strategy
            if epoch % epoch_gap == 0:
                reward = 1 / loss_his_validate[epoch]
                grads_and_vars = opt_tmp.compute_gradients(loss)
                grads, _ = zip(
                    *sess.run(grads_and_vars,
                              feed_dict={x_input: validate_x, y_input: validate_y})
                )
                grad = grads_handle(grads)
                state_features_ = cal_feature(epoch=epoch, per_loss=per_loss_tmp, grads=grad, pre_grads=pre_grad)
                pre_grad = grad
                # store sample
                agent.store_hist(state_features, action_tmp, reward, state_features_, 1, initial_flag)
                # record last loss
                pre_loss = loss_his_validate[epoch]
                # decision making0
                action_tmp, lr_tmp, w1_tmp, w2_tmp, w3_tmp = agent.choose_action(state_features_, train_flag=True)
                state_features = state_features_
                if action_tmp != pre_action:
                    # constructing optimizer
                    m_tmp = []
                    n_tmp = []
                    m_adam_tmp = []
                    n_adam_tmp = []
                    for v in tf.trainable_variables():
                        m_tmp.append(sess.run(opt_tmp.get_slot(v, 'm')))
                        n_tmp.append(sess.run(opt_tmp.get_slot(v, 'n')))
                        m_adam_tmp.append(sess.run(opt_tmp.get_slot(v, 'm_adam')))
                        n_adam_tmp.append(sess.run(opt_tmp.get_slot(v, 'n_adam')))
                    pre_vars = set(tf.global_variables())
                    opt_tmp = MyAdaptiveOptimizerMAA(lr_tmp, w1_tmp, w2_tmp, w3_tmp, m_tmp, n_tmp, m_adam_tmp, n_adam_tmp,
                                                     beta1_power=beta1 ** (epoch + 1), beta2_power=beta2 ** (epoch + 1))
                    train_op = opt_tmp.minimize(loss)
                    sess.run(tf.variables_initializer(set(tf.global_variables()) - pre_vars))
                pre_action = action_tmp
        # final step
        # if (pre_loss - loss_his_validate[epoch]) / pre_loss < -0.05:
        #     agent.store_hist(state_features, action_tmp, -1, state_features, 0, initial_flag)
        #     return agent, loss_his_validate, accuracy_his_validate
        # else:
        reward = 1 / loss_his_validate[epoch]
        grads_and_vars = opt_tmp.compute_gradients(loss)
        grads, _ = zip(
            *sess.run(grads_and_vars, feed_dict={x_input: validate_x, y_input: validate_y})
        )
        grad = grads_handle(grads)
        state_features_ = cal_feature(epoch=epoch, per_loss=per_loss_tmp, grads=grad, pre_grads=pre_grad)
        # store sample
        agent.store_hist(state_features, action_tmp, reward, state_features_, 0, initial_flag)
    return agent, loss_his_validate, accuracy_his_validate


def rl_testing_nn(loss_graph, mnist_train, validate_x, validate_y, agent, config, epoch_max, epoch_gap):
    x_input = loss_graph.get_tensor_by_name('x:0')
    y_input = loss_graph.get_tensor_by_name('y:0')
    loss = loss_graph.get_tensor_by_name('loss:0')
    accuracy = loss_graph.get_tensor_by_name('accuracy:0')
    per_loss = loss_graph.get_tensor_by_name('per_loss/per_loss:0')
    loss_his_validate = np.zeros(epoch_max)
    accuracy_his_validate = np.zeros(epoch_max)
    with tf.Session(graph=loss_graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        start = time()
        # initial step
        opt_tmp = tf.train.GradientDescentOptimizer(learning_rate=1)
        grads_and_vars = opt_tmp.compute_gradients(loss)
        grads, _ = zip(
            *sess.run(grads_and_vars, feed_dict={x_input: validate_x, y_input: validate_y})
        )
        grad = grads_handle(grads)
        pre_grad = grad
        per_loss_tmp, accuracy_his_validate[0] = sess.run([per_loss, accuracy], feed_dict={
            x_input: validate_x, y_input: validate_y
        })
        loss_his_validate[0] = np.mean(per_loss_tmp)
        pre_loss = loss_his_validate[0]

        action_record = []

        state_features = cal_feature(epoch=0, per_loss=per_loss_tmp, grads=grad, pre_grads=pre_grad)
        action_tmp, lr_tmp, w1_tmp, w2_tmp, w3_tmp = agent.choose_action(state_features, train_flag=False)
        action_record.append(action_tmp)
        pre_vars = set(tf.global_variables())
        beta1 = 0.9
        beta2 = 0.999
        opt_tmp = MyAdaptiveOptimizerMAA(lr_tmp, w1_tmp, w2_tmp, w3_tmp, None, None, None, None)
        train_op = opt_tmp.minimize(loss)
        sess.run(tf.variables_initializer(set(tf.global_variables()) - pre_vars))
        pre_action = action_tmp
        # training step
        for epoch in range(1, epoch_max):
            x_train, y_train = mnist_train.next_batch(batch_size=128, shuffle=True)
            sess.run(train_op, feed_dict={x_input: x_train, y_input: y_train})
            per_loss_tmp, accuracy_his_validate[epoch] = sess.run([per_loss, accuracy], feed_dict={
                x_input: validate_x, y_input: validate_y
            })
            loss_his_validate[epoch] = np.mean(per_loss_tmp)

            if (pre_loss - loss_his_validate[epoch]) / pre_loss < -0.1:
                print('The %d th over loss: %g accuracy: %g, Failed !!!!!!!' % (
                    epoch, loss_his_validate[epoch], accuracy_his_validate[epoch]))
                print(agent.action_value_record[-1][-10::])
                return loss_his_validate, accuracy_his_validate, loss_his_validate[epoch], accuracy_his_validate[epoch]

            # checking state and change strategy
            if epoch % epoch_gap == 0:
                grads_and_vars = opt_tmp.compute_gradients(loss)
                grads, _ = zip(
                    *sess.run(grads_and_vars,
                              feed_dict={x_input: validate_x, y_input: validate_y})
                )
                grad = grads_handle(grads)
                state_features_ = cal_feature(epoch=epoch, per_loss=per_loss_tmp, grads=grad, pre_grads=pre_grad)
                pre_grad = grad
                # store sample
                # record last loss
                pre_loss = loss_his_validate[epoch]
                # decision making0
                action_tmp, lr_tmp, w1_tmp, w2_tmp, w3_tmp = agent.choose_action(state_features_, train_flag=False)
                action_record.append(action_tmp)
                state_features = state_features_
                if action_tmp != pre_action:
                    # constructing optimizer
                    m_tmp = []
                    n_tmp = []
                    m_adam_tmp = []
                    n_adam_tmp = []
                    for v in tf.trainable_variables():
                        m_tmp.append(sess.run(opt_tmp.get_slot(v, 'm')))
                        n_tmp.append(sess.run(opt_tmp.get_slot(v, 'n')))
                        m_adam_tmp.append(sess.run(opt_tmp.get_slot(v, 'm_adam')))
                        n_adam_tmp.append(sess.run(opt_tmp.get_slot(v, 'n_adam')))
                    pre_vars = set(tf.global_variables())
                    opt_tmp = MyAdaptiveOptimizerMAA(lr_tmp, w1_tmp, w2_tmp, w3_tmp, m_tmp, n_tmp, m_adam_tmp, n_adam_tmp,
                                                     beta1_power=beta1 ** (epoch + 1), beta2_power=beta2 ** (epoch + 1))
                    train_op = opt_tmp.minimize(loss)
                    sess.run(tf.variables_initializer(set(tf.global_variables()) - pre_vars))
                pre_action = action_tmp
        print('After %d training epoch, loss: %g, accuracy: %g, time_cost: %g' %
              (epoch, loss_his_validate[epoch], accuracy_his_validate[epoch], time() - start))
        print(action_record)
    return loss_his_validate, accuracy_his_validate, loss_his_validate[epoch], accuracy_his_validate[epoch]


'''
    RL training quadratic with 5 epoch
'''


def rl_training_quadratic(loss_graph, agent, config, epoch_max, epoch_gap, initial_flag):
    loss = loss_graph.get_tensor_by_name('loss:0')
    per_loss = loss_graph.get_tensor_by_name('per_loss:0')
    loss_his = np.zeros(epoch_max)
    with tf.Session(graph=loss_graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # initial step
        opt_tmp = tf.train.GradientDescentOptimizer(learning_rate=1)
        grads_and_vars = opt_tmp.compute_gradients(loss)
        grads, _ = zip(*sess.run(grads_and_vars))
        grad = grads_handle(grads)
        pre_grad = grad
        per_loss_tmp = sess.run(per_loss)
        loss_his[0] = np.mean(per_loss_tmp)
        pre_loss = loss_his[0]

        state_features = cal_feature(epoch=0, per_loss=per_loss_tmp, grads=grad, pre_grads=pre_grad)
        action_tmp, lr_tmp, w1_tmp, w2_tmp, w3_tmp = agent.choose_action(state_features, train_flag=True)
        pre_vars = set(tf.global_variables())
        beta1 = 0.9
        beta2 = 0.999
        opt_tmp = MyAdaptiveOptimizerMAA(lr_tmp, w1_tmp, w2_tmp, w3_tmp, None, None, None, None)
        train_op = opt_tmp.minimize(loss)
        sess.run(tf.variables_initializer(set(tf.global_variables()) - pre_vars))
        pre_action = action_tmp
        # training step
        for epoch in range(1, epoch_max):
            sess.run(train_op)
            per_loss_tmp = sess.run(per_loss)
            loss_his[epoch] = np.mean(per_loss_tmp)

            if (pre_loss - loss_his[epoch]) / pre_loss < -0.05:
                agent.store_hist(state_features, action_tmp, 1 / loss_his[epoch], state_features, 0,
                                 initial_flag)
                return agent, loss_his

            # checking state and change strategy
            if epoch % epoch_gap == 0:
                reward = 1 / loss_his[epoch]
                grads_and_vars = opt_tmp.compute_gradients(loss)
                grads, _ = zip(*sess.run(grads_and_vars))
                grad = grads_handle(grads)
                state_features_ = cal_feature(epoch=epoch, per_loss=per_loss_tmp, grads=grad, pre_grads=pre_grad)
                pre_grad = grad
                # store sample
                agent.store_hist(state_features, action_tmp, reward, state_features_, 1, initial_flag)
                # record last loss
                pre_loss = loss_his[epoch]
                # decision making0
                action_tmp, lr_tmp, w1_tmp, w2_tmp, w3_tmp = agent.choose_action(state_features_, train_flag=True)
                state_features = state_features_
                if action_tmp != pre_action:
                    # constructing optimizer
                    m_tmp = []
                    n_tmp = []
                    m_adam_tmp = []
                    n_adam_tmp = []
                    for v in tf.trainable_variables():
                        m_tmp.append(sess.run(opt_tmp.get_slot(v, 'm')))
                        n_tmp.append(sess.run(opt_tmp.get_slot(v, 'n')))
                        m_adam_tmp.append(sess.run(opt_tmp.get_slot(v, 'm_adam')))
                        n_adam_tmp.append(sess.run(opt_tmp.get_slot(v, 'n_adam')))
                    pre_vars = set(tf.global_variables())
                    opt_tmp = MyAdaptiveOptimizerMAA(lr_tmp, w1_tmp, w2_tmp, w3_tmp, m_tmp, n_tmp, m_adam_tmp, n_adam_tmp,
                                                     beta1_power=beta1 ** (epoch + 1), beta2_power=beta2 ** (epoch + 1))
                    train_op = opt_tmp.minimize(loss)
                    sess.run(tf.variables_initializer(set(tf.global_variables()) - pre_vars))
                pre_action = action_tmp
        # final step
        # if (pre_loss - loss_his_validate[epoch]) / pre_loss < -0.05:
        #     agent.store_hist(state_features, action_tmp, -1, state_features, 0, initial_flag)
        #     return agent, loss_his_validate, accuracy_his_validate
        # else:
        reward = 1 / loss_his[epoch]
        grads_and_vars = opt_tmp.compute_gradients(loss)
        grads, _ = zip(*sess.run(grads_and_vars))
        grad = grads_handle(grads)
        state_features_ = cal_feature(epoch=epoch, per_loss=per_loss_tmp, grads=grad, pre_grads=pre_grad)
        # store sample
        agent.store_hist(state_features, action_tmp, reward, state_features_, 0, initial_flag)
    return agent, loss_his


def rl_testing_quadratic(loss_graph, agent, config, epoch_max, epoch_gap):
    loss = loss_graph.get_tensor_by_name('loss:0')
    per_loss = loss_graph.get_tensor_by_name('per_loss:0')
    loss_his = np.zeros(epoch_max)
    with tf.Session(graph=loss_graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        start = time()
        # initial step
        opt_tmp = tf.train.GradientDescentOptimizer(learning_rate=1)
        grads_and_vars = opt_tmp.compute_gradients(loss)
        grads, _ = zip(*sess.run(grads_and_vars))
        grad = grads_handle(grads)
        pre_grad = grad
        per_loss_tmp = sess.run(per_loss)
        loss_his[0] = np.mean(per_loss_tmp)
        pre_loss = loss_his[0]

        action_record = []
        state_features = cal_feature(epoch=0, per_loss=per_loss_tmp, grads=grad, pre_grads=pre_grad)
        action_tmp, lr_tmp, w1_tmp, w2_tmp, w3_tmp = agent.choose_action(state_features, train_flag=False)
        action_record.append(action_tmp)
        pre_vars = set(tf.global_variables())
        beta1 = 0.9
        beta2 = 0.999
        opt_tmp = MyAdaptiveOptimizerMAA(lr_tmp, w1_tmp, w2_tmp, w3_tmp, None, None, None, None)
        train_op = opt_tmp.minimize(loss)
        sess.run(tf.variables_initializer(set(tf.global_variables()) - pre_vars))
        pre_action = action_tmp
        # training step
        for epoch in range(1, epoch_max):
            sess.run(train_op)
            per_loss_tmp = sess.run(per_loss)
            loss_his[epoch] = np.mean(per_loss_tmp)

            if (pre_loss - loss_his[epoch]) / pre_loss < -0.05:
                print('The %d th over loss: %g, Failed !!!!!!!' % (
                    epoch, loss_his[epoch]))
                print(agent.action_value_record[-1][-10::])
                return loss_his, loss_his[epoch]

            # checking state and change strategy
            if epoch % epoch_gap == 0:
                reward = 1 / loss_his[epoch]
                grads_and_vars = opt_tmp.compute_gradients(loss)
                grads, _ = zip(*sess.run(grads_and_vars))
                grad = grads_handle(grads)
                state_features_ = cal_feature(epoch=epoch, per_loss=per_loss_tmp, grads=grad, pre_grads=pre_grad)
                pre_grad = grad
                # record last loss
                pre_loss = loss_his[epoch]
                # decision making0
                action_tmp, lr_tmp, w1_tmp, w2_tmp, w3_tmp = agent.choose_action(state_features_, train_flag=False)
                action_record.append(action_tmp)
                state_features = state_features_
                if action_tmp != pre_action:
                    # constructing optimizer
                    m_tmp = []
                    n_tmp = []
                    m_adam_tmp = []
                    n_adam_tmp = []
                    for v in tf.trainable_variables():
                        m_tmp.append(sess.run(opt_tmp.get_slot(v, 'm')))
                        n_tmp.append(sess.run(opt_tmp.get_slot(v, 'n')))
                        m_adam_tmp.append(sess.run(opt_tmp.get_slot(v, 'm_adam')))
                        n_adam_tmp.append(sess.run(opt_tmp.get_slot(v, 'n_adam')))
                    pre_vars = set(tf.global_variables())
                    opt_tmp = MyAdaptiveOptimizerMAA(lr_tmp, w1_tmp, w2_tmp, w3_tmp, m_tmp, n_tmp, m_adam_tmp, n_adam_tmp,
                                                     beta1_power=beta1 ** (epoch + 1), beta2_power=beta2 ** (epoch + 1))
                    train_op = opt_tmp.minimize(loss)
                    sess.run(tf.variables_initializer(set(tf.global_variables()) - pre_vars))
                pre_action = action_tmp
        # final step
        print('After %d training epoch, loss: %g time_cost: %g' %
              (epoch, loss_his[epoch], time() - start))
        print(action_record)
    return loss_his, loss_his[epoch]


'''
    RL training with 10%
'''


def rl_training(loss_graph, agent, train_config, epoch_max, initial_flag):
    with tf.Session(graph=loss_graph.graph, config=train_config) as sess:
        # print(sess.run(tf.report_uninitialized_variables()))
        sess.run(tf.global_variables_initializer())
        loss_his = np.zeros((epoch_max, 1))

        # initial step
        opt_tmp = tf.train.GradientDescentOptimizer(learning_rate=1)
        grads_and_vars = opt_tmp.compute_gradients(loss_graph)
        grads, _ = zip(*sess.run(grads_and_vars))
        pre_grads = grads
        state_features = cal_feature_1(epoch=0, loss_graph=loss_graph, opt_sess=sess, grads=grads[0],
                                       pre_grads=pre_grads[0])
        action_tmp, lr_tmp, w1_tmp, w2_tmp = agent.choose_action(state_features, train_flag=True)
        loss_his[0, :] = sess.run(loss_graph)
        pre_loss = loss_his[0, :]
        pre_epoch = 0
        # print('The initial loss value is %g' % loss_his[0, :])

        # print('^^^^^^ initialize m and n ^^^^^^')
        # a = sess.run(tf.trainable_variables())[0].shape
        # print(a)
        variable_shape = sess.run(tf.trainable_variables())[0].shape
        m_tmp = np.zeros(variable_shape, dtype=np.float32)
        n_tmp = np.zeros(variable_shape, dtype=np.float32)

        pre_vars = set(tf.global_variables())
        opt_tmp = MyAdaptiveOptimizer(lr_tmp, w1_tmp, w2_tmp, m_tmp, n_tmp)
        train_op = opt_tmp.minimize(loss_graph)
        sess.run(tf.variables_initializer(set(tf.global_variables()) - pre_vars))
        # adaptive adjust record
        control_record = 1
        action_record = [action_tmp]
        action_epoch_record = [0]
        reward_pre = 0
        # optimize the loss graph
        for epoch in range(1, epoch_max):
            # record the loss value
            sess.run(train_op)
            loss_his[epoch, :] = sess.run(loss_graph)
            reduced_ratio = (pre_loss - loss_his[epoch, :]) / pre_loss
            # Changing action process
            if reduced_ratio > 0.1 or reduced_ratio < -0.1:
                if reduced_ratio < -0.1:
                    reward = -1
                    # print('The %d th over %g, Failed !!!!!!!' % (epoch, reduced_ratio))
                    break
                else:
                    reward = reduced_ratio
                    # reward = (loss_his[pre_epoch, :] - loss_his[epoch, :]) / loss_his[0, :] / (epoch - pre_epoch)
                    # reward = (loss_his[0, :] - loss_his[epoch, :]) / loss_his[0, :]
                    # reward = (pre_loss - loss_his[epoch, :]) / loss_his[0, :]

                control_record += 1
                # obtain gradient value
                grads, _ = zip(*sess.run(opt_tmp.compute_gradients(loss_graph)))
                # state calculation
                state_features_ = cal_feature_1(epoch=epoch, loss_graph=loss_graph, opt_sess=sess, grads=grads[0],
                                                pre_grads=pre_grads[0])
                pre_grads = grads
                # store sample
                agent.store_hist(state_features, action_tmp, reward, state_features_, 1, reward_pre, initial_flag)
                # record last loss
                pre_loss = loss_his[epoch, :]
                pre_epoch = epoch
                # decision making
                action_tmp, lr_tmp, w1_tmp, w2_tmp = agent.choose_action(state_features_, train_flag=True)
                action_record.append(action_tmp)
                action_epoch_record.append(epoch)
                state_features = state_features_
                # constructing optimizer
                for v in tf.trainable_variables():
                    m_tmp = sess.run(opt_tmp.get_slot(v, 'm'))
                    n_tmp = sess.run(opt_tmp.get_slot(v, 'n'))

                pre_vars = set(tf.global_variables())
                opt_tmp = MyAdaptiveOptimizer(lr_tmp, w1_tmp, w2_tmp, m_tmp, n_tmp)
                train_op = opt_tmp.minimize(loss_graph)
                sess.run(tf.variables_initializer(set(tf.global_variables()) - pre_vars))
        # final step reward
        if reward == -1:
            final_reward = reward
            agent.store_hist(state_features, action_tmp, final_reward, state_features, 0, reward_pre, initial_flag)
            # if control_record > 1:
            #     reward_pre = (loss_his[0] - loss_his[action_epoch_record[-1]]) / loss_his[0]
        else:
            final_reward = (loss_his[0, :] - loss_his[epoch, :]) / loss_his[0, :]
            grads, _ = zip(*sess.run(opt_tmp.compute_gradients(loss_graph)))
            # state calculation
            state_features_ = cal_feature_1(epoch=epoch, loss_graph=loss_graph, opt_sess=sess, grads=grads[0],
                                            pre_grads=pre_grads[0])
            agent.store_hist(state_features, action_tmp, final_reward, state_features_, 0, reward_pre, initial_flag)
        # print('After %d training epoch, loss_value: %g, call controller %d times, time_cost: %g' %
        #       (epoch, loss_his[epoch, 0], control_record, end - start))
        # print(action_epoch_record)
        # print(action_record)
    return agent, loss_his


def rl_testing(loss_graph, agent, train_config, epoch_max):
    with tf.Session(graph=loss_graph.graph, config=train_config) as sess:
        # print(sess.run(tf.report_uninitialized_variables()))
        sess.run(tf.global_variables_initializer())
        loss_his = np.zeros((epoch_max, 1))

        # initial step
        opt_tmp = tf.train.GradientDescentOptimizer(learning_rate=1)
        grads_and_vars = opt_tmp.compute_gradients(loss_graph)
        grads, _ = zip(*sess.run(grads_and_vars))
        pre_grads = grads
        state_features = cal_feature_1(epoch=0, loss_graph=loss_graph, opt_sess=sess, grads=grads[0],
                                       pre_grads=pre_grads[0])
        action_tmp, lr_tmp, w1_tmp, w2_tmp = agent.choose_action(state_features, train_flag=False)
        loss_his[0, :] = sess.run(loss_graph)
        pre_loss = loss_his[0, :]
        print('The initial loss value is %g' % loss_his[0, :])

        # print('^^^^^^ initialize m and n ^^^^^^')
        # a = sess.run(tf.trainable_variables())[0].shape
        # print(a)
        variable_shape = sess.run(tf.trainable_variables())[0].shape
        m_tmp = np.zeros(variable_shape, dtype=np.float32)
        n_tmp = np.zeros(variable_shape, dtype=np.float32)

        pre_vars = set(tf.global_variables())
        opt_tmp = MyAdaptiveOptimizer(lr_tmp, w1_tmp, w2_tmp, m_tmp, n_tmp)
        train_op = opt_tmp.minimize(loss_graph)
        sess.run(tf.variables_initializer(set(tf.global_variables()) - pre_vars))
        # adaptive adjust record
        control_record = 1
        action_record = [action_tmp]
        action_epoch_record = [0]
        start = time()
        # optimize the loss graph
        for epoch in range(1, epoch_max):
            # record the loss value
            sess.run(train_op)
            loss_his[epoch, :] = sess.run(loss_graph)
            reduced_ratio = (pre_loss - loss_his[epoch, :]) / pre_loss
            # Changing action process
            if reduced_ratio > 0.1 or reduced_ratio < -0.1:
                if reduced_ratio < -0.1:
                    print('The %d th over %g, Failed !!!!!!!' % (epoch, reduced_ratio))
                    print(agent.action_value_record[-1][-10::])
                    break
                control_record += 1
                # obtain gradient value
                grads, _ = zip(*sess.run(opt_tmp.compute_gradients(loss_graph)))
                # state calculation
                state_features_ = cal_feature_1(epoch=epoch, loss_graph=loss_graph, opt_sess=sess, grads=grads[0],
                                                pre_grads=pre_grads[0])
                pre_grads = grads
                # record last loss
                pre_loss = loss_his[epoch, :]
                # decision making
                action_tmp, lr_tmp, w1_tmp, w2_tmp = agent.choose_action(state_features_, train_flag=False)
                action_record.append(action_tmp)
                action_epoch_record.append(epoch)
                state_features = state_features_
                # constructing optimizer
                for v in tf.trainable_variables():
                    m_tmp = sess.run(opt_tmp.get_slot(v, 'm'))
                    n_tmp = sess.run(opt_tmp.get_slot(v, 'n'))

                pre_vars = set(tf.global_variables())
                opt_tmp = MyAdaptiveOptimizer(lr_tmp, w1_tmp, w2_tmp, m_tmp, n_tmp)
                train_op = opt_tmp.minimize(loss_graph)
                sess.run(tf.variables_initializer(set(tf.global_variables()) - pre_vars))
        end = time()
        print('After %d training epoch, loss_value: %g, call controller %d times, time_cost: %g' %
              (epoch, loss_his[epoch, 0], control_record, end - start))
        print(action_epoch_record)
        print(action_record)
    return loss_his, loss_his[epoch, 0]

# def test_result(model_meta_dir, model_dir, data_set):
#     with tf.Session() as sess:
#         graph_importer = tf.train.import_meta_graph(model_meta_dir)
#         graph_importer.restore(sess, model_dir)
#         print(tf.get_collection())
#         x = tf.get_default_graph().get_tensor_by_name("x_input:0")
#         y = tf.get_default_graph().get_tensor_by_name("y_input:0")
#         loss_value, y_hat = sess.run([loss_func, y_pro], feed_dict={x: data_set['train_x'], y: data_set['train_y']})
#         accuracy_value = np.mean(np.argmax(y_hat, 1) == np.argmax(data_set['train_y'], 1))
