# -*- coding: utf-8 -*-
# @Author   ：Frank
# @Time     ：2018/8/22 11:15


import numpy as np
import tensorflow as tf
from time import time
from tensorflow.python.framework import ops

np.random.seed(1)
log_dir = 'DQN_logs/'
SAVE_PATH = 'MODEL_logs/'


class DQNAgent:

    def __init__(
            self, n_features, config, learning_rate=0.001, reward_decay=0.9, e_greedy=0.9, replace_target_iter=200,
            memory_size=30000, final_n_memory_size=3000, final_p_memory_size=3000, batch_size=16,
            e_greedy_increment=0.0003, double_flag=False
    ):
        # problem parameter
        self.step = 0
        # self.w1_set = [0.9, 0.7, 0.5, 0.3, 0.1]
        self.w1_set = [1.0, 0.0]
        self.w2_set = [1.0, 0.0]
        self.w3_set = [1.0, 0.0]
        self.lr_set = [1.0, 1e-1, 1e-2, 1e-3]
        self.action_set = []
        for lr in self.lr_set:
            for w1 in self.w1_set:
                for w2 in self.w1_set:
                    for w3 in self.w3_set:
                        self.action_set.append([lr, w1, w2, w3])
        np.savetxt('action_set.csv', np.array(self.action_set), delimiter=',')
        # DQN parameter
        self.n_actions = len(self.action_set)
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        # total memory
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))
        self.memory_pointer = 0
        # final negative memory
        self.final_n_memory_size = final_n_memory_size
        self.final_n_memory = np.zeros((self.final_n_memory_size, n_features * 2 + 3))
        self.final_n_memory_pointer = 0
        # final positive memory
        self.final_p_memory_size = final_p_memory_size
        self.final_p_memory = np.zeros((self.final_p_memory_size, n_features * 2 + 3))
        self.final_p_memory_pointer=0

        self.action_value_record = []

        if double_flag is True:
            self.agent_graph, self.train_op, self.errors, self.q_eval, self.q_next, self.s, self.s_, self.r, self.a, self.d, self.double_q_value, self.tf_is_train, self.param_replace_op = self.build_dqn()
        else:
            self.agent_graph, self.train_op, self.errors, self.q_eval, self.q_next, self.s, self.s_, self.r, self.a, self.d, self.tf_is_train, self.param_replace_op = self.build_dqn()

        self.learn_step = 0

        with self.agent_graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=100)
            self.sess = tf.Session(graph=self.agent_graph, config=config)
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
            # print(self.sess.run(tf.report_uninitialized_variables()))
            self.sess.run(tf.global_variables_initializer())
            # a1 = self.sess.run(tf.global_variables())
            # graph = self.train_op.graph
            # print(self.sess.run(graph.get_tensor_by_name("eval_net/l1/kernel:0"))[0])

        self.loss_his = []

        self.train_loss_his = []

    def build_dqn(self):
        agent_graph = tf.Graph()
        with agent_graph.as_default():
            tf.set_random_seed(2)

            s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input state
            s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
            r = tf.placeholder(tf.float32, [None, ], name='r')  # input reward
            a = tf.placeholder(tf.int32, [None, ], name='a')  # input action.
            d = tf.placeholder(tf.float32, [None, ], name='d')  # input whether done
            tf_is_train = tf.placeholder(tf.bool, None, name='train_flag')  # for batch normalization

            w_initializer, b_initializer = tf.random_normal_initializer(0.001, 0.01), tf.constant_initializer(0.001)

            with tf.variable_scope('eval_net'):
                s_norm = tf.layers.batch_normalization(s, training=tf_is_train, name='input_norm')
                e1_dense = tf.layers.dense(s_norm, units=self.n_features, kernel_initializer=w_initializer,
                                           bias_initializer=b_initializer, name='l1')
                e1_norm = tf.layers.batch_normalization(e1_dense, momentum=0.6, training=tf_is_train, name='l1_norm')
                e1_ac = tf.nn.relu(e1_norm, name='l1_ac')

                tf.summary.histogram(e1_norm.name, e1_norm)

                e2_dense = tf.layers.dense(e1_ac, units=self.n_features, kernel_initializer=w_initializer,
                                           bias_initializer=b_initializer, name='l2')
                e2_norm = tf.layers.batch_normalization(e2_dense, momentum=0.6, training=tf_is_train, name='l2_norm')
                e2_ac = tf.nn.relu(e2_norm, name='l2_ac')

                tf.summary.histogram(e2_norm.name, e2_norm)

                q_eval = tf.layers.dense(e2_ac, units=self.n_actions, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name='l3')

                tf.summary.histogram(q_eval.name, q_eval)
                # e1 = tf.layers.dense(s_norm, units=self.n_features, activation=tf.nn.relu, kernel_initializer=w_initializer,
                #                      bias_initializer=b_initializer, name='l1')
                # e2 = tf.layers.dense(e1, units=self.n_features, activation=tf.nn.relu, kernel_initializer=w_initializer,
                #                      bias_initializer=b_initializer, name='l2')
                # q_eval = tf.layers.dense(e2, units=self.n_actions, kernel_initializer=w_initializer,
                #                          bias_initializer=b_initializer, name='l3')

            with tf.variable_scope('target_net'):
                s__norm = tf.layers.batch_normalization(s_, training=tf_is_train, name='input_norm')
                t1_dense = tf.layers.dense(s__norm, units=self.n_features, kernel_initializer=w_initializer,
                                           bias_initializer=b_initializer, name='l1')
                t1_norm = tf.layers.batch_normalization(t1_dense, momentum=0.6, training=tf_is_train, name='l1_norm')
                t1_ac = tf.nn.relu(t1_norm, name='l1_ac')

                tf.summary.histogram(t1_norm.name, t1_norm)

                t2_dense = tf.layers.dense(t1_ac, units=self.n_features, kernel_initializer=w_initializer,
                                           bias_initializer=b_initializer, name='l2')
                t2_norm = tf.layers.batch_normalization(t2_dense, momentum=0.6, training=tf_is_train, name='l2_norm')
                t2_ac = tf.nn.relu(t2_norm, name='l2_ac')

                tf.summary.histogram(t2_norm.name, t2_norm)

                q_next = tf.layers.dense(t2_ac, units=self.n_actions, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name='l3')

                tf.summary.histogram(q_next.name, q_next)
                # t1 = tf.layers.dense(s__norm, units=self.n_features, activation=tf.nn.relu, kernel_initializer=w_initializer,
                #                      bias_initializer=b_initializer, name='l1')
                # t2 = tf.layers.dense(t1, units=self.n_features, activation=tf.nn.relu, kernel_initializer=w_initializer,
                #                      bias_initializer=b_initializer, name='l2')
                # q_next = tf.layers.dense(t2, units=self.n_actions, kernel_initializer=w_initializer,
                #                          bias_initializer=b_initializer, name='l3')

            with tf.variable_scope('q_target'):
                q_target = r + tf.multiply(d, self.gamma * tf.reduce_max(q_next, axis=1, name='Q_max'))
                q_target = tf.stop_gradient(q_target)

            with tf.variable_scope('loss'):
                a_indices = tf.stack([tf.range(tf.shape(a)[0], dtype=tf.int32), a], axis=1)
                q_predict = tf.gather_nd(params=q_eval, indices=a_indices)
                # self.perloss = tf.squared_difference(q_target, q_predict)
                diff = tf.abs(q_target - q_predict)
                square_part = tf.clip_by_value(diff, 0.0, 1.0)
                linear_part = diff - square_part
                errors = tf.square(square_part) + linear_part

                loss = tf.reduce_mean(errors)
                # loss = tf.reduce_mean(tf.squared_difference(q_target, q_predict, name='TD_error'))

            tf.summary.scalar(loss.name, loss)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.RMSPropOptimizer(self.lr).minimize(loss)

            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
            with tf.variable_scope('param_replacement'):
                param_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        return agent_graph, train_op, errors, q_eval, q_next, s, s_, r, a, d, tf_is_train, param_replace_op

    def store_hist(self, s, a, r, s_, done, initial_flag):
        # param done - 0: done; 1: under optimizing
        # if r_pre > 0 and self.memory_pointer > 0:
        #     index_pre = (self.memory_pointer - 1) % self.memory_size
        #     self.memory[index_pre, self.n_features + 1] = r_pre
        memory = np.hstack((s, [a, r, done], s_))
        index = self.memory_pointer % self.memory_size
        self.memory[index, :] = memory
        self.memory_pointer += 1
        # store redundant memory into ram and perform gradient
        if done == 0:
            # if r == -1:
            index = self.final_n_memory_pointer % self.final_n_memory_size
            self.final_n_memory[index, :] = memory
            self.final_n_memory_pointer += 1
            # else:
            #     index = self.final_p_memory_pointer % self.final_p_memory_size
            #     self.final_p_memory[index, :] = memory
            #     self.final_p_memory_pointer += 1
        if initial_flag is False:
            self.learn()

    def choose_action(self, state, train_flag):
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon or train_flag is False:
            with self.agent_graph.as_default():
                action_value = self.sess.run(self.q_eval, feed_dict={self.s: state, self.tf_is_train: False})
            action_index = np.argmax(action_value)
            if train_flag is False:
                self.action_value_record.append(np.append(np.hstack([state, action_value])[0], action_index))
        else:
            action_index = np.random.randint(0, self.n_actions)
        lr_value, w1_value, w2_value, w_3value = self.action_set[action_index]

        return action_index, lr_value, w1_value, w2_value, w_3value

    def learn(self):

        if self.memory_pointer > self.memory_size:
            sample_index = np.random.choice(self.memory_size, self.batch_size)
            memory_test = self.memory
        else:
            memory_test = self.memory[0:self.memory_pointer, :]
            sample_index = np.random.choice(self.memory_pointer, self.batch_size)

        if self.final_n_memory_pointer > self.final_n_memory_size:
            final_n_sample_index = np.random.choice(self.final_n_memory_size, self.batch_size)
        else:
            final_n_sample_index = np.random.choice(self.final_n_memory_pointer, self.batch_size)

        # if self.final_p_memory_pointer > self.final_p_memory_size:
        #     final_p_sample_index = np.random.choice(self.final_p_memory_size, self.batch_size)
        # else:
        #     final_p_sample_index = np.random.choice(self.final_p_memory_pointer, self.batch_size)

        batch_memory1 = self.memory[sample_index, :]
        batch_memory2 = self.final_n_memory[final_n_sample_index, :]
        # batch_memory3 = self.final_p_memory[final_p_sample_index, :]
        batch_memory = np.vstack((batch_memory1, batch_memory2))
        # batch_memory = np.vstack((batch_memory1, batch_memory2, batch_memory3))
        # for i in range(1):
        with self.agent_graph.as_default():
            if self.learn_step % self.replace_target_iter == 0:
                self.sess.run(self.param_replace_op)
                print('\n^^^^^^^^parameter has been replaced^^^^^^^^\n')

            _, loss_value = self.sess.run(
                [self.train_op, self.errors],
                feed_dict={
                    self.s: batch_memory[:, :self.n_features], self.a: batch_memory[:, self.n_features],
                    self.r: batch_memory[:, self.n_features + 1], self.s_: batch_memory[:, -self.n_features:],
                    self.d: batch_memory[:, self.n_features + 2], self.tf_is_train: True
                }
            )

            self.train_loss_his.append(np.mean(loss_value))

            loss_value, summary, action_value = self.sess.run(
                [self.errors, self.merged, self.q_eval],
                feed_dict={
                    self.s: memory_test[:, :self.n_features], self.a: memory_test[:, self.n_features],
                    self.r: memory_test[:, self.n_features + 1], self.s_: memory_test[:, -self.n_features:],
                    self.d: memory_test[:, self.n_features + 2], self.tf_is_train: False
                }
            )
            self.writer.add_summary(summary, self.learn_step)

        if self.learn_step % 50000 == 0:
            np.savetxt('check_loss_value.csv', loss_value, delimiter=',')
            np.savetxt('check_action_value.csv', action_value, delimiter=',')
            np.savetxt('check_memory.csv', self.memory, delimiter=',')
        self.loss_his.append(np.mean(loss_value))

        self.learn_step += 1


class DDQNAgent(DQNAgent):

    def __init__(
            self, n_features, config, learning_rate=0.001, reward_decay=0.9, e_greedy=0.9, replace_target_iter=200,
            memory_size=20000, final_n_memory_size=2000, final_p_memory_size=2000, batch_size=16,
            e_greedy_increment=0.0003, double_flag=True
    ):
        super(DDQNAgent, self).__init__(
            n_features, config, learning_rate, reward_decay, e_greedy, replace_target_iter, memory_size,
            final_n_memory_size, final_p_memory_size, batch_size, e_greedy_increment, double_flag
        )

    def build_dqn(self):
        print('I am double DQN!!!')
        agent_graph = tf.Graph()
        with agent_graph.as_default():
            tf.set_random_seed(1)

            s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input state
            s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
            r = tf.placeholder(tf.float32, [None, ], name='r')  # input reward
            a = tf.placeholder(tf.int32, [None, ], name='a')  # input action.
            d = tf.placeholder(tf.float32, [None, ], name='d')  # input whether done
            double_q = tf.placeholder(tf.float32, [None, ], name='double_q_value')
            tf_is_train = tf.placeholder(tf.bool, None, name='train_flag')  # for batch normalization

            w_initializer, b_initializer = tf.random_normal_initializer(0.001, 0.01), tf.constant_initializer(0.001)

            with tf.variable_scope('eval_net'):
                s_norm = tf.layers.batch_normalization(s, training=tf_is_train, name='input_norm')
                e1_dense = tf.layers.dense(s_norm, units=self.n_features, kernel_initializer=w_initializer,
                                           bias_initializer=b_initializer, name='l1')
                e1_norm = tf.layers.batch_normalization(e1_dense, training=tf_is_train, name='l1_norm')
                e1_ac = tf.nn.relu(e1_norm, name='l1_ac')

                tf.summary.histogram(e1_norm.name, e1_norm)

                e2_dense = tf.layers.dense(e1_ac, units=self.n_features, kernel_initializer=w_initializer,
                                           bias_initializer=b_initializer, name='l2')
                e2_norm = tf.layers.batch_normalization(e2_dense, training=tf_is_train, name='l2_norm')
                e2_ac = tf.nn.relu(e2_norm, name='l2_ac')

                tf.summary.histogram(e2_norm.name, e2_norm)

                q_eval = tf.layers.dense(e2_ac, units=self.n_actions, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name='l3')

                tf.summary.histogram(q_eval.name, q_eval)

            with tf.variable_scope('target_net'):
                s__norm = tf.layers.batch_normalization(s_, training=tf_is_train, name='input_norm')
                t1_dense = tf.layers.dense(s__norm, units=self.n_features, kernel_initializer=w_initializer,
                                           bias_initializer=b_initializer, name='l1')
                t1_norm = tf.layers.batch_normalization(t1_dense, training=tf_is_train, name='l1_norm')
                t1_ac = tf.nn.relu(t1_norm, name='l1_ac')

                tf.summary.histogram(t1_norm.name, t1_norm)

                t2_dense = tf.layers.dense(t1_ac, units=self.n_features, kernel_initializer=w_initializer,
                                           bias_initializer=b_initializer, name='l2')
                t2_norm = tf.layers.batch_normalization(t2_dense, training=tf_is_train, name='l2_norm')
                t2_ac = tf.nn.relu(t2_norm, name='l2_ac')

                tf.summary.histogram(t2_norm.name, t2_norm)

                q_next = tf.layers.dense(t2_ac, units=self.n_actions, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name='l3')

                tf.summary.histogram(q_next.name, q_next)

            with tf.variable_scope('q_target'):
                q_target = r + self.gamma * double_q
                q_target = tf.stop_gradient(q_target)

            with tf.variable_scope('loss'):
                a_indices = tf.stack([tf.range(tf.shape(a)[0], dtype=tf.int32), a], axis=1)
                q_predict = tf.gather_nd(params=q_eval, indices=a_indices)
                # self.perloss = tf.squared_difference(q_target, q_predict)
                diff = tf.abs(q_target - q_predict)
                square_part = tf.clip_by_value(diff, 0.0, 1.0)
                linear_part = diff - square_part
                errors = tf.square(square_part) + linear_part

                loss = tf.reduce_mean(errors)
                # loss = tf.reduce_mean(tf.squared_difference(q_target, q_predict, name='TD_error'))

            tf.summary.scalar(loss.name, loss)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.RMSPropOptimizer(self.lr).minimize(loss)

            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
            with tf.variable_scope('param_replacement'):
                param_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        return agent_graph, train_op, errors, q_eval, q_next, s, s_, r, a, d, double_q, tf_is_train, param_replace_op

    def learn(self):

        if self.memory_pointer > self.memory_size:
            sample_index = np.random.choice(self.memory_size, self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_pointer, self.batch_size)

        if self.final_n_memory_pointer > self.final_n_memory_size:
            final_n_sample_index = np.random.choice(self.final_n_memory_size, self.batch_size)
        else:
            final_n_sample_index = np.random.choice(self.final_n_memory_pointer, self.batch_size)

        # if self.final_p_memory_pointer > self.final_p_memory_size:
        #     final_p_sample_index = np.random.choice(self.final_p_memory_size, self.batch_size)
        # else:
        #     final_p_sample_index = np.random.choice(self.final_p_memory_pointer, self.batch_size)

        batch_memory1 = self.memory[sample_index, :]
        batch_memory2 = self.final_n_memory[final_n_sample_index, :]
        # batch_memory3 = self.final_p_memory[final_p_sample_index, :]
        batch_memory = np.vstack((batch_memory1, batch_memory2))
        # batch_memory = np.vstack((batch_memory1, batch_memory2, batch_memory3))
        # for i in range(1):
        with self.agent_graph.as_default():
            if self.learn_step % self.replace_target_iter == 0 and self.learn_step >= 1000:
                self.sess.run(self.param_replace_op)
                print('\n^^^^^^^^parameter has been replaced^^^^^^^^\n')

            eval_next, target_next = self.sess.run(
                [self.q_eval, self.q_next], feed_dict={
                    self.s: batch_memory[:, -self.n_features:], self.s_: batch_memory[:, -self.n_features:],
                    self.tf_is_train: True
                }
            )
            eval_next_max_index = np.argmax(eval_next, axis=1)
            batch_index = np.arange(2 * self.batch_size, dtype=np.int32)
            double_q_value = target_next[batch_index, eval_next_max_index]

            _, loss_value = self.sess.run(
                [self.train_op, self.errors],
                feed_dict={
                    self.s: batch_memory[:, :self.n_features], self.a: batch_memory[:, self.n_features],
                    self.r: batch_memory[:, self.n_features + 1], self.s_: batch_memory[:, -self.n_features:],
                    self.d: batch_memory[:, self.n_features + 2], self.double_q_value: double_q_value,
                    self.tf_is_train: True, self.s_: batch_memory[:, -self.n_features:]
                }
            )
            self.train_loss_his.append(np.mean(loss_value))

            # for testing
            if self.learn_step % 10000 == 0:
                if self.memory_pointer > self.memory_size:
                    sample_index = np.random.choice(self.memory_size, self.batch_size * 5)

                else:
                    sample_index = np.random.choice(self.memory_pointer, self.batch_size)
                memory_test = self.memory[sample_index, :]
                # For testing performance
                eval_next, target_next = self.sess.run(
                    [self.q_eval, self.q_next], feed_dict={
                        self.s: memory_test[:, -self.n_features:], self.s_: memory_test[:, -self.n_features:],
                        self.tf_is_train: False
                    }
                )
                eval_next_max_index = np.argmax(eval_next, axis=1)
                batch_index = np.arange(eval_next_max_index.shape[0], dtype=np.int32)
                double_q_value = target_next[batch_index, eval_next_max_index]

                loss_value, summary, action_value = self.sess.run(
                    [self.errors, self.merged, self.q_eval],
                    feed_dict={
                        self.s: memory_test[:, :self.n_features], self.a: memory_test[:, self.n_features],
                        self.r: memory_test[:, self.n_features + 1], self.s_: memory_test[:, -self.n_features:],
                        self.d: memory_test[:, self.n_features + 2], self.double_q_value: double_q_value,
                        self.tf_is_train: False, self.s_: memory_test[:, -self.n_features:]
                    }
                )

                self.writer.add_summary(summary, self.learn_step)
                self.loss_his.append(np.mean(loss_value))

                # if self.learn_step % 1000 == 0:
                #     np.savetxt('check_loss_value.csv', loss_value, delimiter=',')
                #     np.savetxt('check_action_value.csv', action_value, delimiter=',')
                #     np.savetxt('check_memory.csv', self.memory, delimiter=',')

        self.learn_step += 1

    def save_recover_model(self, step):
        if step == 0:
            self.saver.save(self.sess, SAVE_PATH + 'model.ckpt', global_step=step, write_meta_graph=True)
        else:
            self.saver.save(self.sess, SAVE_PATH + 'model.ckpt', global_step=step, write_meta_graph=False)