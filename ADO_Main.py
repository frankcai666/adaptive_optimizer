# -*- coding: utf-8 -*-
# @Author   ：Frank
# @Time     ：2018/8/30 9:43

import os
import tensorflow as tf
import numpy as np
from Problem_Tools import quadratic, rl_training_quadratic, rl_testing_quadratic, one_layer_mnist, rl_training_nn, rl_testing_nn
from DQN_Agent_new import DDQNAgent
from tensorflow.examples.tutorials.mnist import input_data

RESULT_PATH = 'ado_result/'
GRAPH_PATH = 'graph/'
N_FEATURE = 18

config = tf.ConfigProto(
    device_count={"CPU": 24},
    inter_op_parallelism_threads=24,
    intra_op_parallelism_threads=24)


def main(argv=None):
    os.environ["KMP_BLOCKTIME"] = '0'
    os.environ["KMP_SETTINGS"] = '0'
    # os.environ["KMP_AFFINITY"] = 'granularity=fine,verbose,compact,1,0'
    os.environ["OMP_NUM_THREADS"] = '24'
    optimizer_name = 'ado'
    case_name_set = ['quadratic']

    for case_name in case_name_set:
        if case_name == 'com':
            # data extract
            mnist_data = input_data.read_data_sets('mnist_data/', one_hot=True)
            mnist_train = mnist_data.train
            validate_x = mnist_data.validation.images
            validate_y = mnist_data.validation.labels
            input_node = validate_x.shape[1]
            output_node = validate_y.shape[1]
            hidden_node = 50
            GAP = 10
            epoch_max = 200
            episode_max = 8001
            agent = DDQNAgent(n_features=N_FEATURE, config=config)
            while agent.memory_pointer <= 1000:
                print('\n ******** Initialize agent memory ******** ')
                loss_graph = one_layer_mnist(input_node, output_node, hidden_node, train_flag=False)
                agent, loss_his, accuracy_his = rl_training_nn(loss_graph, mnist_train, validate_x, validate_y, agent, config, epoch_max, GAP, initial_flag=True)
            np.savetxt(RESULT_PATH + case_name + '_agent_initial_memory_.csv', np.array(agent.memory), delimiter=',')
            rl_loss_record = []
            rl_accuracy_record = []
            for episode in range(0, episode_max):
                loss_graph = one_layer_mnist(input_node, output_node, hidden_node, train_flag=False)
                agent, loss_his, accuracy_his = rl_training_nn(loss_graph, mnist_train, validate_x, validate_y, agent, config, epoch_max, GAP, initial_flag=False)
                agent.epsilon = agent.epsilon + agent.epsilon_increment if agent.epsilon < agent.epsilon_max else agent.epsilon_max
                if episode % 100 == 0:
                    print('\n******************** ' + str(int(episode)) + ' th rl on train problem******************')
                    loss_graph = one_layer_mnist(input_node, output_node, hidden_node, train_flag=True)
                    loss_his, accuracy_his, loss_perform, accuracy_perform = rl_testing_nn(loss_graph, mnist_train, validate_x, validate_y, agent, config, epoch_max, GAP)
                    rl_loss_record.append(loss_perform)
                    rl_accuracy_record.append(accuracy_perform)
                    np.savetxt(RESULT_PATH + case_name + '_agent_train_validate_loss_' + optimizer_name + '_round_' +
                               str(episode) + '.csv', loss_his, delimiter=',')
                    np.savetxt(RESULT_PATH + case_name + '_agent_train_validate_accuracy_' + optimizer_name + '_round_' +
                               str(episode) + '.csv', accuracy_his, delimiter=',')
                    agent.save_recover_model(episode)
                    print('\n******************** rl on test problem ******************')
                    loss_graph = one_layer_mnist(input_node, output_node, hidden_node, train_flag=False)
                    loss_his, accuracy_his, loss_perform, accuracy_perform = rl_testing_nn(loss_graph, mnist_train, validate_x, validate_y, agent, config, epoch_max, GAP)
                    np.savetxt(RESULT_PATH + case_name + '_agent_test_validate_loss_' + optimizer_name + '_round_' +
                               str(episode) + '.csv', loss_his, delimiter=',')
                    np.savetxt(RESULT_PATH + case_name + '_agent_test_validate_accuracy_' + optimizer_name + '_round_' +
                               str(episode) + '.csv', accuracy_his, delimiter=',')
                    print('RL\'s total_loss: %g, training_loss: %g' % (agent.loss_his[-1], agent.train_loss_his[-1]))

            np.savetxt(RESULT_PATH + case_name + '_agent_loss_.csv', np.array(agent.loss_his), delimiter=',')
            np.savetxt(RESULT_PATH + case_name + '_agent_memory_.csv', np.array(agent.memory), delimiter=',')
            np.savetxt(RESULT_PATH + case_name + '_agent_action_value_.csv', np.array(agent.action_value_record),
                       delimiter=',')
            np.savetxt(RESULT_PATH + case_name + '_agent_round_loss_.csv', np.array(rl_loss_record), delimiter=',')
            np.savetxt(RESULT_PATH + case_name + '_agent_round_accuracy_.csv', np.array(rl_accuracy_record), delimiter=',')
        elif case_name == 'mnist':
            # data extract
            mnist_data = input_data.read_data_sets('mnist_data/', one_hot=True)
            mnist_train = mnist_data.train
            validate_x = mnist_data.validation.images
            validate_y = mnist_data.validation.labels
            input_node = validate_x.shape[1]
            output_node = validate_y.shape[1]
            hidden_node = 50
            GAP = 10
            epoch_max = 200
            episode_max = 10001
            agent = DDQNAgent(n_features=N_FEATURE, config=config)
            while agent.memory_pointer <= 1000:
                print('\n ******** Initialize agent memory ******** ')
                loss_graph = one_layer_mnist(input_node, output_node, hidden_node, train_flag=False)
                agent, loss_his, accuracy_his = rl_training_nn(loss_graph, mnist_train, validate_x, validate_y, agent, config, epoch_max, GAP, initial_flag=True)
            np.savetxt(RESULT_PATH + case_name + '_agent_initial_memory_.csv', np.array(agent.memory), delimiter=',')
            rl_loss_record = []
            rl_accuracy_record = []
            for episode in range(0, episode_max):
                loss_graph = one_layer_mnist(input_node, output_node, hidden_node, train_flag=False)
                agent, loss_his, accuracy_his = rl_training_nn(loss_graph, mnist_train, validate_x, validate_y, agent, config, epoch_max, GAP, initial_flag=False)
                agent.epsilon = agent.epsilon + agent.epsilon_increment if agent.epsilon < agent.epsilon_max else agent.epsilon_max
                if episode % 1000 == 0:
                    print('\n******************** ' + str(int(episode)) + ' th rl on train problem******************')
                    loss_graph = one_layer_mnist(input_node, output_node, hidden_node, train_flag=True)
                    loss_his, accuracy_his, loss_perform, accuracy_perform = rl_testing_nn(loss_graph, mnist_train, validate_x, validate_y, agent, config, epoch_max, GAP)
                    rl_loss_record.append(loss_perform)
                    rl_accuracy_record.append(accuracy_perform)
                    np.savetxt(RESULT_PATH + case_name + '_agent_train_validate_loss_' + optimizer_name + '_round_' +
                               str(episode) + '.csv', loss_his, delimiter=',')
                    np.savetxt(RESULT_PATH + case_name + '_agent_train_validate_accuracy_' + optimizer_name + '_round_' +
                               str(episode) + '.csv', accuracy_his, delimiter=',')
                    agent.save_recover_model(episode)
                    print('\n******************** rl on test problem ******************')
                    loss_graph = one_layer_mnist(input_node, output_node, hidden_node, train_flag=False)
                    loss_his, accuracy_his, loss_perform, accuracy_perform = rl_testing_nn(loss_graph, mnist_train, validate_x, validate_y, agent, config, epoch_max, GAP)
                    np.savetxt(RESULT_PATH + case_name + '_agent_test_validate_loss_' + optimizer_name + '_round_' +
                               str(episode) + '.csv', loss_his, delimiter=',')
                    np.savetxt(RESULT_PATH + case_name + '_agent_test_validate_accuracy_' + optimizer_name + '_round_' +
                               str(episode) + '.csv', accuracy_his, delimiter=',')
                    print('RL\'s total_loss: %g, training_loss: %g' % (agent.loss_his[-1], agent.train_loss_his[-1]))

            np.savetxt(RESULT_PATH + case_name + '_agent_loss_.csv', np.array(agent.loss_his), delimiter=',')
            np.savetxt(RESULT_PATH + case_name + '_agent_memory_.csv', np.array(agent.memory), delimiter=',')
            np.savetxt(RESULT_PATH + case_name + '_agent_action_value_.csv', np.array(agent.action_value_record),
                       delimiter=',')
            np.savetxt(RESULT_PATH + case_name + '_agent_round_loss_.csv', np.array(rl_loss_record), delimiter=',')
            np.savetxt(RESULT_PATH + case_name + '_agent_round_accuracy_.csv', np.array(rl_accuracy_record), delimiter=',')
        elif case_name == 'quadratic':
            batch_size = 128
            num_dim = 10
            GAP = 5
            epoch_max = 100
            episode_max = 10001
            # initialize the DQN Agent and its memory
            agent = DDQNAgent(n_features=N_FEATURE, config=config)
            while agent.memory_pointer <= 1000:
                print('\n ******** Initialize agent memory ******** ')
                loss_graph = quadratic(batch_size, num_dim, train_flag=False)
                agent, loss_his = rl_training_quadratic(loss_graph, agent, config, epoch_max, epoch_gap=GAP, initial_flag=True)
            np.savetxt(RESULT_PATH + case_name + '_agent_initial_memory_.csv', np.array(agent.memory), delimiter=',')
            rl_perform_record = []
            for episode in range(0, episode_max):
                # print('\n******** This is the ' + str(episode) + ' th round for ' + case_name + ' ********')
                # print('The egreedy value is %g' % agent.epsilon)
                # initialize the loss graph
                loss_graph = quadratic(batch_size, num_dim, train_flag=False)
                agent, loss_his = rl_training_quadratic(loss_graph, agent, config, epoch_max, epoch_gap=GAP, initial_flag=False)
                # np.savetxt(RESULT_PATH + case_name + '_training_loss_' + optimizer_name + '_round_' + str(episode)
                #            + '.csv', loss_his, delimiter=',')
                agent.epsilon = agent.epsilon + agent.epsilon_increment if agent.epsilon < agent.epsilon_max else agent.epsilon_max
                if episode % 1000 == 0:
                    print('\n******************** ' + str(int(episode)) + ' th rl on train problem******************')
                    loss_graph = quadratic(batch_size, num_dim, train_flag=True)
                    loss_his_agent, perform = rl_testing_quadratic(loss_graph, agent, config, epoch_max, epoch_gap=GAP)
                    rl_perform_record.append(perform)
                    np.savetxt(RESULT_PATH + case_name + '_agent_train_loss_' + optimizer_name + '_round_' +
                               str(episode) + '.csv', loss_his_agent, delimiter=',')
                    agent.save_recover_model(episode)
                    print('\n******************** rl on test problem ******************')
                    loss_graph = quadratic(batch_size, num_dim, train_flag=False)
                    loss_his_agent, perform = rl_testing_quadratic(loss_graph, agent, config, epoch_max, epoch_gap=GAP)
                    rl_perform_record.append(perform)
                    np.savetxt(RESULT_PATH + case_name + '_agent_test_loss_' + optimizer_name + '_round_' +
                               str(episode) + '.csv', loss_his_agent, delimiter=',')
                    print('RL\'s total_loss: %g, training_loss: %g' % (agent.loss_his[-1], agent.train_loss_his[-1]))

            np.savetxt(RESULT_PATH + case_name + '_agent_loss_.csv', np.array(agent.loss_his), delimiter=',')
            np.savetxt(RESULT_PATH + case_name + '_agent_memory_.csv', np.array(agent.memory), delimiter=',')
            np.savetxt(RESULT_PATH + case_name + '_agent_action_value_.csv', np.array(agent.action_value_record),
                       delimiter=',')
            np.savetxt(RESULT_PATH + case_name + '_agent_round_perform_.csv', np.array(rl_perform_record), delimiter=',')


if __name__ == "__main__":
    tf.app.run()
