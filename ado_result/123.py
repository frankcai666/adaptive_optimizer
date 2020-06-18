# -*- coding: utf-8 -*-
# @Author   ：Frank
# @Time     ：2018/11/1 14:24

import pandas as pd
import numpy as np

record_test_loss = []
record_train_loss = []


for i in range(0, 4800, 100):
     tmp = np.loadtxt('quadratic_agent_test_loss_ado_round_' + str(i) + '.csv', delimiter=',')
     record_test_loss.append(tmp[-1])
     tmp = np.loadtxt('quadratic_agent_train_loss_ado_round_' + str(i) + '.csv', delimiter=',')
     record_train_loss.append(tmp[-1])
np.savetxt('quadratic_agent_round_perform_train.csv', np.array(record_train_loss), delimiter=',')
np.savetxt('quadratic_agent_round_perform_test.csv', np.array(record_test_loss), delimiter=',')
