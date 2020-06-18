# -*- coding: utf-8 -*-
# @Author   ：Frank
# @Time     ：2018/8/27 11:34
# This version is tuning momentum and adagrad
# u = 0.9/0.5; v = 0.9/0.5


from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.training import optimizer


class MyAdaptiveOptimizerMAA(optimizer.Optimizer):

    def __init__(self, learning_rate, w1, w2, w3, m, n, m_adam, n_adam, beta1=0.9, beta2=0.999, beta1_power=0.9,
                 beta2_power=0.999, use_locking=False, name="MyAdaptiveOptimizerMAA"):

        super(MyAdaptiveOptimizerMAA, self).__init__(use_locking, name)
        self.lr = learning_rate
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.m = m
        self.n = n
        self.m_adam = m_adam
        self.n_adam = n_adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_power = beta1_power
        self.beta2_power = beta2_power

    def _get_beta_accumulators(self):
        if context.executing_eagerly():
            graph = None
        else:
            graph = ops.get_default_graph()
        return (self._get_non_slot_variable("beta1_power", graph=graph),
                self._get_non_slot_variable("beta2_power", graph=graph))

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=self.beta1_power, name="beta1_power", colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self.beta2_power, name="beta2_power", colocate_with=first_var)

        if self.m is not None:
            i = 0
            for v in var_list:
                self._get_or_make_slot_with_initializer(v, self.m[i], v.get_shape(), v.dtype.base_dtype, "m",
                                                        self._name)
                self._get_or_make_slot_with_initializer(v, self.n[i], v.get_shape(), v.dtype.base_dtype, "n",
                                                        self._name)
                self._get_or_make_slot_with_initializer(v, self.m_adam[i], v.get_shape(), v.dtype.base_dtype, "m_adam",
                                                        self._name)
                self._get_or_make_slot_with_initializer(v, self.n_adam[i], v.get_shape(), v.dtype.base_dtype, "n_adam",
                                                        self._name)
                i += 1
        else:
            for v in var_list:
                self._zeros_slot(v, "m", self._name)
                self._zeros_slot(v, "n", self._name)
                # init = init_ops.constant_initializer(0.1, dtype=v.dtype.base_dtype)
                # self._get_or_make_slot_with_initializer(v, init, v.get_shape(), v.dtype.base_dtype, "n", self._name)
                self._zeros_slot(v, "m_adam", self._name)
                self._zeros_slot(v, "n_adam", self._name)

    def _prepare(self):
        self.lr_t = ops.convert_to_tensor(self.lr, name="learning_rate")
        self.w1_t = ops.convert_to_tensor(self.w1, name="w1")
        self.w2_t = ops.convert_to_tensor(self.w2, name="w2")
        self.w3_t = ops.convert_to_tensor(self.w3, name="w3")
        self.beta1_t = ops.convert_to_tensor(self.beta1, name="beta1")
        self.beta2_t = ops.convert_to_tensor(self.beta2, name="beta2")

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self.lr_t, var.dtype.base_dtype)
        w1_t = math_ops.cast(self.w1_t, var.dtype.base_dtype)
        w2_t = math_ops.cast(self.w2_t, var.dtype.base_dtype)
        w3_t = math_ops.cast(self.w3_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self.beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self.beta2_t, var.dtype.base_dtype)

        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)

        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, 0.9 * m + grad, use_locking=self._use_locking)
        # adgrad part
        n = self.get_slot(var, "n")
        n_t = state_ops.assign(n, n + grad * grad, use_locking=self._use_locking)
        # adam part
        m_adam = self.get_slot(var, 'm_adam')
        m_adam_t = state_ops.assign(m_adam, beta1_t * m_adam + (1 - beta1_t) * grad, use_locking=self._use_locking)
        n_adam = self.get_slot(var, 'n_adam')
        n_adam_t = state_ops.assign(n_adam, beta2_t * n_adam + (1 - beta2_t) * grad * grad,
                                    use_locking=self._use_locking)

        # gradient part
        w4_t = (1.0 - math_ops.ceil((w1_t + w2_t + w3_t) / 3.0))

        n_adam_sqrt = math_ops.sqrt(n_adam_t)
        coefficient = math_ops.sqrt(1 - beta2_power) / (1 - beta1_power)
        var_update = state_ops.assign_sub(var, lr_t * (w1_t * m_t + w2_t * grad / math_ops.sqrt(n_t + 1e-8) + w3_t * coefficient * m_adam_t / (n_adam_sqrt + 1e-8) + w4_t * grad), use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, n_t, m_adam_t, n_adam_t])

    def _resource_apply_dense(self, grad, handle):
        pass

    def _resource_apply_sparse(self, grad, handle, indices):
        pass

    def _apply_sparse(self, grad, var):
        pass

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            beta1_power, beta2_power = self._get_beta_accumulators()
            with ops.colocate_with(beta1_power):
                update_beta1 = beta1_power.assign(
                    beta1_power * self.beta1, use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(
                    beta2_power * self.beta2, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                      name=name_scope)


class MyAdaptiveOptimizer(optimizer.Optimizer):

    def __init__(self, learning_rate, w1, w2, m, n, use_locking=False, name="MyAdaptiveOptimizer"):

        super(MyAdaptiveOptimizer, self).__init__(use_locking, name)
        self.lr = learning_rate
        self.w1 = w1
        self.w2 = w2
        self.m = m
        self.n = n

    def _create_slots(self, var_list):
        if self.m is not None:
            i = 0
            for v in var_list:
                self._get_or_make_slot_with_initializer(v, self.m[i], v.get_shape(), v.dtype.base_dtype, "m",
                                                        self._name)
                self._get_or_make_slot_with_initializer(v, self.n[i], v.get_shape(), v.dtype.base_dtype, "n",
                                                        self._name)
                i += 1
        else:
            for v in var_list:
                self._zeros_slot(v, "m", self._name)
                self._zeros_slot(v, "n", self._name)

    def _prepare(self):
        self.lr_t = ops.convert_to_tensor(self.lr, name="learning_rate")
        self.w1_t = ops.convert_to_tensor(self.w1, name="w1")
        self.w2_t = ops.convert_to_tensor(self.w2, name="w2")

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self.lr_t, var.dtype.base_dtype)
        w1_t = math_ops.cast(self.w1_t, var.dtype.base_dtype)
        w2_t = math_ops.cast(self.w2_t, var.dtype.base_dtype)

        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, 0.9 * m + grad, use_locking=self._use_locking)

        v = self.get_slot(var, "n")
        v_t = state_ops.assign(v, 0.9 * v + grad * grad, use_locking=self._use_locking)

        # var_update = state_ops.assign_sub(var, lr_t * (w1_t * m_t + w2_t * grad / (math_ops.sqrt(v_t) + 1e-6)))
        var_update = state_ops.assign_sub(var, lr_t * (w1_t * m + w2_t * grad / (math_ops.sqrt(v) + 1e-6) + grad))
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _resource_apply_dense(self, grad, handle):
        pass

    def _resource_apply_sparse(self, grad, handle, indices):
        pass

    def _apply_sparse(self, grad, var):
        pass


class MyMomentum(optimizer.Optimizer):

    def __init__(self, learning_rate, momentum, use_locking=False, name="MyMomentum"):
        super(MyMomentum, self).__init__(use_locking, name)
        self.lr = learning_rate
        self.momentum = momentum

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _prepare(self):
        self.lr_t = ops.convert_to_tensor(self.lr, name="learning_rate")
        self.momentum_t = ops.convert_to_tensor(self.momentum, name="Momentum")

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self.lr_t, var.dtype.base_dtype)
        momentum_t = math_ops.cast(self.momentum_t, var.dtype.base_dtype)

        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, momentum_t * m + grad, use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * m_t)

        return control_flow_ops.group(*[var_update, m_t])

    def _resource_apply_dense(self, grad, handle):
        pass

    def _resource_apply_sparse(self, grad, handle, indices):
        pass

    def _apply_sparse(self, grad, var):
        pass


class MyAdagrad(optimizer.Optimizer):

    def __init__(self, learning_rate, use_locking=False, name="MyAdagrad"):
        super(MyAdagrad, self).__init__(use_locking, name)
        self.lr = learning_rate

    def _create_slots(self, var_list):
        for v in var_list:
            init = init_ops.constant_initializer(0.1, dtype=v.dtype.base_dtype)
            self._get_or_make_slot_with_initializer(v, init, v.get_shape(), v.dtype.base_dtype, "n", self._name)

    def _prepare(self):
        self.lr_t = ops.convert_to_tensor(self.lr, name="learning_rate")

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self.lr_t, var.dtype.base_dtype)

        n = self.get_slot(var, "n")
        n_t = state_ops.assign(n, n + grad * grad, use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * grad / (math_ops.sqrt(n_t + 1e-6)))

        return control_flow_ops.group(*[var_update, n_t])

    def _resource_apply_dense(self, grad, handle):
        pass

    def _resource_apply_sparse(self, grad, handle, indices):
        pass

    def _apply_sparse(self, grad, var):
        pass


class MyAdam(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, name="MyAdam"):
        super(MyAdam, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _get_beta_accumulators(self):
        if context.executing_eagerly():
            graph = None
        else:
            graph = ops.get_default_graph()
        return (self._get_non_slot_variable("beta1_power_ado", graph=graph),
                self._get_non_slot_variable("beta2_power_ado", graph=graph),
                self._get_non_slot_variable("track_variable", graph=graph))

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=self._beta1, name="beta1_power_ado", colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._beta2, name="beta2_power_ado", colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._beta2, name="track_variable", colocate_with=first_var)
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        beta1_power, beta2_power, track_variable = self._get_beta_accumulators()

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        beta1_power_t = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power_t = math_ops.cast(beta2_power, var.dtype.base_dtype)

        beta2_power_next = state_ops.assign(beta2_power, beta2_power_t * beta2_t, use_locking=self._use_locking)
        beta1_power_next = state_ops.assign(beta1_power, beta1_power_t * beta1_t, use_locking=self._use_locking)

        lr = lr_t * math_ops.sqrt((1 - beta2_power_next)) / (1 - beta1_power_next)

        track_variable = state_ops.assign(track_variable, lr_t)

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, beta1_t * m + (1 - beta1_t) * grad, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        n = self.get_slot(var, "v")
        n_t = state_ops.assign(n, beta2_t * n + (1 - beta2_t) * (grad * grad), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr * m_t / (math_ops.sqrt(n_t) + epsilon_t),
                                          use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, n_t, beta1_power_t, beta2_power_t, track_variable])

    def _resource_apply_dense(self, grad, handle):
        pass

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _resource_apply_sparse(self, grad, handle, indices):
        pass
