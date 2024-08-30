import copy
import numpy as np
from scipy.optimize import minimize_scalar
from utils.mkp_solver import mkp_interface


class TaskSelector:
    def __init__(self, params):
        self.params = params
        self.task_no = params.task_no

        self.opportunistic_list = ['PF', 'noUtil']
        self.opportunistic_flag = False

        if params.selection_mode in self.opportunistic_list:
            self.opportunistic_flag = True
            self.multipliers_lambda = np.ones(self.task_no) * params.sel_params.multiplier_init
            self.multipliers_mu = np.ones(self.task_no) * params.sel_params.multiplier_init
            self.aux_variables = np.zeros(self.task_no)
        elif params.selection_mode == 'RR':
            self.last_sel_task = 0

    def do_select(self, state, bw, mem, comput, iter):
        if self.params.selection_mode in self.opportunistic_list:
            if self.params.selection_mode == 'noUtil':
                noutil_flag = True
            else:
                noutil_flag = False
            task_values = np.zeros(self.task_no)
            for task_idx in range(self.task_no):
                if noutil_flag:
                    task_values[task_idx] = state[task_idx] * (1 + self.multipliers_mu[task_idx])
                else:
                    task_values[task_idx] = state[task_idx] * (self.multipliers_lambda[task_idx] + self.multipliers_mu[task_idx])
            sol = mkp_interface(task_values, bw, mem, comput, self.params.bw_max, self.params.mem_max, self.params.comput_max)
            self.update_multiplier(iter, state, sol, noutil_flag)
            return sol
        elif self.params.selection_mode == 'RR':
            return self.round_robin(bw, mem, comput)
        elif self.params.selection_mode == 'Greedy':
            return self.greedy(state, bw, mem, comput)
        elif self.params.selection_mode == 'noFL':
            return np.zeros(self.task_no)
        elif self.params.selection_mode == 'Bench':
            return np.ones(self.task_no)

    def round_robin(self, bw, mem, comput):
        end_flag = False
        sol = np.zeros(self.task_no)
        check = np.ones(self.task_no)
        curr_task = self.last_sel_task % self.task_no
        bw_budget = self.params.bw_max
        mem_budget = self.params.mem_max
        comput_budget = self.params.comput_max
        while not end_flag:
            if (bw_budget >= bw[curr_task]) & (mem_budget >= mem[curr_task]) & (comput_budget >= comput[curr_task]):
                if bw[curr_task] != 0:
                    sol[curr_task] = 1
                    bw_budget = bw_budget - bw[curr_task]
                    mem_budget = mem_budget - mem[curr_task]
                    comput_budget = comput_budget - comput[curr_task]
                check[curr_task] -= 1
                curr_task = (curr_task + 1) % self.task_no
            else:
                end_flag = True

            if np.sum(check) == 0:
                end_flag = True
        self.last_sel_task = curr_task

        return sol

    def greedy(self, state, bw, mem, comput):
        state_=copy.deepcopy(state)
        end_flag = False
        sol = np.zeros(self.task_no)
        check = np.ones(self.task_no)
        bw_budget = self.params.bw_max
        mem_budget = self.params.mem_max
        comput_budget = self.params.comput_max
        while not end_flag:
            curr_task = np.argmax(state_)
            if (state_[curr_task] > 0) & \
                    (bw_budget >= bw[curr_task]) & \
                    (mem_budget >= mem[curr_task]) & \
                    (comput_budget >= comput[curr_task]):
                sol[curr_task] = 1
                check[curr_task] -= 1
                state_[curr_task] = -1
                bw_budget = bw_budget - bw[curr_task]
                mem_budget = mem_budget - mem[curr_task]
                comput_budget = comput_budget - comput[curr_task]
            else:
                end_flag = True

            if np.sum(check) == 0:
                end_flag = True

        return sol

    def utility_func(self, x):
        if self.params.selection_mode == 'PF':
            return np.log(x)

    def aux_update(self, x, multiplier):
        return -(self.utility_func(x) - multiplier * x)

    def update_multiplier(self, iter, state, sol, noutil_flag):
        for idx in range(self.task_no):
            if not noutil_flag:
                res = minimize_scalar(self.aux_update, args=(self.multipliers_lambda[idx]), bounds=(0, 100), method='bounded')
                self.aux_variables[idx] = res.x
                self.multipliers_lambda[idx] = np.maximum(self.multipliers_lambda[idx] - self.params.sel_params.step_size * (sol[idx] * state[idx] - self.aux_variables[idx]), 0)

            self.multipliers_mu[idx] = np.maximum(self.multipliers_mu[idx] - self.params.sel_params.step_size * (sol[idx] * state[idx] - self.params.sel_params.minimum_participants[idx]), 0)
