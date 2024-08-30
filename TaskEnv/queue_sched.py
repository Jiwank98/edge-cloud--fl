import torch
import numpy as np
import random


class QueueEnv:
    def __init__(self, parent, scenario_no, params):
        self.arrive_iter = parent.parent.params.arrive_iter * parent.params.sub_iteration
        self.iteration = 0
        self.parent = parent
        self.total_iterations = parent.total_iterations
        self.params = params

        self.discounted_factor = params.task_specific_params.discount_factor

        # scenario setting
        self.user_no, self.p_min, self.p_max = self.scenarios(scenario_no)

        ####################################################
        # state related part
        ####################################################
        self.p_no = params.task_specific_params.p_no
        self.p_partition = np.linspace(self.p_min, self.p_max, self.p_no + 1)

        self.x_no = params.task_specific_params.x_no
        self.x_max = self.x_no - 1

        self.curr_state = None
        self.next_state = self.gen_state()

    def gen_state(self):
        state = np.zeros((self.user_no, 2))
        state[:, 0] = np.random.randint(0, self.x_max + 1, (self.user_no,))
        state[:, 1] = np.random.uniform(self.p_min, self.p_max, self.user_no)

        return state

    def mask_state_value(self, next_state_values, state, reshape=False):
        if not reshape:
            return next_state_values - ((state == 0) * 1e20).flatten()
        else:
            next_state_values = torch.reshape(next_state_values, [self.x_no, self.p_no])
            next_state_values = next_state_values - (state == 0) * 1e20
            return next_state_values

    def random_action(self, action_idx):

        action = np.ravel_multi_index(action_idx, (self.x_no, self.p_no))

        return action

    # 환경 update
    def update(self):
        self.curr_state = self.next_state
        self.next_state = self.gen_state()

    def scenarios(self, scenario):
        if scenario == 0:
            N = 2
            p_min = 0
            p_max = 1

        elif scenario == 1:
            N = 6
            p_min = 0
            p_max = 1

        elif scenario == 2:
            N = 10
            p_min = 0
            p_max = 1

        return N, p_min, p_max


class QueueSys:
    def __init__(self, env):
        self.arrive_iter = env.arrive_iter
        self.avg_rew = np.zeros(env.total_iterations)
        self.env = env

    # 시스템 업데이트
    def update(self, scheduling):
        est_reward = self.env.curr_state[:, 0] * self.env.curr_state[:, 1]
        # cost 계산 (W)
        reward = - est_reward[scheduling]

        if self.env.iteration == 0:
            self.avg_rew[self.arrive_iter + 0] = reward
        else:
            self.avg_rew[self.arrive_iter + self.env.iteration] = \
                (self.avg_rew[self.arrive_iter + self.env.iteration - 1] * self.env.iteration + reward) \
                / (self.env.iteration + 1)

        return reward

    # 현재 시스템 state 확인
    def get_state(self, next_state=False):
        # state 정의
        state = np.zeros((self.env.x_no, self.env.p_no))
        corr_user = [[[] for _ in range(self.env.p_no)] for _ in range(self.env.x_no)]

        if next_state:
            target_state = self.env.next_state
        else:
            target_state = self.env.curr_state
        for user_idx in range(self.env.user_no):
            if self.env.x_no == self.env.x_max + 1:
                x_idx = int(target_state[user_idx, 0])
            else:
                x_idx = int(np.floor(target_state[user_idx, 0] / (self.env.x_max + 1) * self.env.x_no))
            p_idx = np.max(np.nonzero(self.env.p_partition <= target_state[user_idx, 1]))

            state[x_idx, p_idx] = 1
            corr_user[x_idx][p_idx].append(user_idx)

        return state, corr_user

    # descriptive action 변환
    def interpret_action(self, action, corr_user=None):
        action = list(np.unravel_index(action, (self.env.x_no, self.env.p_no)))
        scheduling = random.sample(corr_user[action[0]][action[1]], 1)

        return scheduling

    # 현재 시스템 action 수행
    def do_action(self, action, corr_user, training=True):
        scheduling = self.interpret_action(action, corr_user)
        reward = self.update(scheduling)

        return reward

    def check_action(self, state, corr_user, action):
        action = list(np.unravel_index(action, (self.env.x_no, self.env.p_no)))

        return corr_user[action[0]][action[1]]

    def getResults(self):
        return dict(avg_rew=self.avg_rew)
