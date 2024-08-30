import torch
import numpy as np
import random
import utils.function_library as fl


class DataAggEnv:
    def __init__(self, parent, scenario_no, params):
        self.arrive_iter = parent.parent.params.arrive_iter * parent.parent.params.sub_iteration
        self.iteration = 0
        self.parent = parent
        self.total_iterations = parent.total_iterations
        self.params = params

        self.discounted_factor = params.task_specific_params.discount_factor
        self.max_buffer = params.task_specific_params.max_buffer

        self.remain_no = params.task_specific_params.remain_no
        self.cap_no = params.task_specific_params.cap_no

        # scenario setting
        self.dev_no, self.avg_cap, self.arr_rate, self.max_cap, self.init_sample = self.scenarios(scenario_no)

        ####################################################
        # state related part
        # state 정의 : 현재 버퍼 남은 공간 / 전송 용량
        ####################################################
        # remaining buffer partition
        self.remain_partition = np.linspace(0, self.max_buffer + 1, self.remain_no + 1)

        # channel capacity partition
        self.cap_partition = np.linspace(0, self.max_cap + 1, self.cap_no + 1)

        self.feature_no = 2

        self.cap = None
        self.arr = None
        self.next_cap, self.next_arr = self.randomness_generator()

    def randomness_generator(self):
        cap = np.clip(self.avg_cap + np.floor(np.random.randn(self.dev_no)*5), 0, self.max_cap)
        arr = np.random.poisson(self.arr_rate)

        return cap, arr

    # state == 0 선택 안되게끔 masking
    def mask_state_value(self, next_state_values, state, reshape=False):
        if not reshape:
            return next_state_values - ((state == 0) * 1e20).flatten()
        else:
            next_state_values = torch.reshape(next_state_values, [self.remain_no, self.cap_no])
            next_state_values = next_state_values - (state == 0) * 1e20
            return next_state_values

    def random_action(self, action_idx):

        action = np.ravel_multi_index(action_idx, (self.remain_no, self.cap_no))

        return action

    # 환경 update
    def update(self):
        self.cap = self.next_cap
        self.arr = self.next_arr
        self.next_cap, self.next_arr = self.randomness_generator()

    # 시나리오 설정
    def scenarios(self, scenario):
        if scenario == 0:
            dev_no = 4
            avg_cap = [10, 20, 20, 30]
            init_sample = [10]*4
            arr_rate = list(np.ones(dev_no)*5)
            max_cap = 40

        elif scenario == 1:
            dev_no = 9
            avg_cap = [10, 10, 10, 20, 20, 20, 30, 30, 30]
            init_sample = [10]*9
            arr_rate = list(np.ones(dev_no)*2)
            max_cap = 40

        elif scenario == 2:
            dev_no = 20
            avg_cap = np.zeros(dev_no)
            avg_cap[0:5] = 10
            avg_cap[5:15] = 20
            avg_cap[15:20] = 30
            init_sample = [10]*20
            arr_rate = list(np.ones(dev_no))
            max_cap = 40

        elif scenario == 3:
            dev_no = 12
            avg_cap = np.zeros(dev_no)
            avg_cap[0:4] = 10
            avg_cap[4:8] = 20
            avg_cap[8:12] = 30
            init_sample = [10]*12
            arr_rate = list(np.ones(dev_no)*2)
            max_cap = 40

        elif scenario == 4:
            dev_no = 15
            avg_cap = np.zeros(dev_no)
            avg_cap[0:5] = 10
            avg_cap[5:10] = 20
            avg_cap[10:15] = 30
            init_sample = [10]*15
            arr_rate = list(np.ones(dev_no)*3)
            max_cap = 40

        return dev_no, avg_cap, np.array(arr_rate), max_cap, np.array(init_sample)


class DataAggSys:
    def __init__(self, env):
        self.arrive_iter = env.arrive_iter
        self.loss = np.zeros((env.dev_no, env.total_iterations))
        self.agg_data = np.zeros((env.dev_no, env.total_iterations))
        self.avg_tot_agg_data = np.zeros(env.total_iterations)
        self.avg_tot_loss = np.zeros(env.total_iterations)
        self.curr_buffer = env.init_sample
        self.curr_remain_buffer = np.zeros(env.dev_no)
        self.max_buffer = env.max_buffer
        self.avg_reward = np.zeros(env.total_iterations)
        self.reward = np.zeros(env.total_iterations)

        self.scheduled_no = np.zeros(env.dev_no)
        self.env = env

    # 시스템 업데이트
    def update(self, scheduling, training=True):
        # loss 고려안한 버퍼 계산
        tmp_buffer = self.curr_buffer - scheduling * self.env.cap + self.env.arr
        # loss 계산
        self.loss[:, self.arrive_iter + self.env.iteration] = np.maximum(tmp_buffer - self.max_buffer, 0)
        # 현재 버퍼 계산
        self.curr_buffer = np.clip(tmp_buffer, 0, self.max_buffer)
        self.curr_remain_buffer = np.ones(self.env.dev_no) * self.max_buffer - self.curr_buffer
        self.agg_data[:, self.arrive_iter + self.env.iteration] = scheduling * self.env.cap
        self.avg_tot_agg_data[self.arrive_iter + self.env.iteration] = np.sum(self.agg_data[:, 0:self.env.iteration]) / (self.env.iteration + 1)
        self.avg_tot_loss[self.arrive_iter + self.env.iteration] = np.sum(self.loss[:, 0:self.env.iteration]) / (self.env.iteration + 1)

        reward = np.sum(self.agg_data[:, self.env.iteration] - self.loss[:, self.env.iteration])
        self.reward[self.arrive_iter + self.env.iteration] = reward
        if self.env.iteration == 0:
            self.avg_reward[self.arrive_iter + self.env.iteration] = reward
        else:
            self.avg_reward[self.arrive_iter + self.env.iteration] = (self.avg_reward[self.arrive_iter + self.env.iteration-1]*self.env.iteration + reward)/(self.env.iteration + 1)

        return reward

    # 현재 시스템 state 확인
    def get_state(self, next_state=False):
        # state 정의
        state = np.zeros((self.env.remain_no, self.env.cap_no))
        # 각 state 해당하는 유저 저장할 리스트 생성
        corr_user = [[[] for _ in range(self.env.cap_no)] for _ in range(self.env.remain_no)]

        # 각 유저별로 state 확인
        for dev_idx in range(self.env.dev_no):
            if next_state is True:
                remain_idx, cap_idx = self.get_state_idx(self.curr_remain_buffer[dev_idx],
                                                         self.env.next_cap[dev_idx])
            else:
                remain_idx, cap_idx = self.get_state_idx(self.curr_remain_buffer[dev_idx],
                                                         self.env.cap[dev_idx])

            # state point에 현재 유저 더해줌
            state[remain_idx, cap_idx] = 1
            # state 해당하는 유저에 현재 유저 인덱스 추가
            corr_user[remain_idx][cap_idx].append(dev_idx)

        return state, corr_user

    # 주어진 channel, mu에 따른 state idx 확인
    def get_state_idx(self, remain_buffer, cap):
        remain_idx = np.max(np.nonzero(self.env.remain_partition <= remain_buffer))
        cap_idx = np.max(np.nonzero(self.env.cap_partition <= cap))

        return remain_idx, cap_idx

    # descriptive action 변환
    def interpret_action(self, action, corr_user=None):
        action = list(np.unravel_index(action, (self.env.remain_no, self.env.cap_no)))
        # action에 해당하는 유저들 중 하나 선택하여 user idx 반환
        scheduling = np.zeros(self.env.dev_no)
        scheduling[random.sample(corr_user[action[0]][action[1]], 1)] = 1

        return scheduling

    # 현재 시스템 action 수행
    def do_action(self, action, corr_user, training=True):
        # descriptive action 변환
        scheduling = self.interpret_action(action, corr_user)
        reward = self.update(scheduling, training=training)

        return reward  # reward

    def getResults(self):
        return dict(avg_reward=self.avg_reward, reward=self.reward)
        # return dict(avg_tot_loss=self.avg_tot_loss, avg_tot_agg_data=self.avg_tot_agg_data, avg_reward=self.avg_reward)