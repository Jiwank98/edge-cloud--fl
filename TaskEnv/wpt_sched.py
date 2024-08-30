import torch
import copy
import numpy as np
import random
import utils.function_library as fl


class WPTEnv:
    def __init__(self, parent, scenario_no, params):
        self.arrive_iter = parent.parent.params.arrive_iter * parent.parent.params.sub_iteration
        self.iteration = 0
        self.parent = parent
        self.total_iterations = parent.total_iterations
        self.params = params

        self.discounted_factor = params.task_specific_params.discount_factor
        self.max_battery = params.task_specific_params.max_battery

        self.battery_no = params.task_specific_params.battery_no
        self.power_no = params.task_specific_params.power_no
        self.active_no = 2

        # scenario setting
        self.dev_no, self.init_battery, self.active_rate, self.ch_rate, self.disch_rate = self.scenarios(scenario_no)

        ####################################################
        # state related part
        # state 정의 : 현재 배터리 레벨 / active 여부
        ####################################################
        # remaining buffer partition
        self.batt_partition = np.linspace(0, self.max_battery+1, self.battery_no + 1)
        self.p_partition = np.linspace(0, 1, self.power_no)

        self.feature_no = 2

        self.active = [True] * self.dev_no
        self.next_active = self.randomness_generator()

    def randomness_generator(self):
        active_ = copy.deepcopy(self.active)
        for dev_idx in range(self.dev_no):
            if active_[dev_idx]:  # True
                if np.random.uniform() > self.active_rate[0]:
                    active_[dev_idx] = False
            else:  # False
                if np.random.uniform() > self.active_rate[1]:
                    active_[dev_idx] = True

        return active_

    # state == 0 선택 안되게끔 masking
    def mask_state_value(self, next_state_values, state, reshape=False):
        if not reshape:
            return next_state_values - np.tile((state == 0) * 1e20, (self.power_no, 1, 1)).flatten()
        else:
            next_state_values = torch.reshape(next_state_values, [-1, self.power_no, self.battery_no, self.active_no])
            next_state_values = next_state_values - np.tile((state == 0) * 1e20, (self.power_no, 1, 1))
            return next_state_values

    def random_action(self, action_idx):
        # power 선택
        power_idx = np.random.randint(self.power_no)

        action = np.ravel_multi_index([power_idx] + action_idx, (self.power_no, self.battery_no, self.active_no))

        return action

    # 환경 update
    def update(self):
        self.active = self.next_active
        self.next_active = self.randomness_generator()

    # 시나리오 설정
    def scenarios(self, scenario):
        if scenario == 0:
            dev_no = 4
            init_battery = [10] * dev_no
            active_rate = [0.9, 0.2]
            ch_rate = [3] * dev_no
            disch_rate = [1] * dev_no

        elif scenario == 1:
            dev_no = 9
            init_battery = [20] * dev_no
            active_rate = [0.9, 0.2]
            ch_rate = [8] * dev_no
            disch_rate = [1] * dev_no

        elif scenario == 2:
            dev_no = 20
            init_battery = [40] * dev_no
            active_rate = [0.9, 0.2]
            ch_rate = [15] * dev_no
            disch_rate = [1] * dev_no

        elif scenario == 3:
            dev_no = 12
            init_battery = [15] * dev_no
            active_rate = [0.9, 0.2]
            ch_rate = [9] * dev_no
            disch_rate = [1] * dev_no

        elif scenario == 4:
            dev_no = 15
            init_battery = [15] * dev_no
            active_rate = [0.9, 0.2]
            ch_rate = [12] * dev_no
            disch_rate = [1] * dev_no

        return dev_no, np.array(init_battery), np.array(active_rate), np.array(ch_rate), np.array(disch_rate)


class WPTSys:
    def __init__(self, env):
        self.arrive_iter = env.arrive_iter
        self.avg_reward = np.zeros(env.total_iterations)
        self.reward = np.zeros(env.total_iterations)
        self.avg_power = np.zeros(env.total_iterations)
        self.batt_lvls = np.zeros((env.dev_no, env.total_iterations))
        self.outage = np.zeros(env.total_iterations)
        self.curr_batt = np.array(env.init_battery)
        self.batt_cautious_lvl = env.max_battery * 0.1

        self.env = env

    # 시스템 업데이트
    def update(self, scheduling, power_rate, training=True):
        # 배터리 레벨 계산
        tmp_batt = self.curr_batt + scheduling * self.env.ch_rate * power_rate - self.env.active * self.env.disch_rate
        # outage 계산
        self.outage[self.arrive_iter + self.env.iteration] = np.sum(tmp_batt <= 0)
        tmp_batt[np.nonzero(tmp_batt <= 0)] = self.env.init_battery[np.nonzero(tmp_batt <= 0)]
        # 현재 현재 배터리 계산
        self.curr_batt = np.clip(tmp_batt, 0, self.env.max_battery)

        reward = - self.outage[self.arrive_iter + self.env.iteration] * 10 - np.sum(self.curr_batt <= self.batt_cautious_lvl) - power_rate * 10
        self.reward[self.arrive_iter + self.env.iteration] = reward
        if self.env.iteration == 0:
            self.avg_power[self.arrive_iter + 0] = power_rate
            self.avg_reward[self.arrive_iter + 0] = reward
        else:
            self.avg_power[self.arrive_iter + self.env.iteration] = (self.avg_power[self.arrive_iter + self.env.iteration-1]*self.env.iteration + power_rate)/(self.env.iteration + 1)
            self.avg_reward[self.arrive_iter + self.env.iteration] = (self.avg_reward[self.arrive_iter + self.env.iteration-1]*self.env.iteration + reward)/(self.env.iteration + 1)

        self.batt_lvls[:, self.arrive_iter + self.env.iteration] = self.curr_batt

        return reward

    # 현재 시스템 state 확인
    def get_state(self, next_state=False):
        # state 정의
        state = np.zeros((self.env.battery_no, self.env.active_no))
        # 각 state 해당하는 유저 저장할 리스트 생성
        corr_user = [[[] for _ in range(self.env.active_no)] for _ in range(self.env.battery_no)]

        # 각 유저별로 state 확인
        for dev_idx in range(self.env.dev_no):
            if next_state is True:
                battery_idx, active_idx = self.get_state_idx(self.curr_batt[dev_idx],
                                                             self.env.next_active[dev_idx])
            else:
                battery_idx, active_idx = self.get_state_idx(self.curr_batt[dev_idx],
                                                             self.env.active[dev_idx])

            # state point에 현재 유저 더해줌
            state[battery_idx, active_idx] = 1
            # state 해당하는 유저에 현재 유저 인덱스 추가
            corr_user[battery_idx][active_idx].append(dev_idx)

        return state, corr_user

    # 주어진 channel, mu에 따른 state idx 확인
    def get_state_idx(self, batt, active):
        battery_idx = np.max(np.nonzero(self.env.batt_partition <= batt))
        if active:
            active_idx = 1
        else:
            active_idx = 0

        return battery_idx, active_idx

    # descriptive action 변환
    def interpret_action(self, action, corr_user=None):
        action = list(np.unravel_index(action, (self.env.power_no, self.env.battery_no, self.env.active_no)))

        power_rate = self.env.p_partition[action[0]]

        # action에 해당하는 유저들 중 하나 선택하여 user idx 반환
        scheduling = np.zeros(self.env.dev_no)
        scheduling[random.sample(corr_user[action[1]][action[2]], 1)] = 1

        return scheduling, power_rate

    # 현재 시스템 action 수행
    def do_action(self, action, corr_user, training=True):
        # descriptive action 변환
        scheduling, power_rate = self.interpret_action(action, corr_user)
        reward = self.update(scheduling, power_rate, training=training)

        return reward  # reward

    def getResults(self):
        return dict(avg_reward=self.avg_reward, reward=self.reward)
        # return dict(avg_reward=self.avg_reward, outage=self.outage, batt_lvls=self.batt_lvls, avg_power=self.avg_power)