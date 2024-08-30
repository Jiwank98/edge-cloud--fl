import torch
import numpy as np
import random
import utils.function_library as fl


class RRAEnv:
    def __init__(self, parent, scenario_no, params):
        self.arrive_iter = parent.parent.params.arrive_iter * parent.parent.params.sub_iteration
        self.iteration = 0
        self.parent = parent
        self.total_iterations = parent.total_iterations
        self.params = params

        self.discounted_factor = params.task_specific_params.discount_factor
        self.pathloss_coeff = params.task_specific_params.pathloss_coeff
        self.log_normal = params.task_specific_params.log_normal
        self.bandwidth = params.task_specific_params.bandwidth
        self.noise = fl.db2normal(params.task_specific_params.noise_density) * 1e6
        self.max_trans_pwr = params.task_specific_params.max_trans_pwr
        if self.arrive_iter > 0:
            self.initial_mu = 0.1
        else:
            self.initial_mu = params.task_specific_params.initial_mu
        self.max_mu = params.task_specific_params.max_mu
        self.min_h = params.task_specific_params.min_channel
        self.max_h = params.task_specific_params.max_channel

        self.h_no = params.task_specific_params.h_no
        self.mu_no = params.task_specific_params.mu_no
        self.p_no = params.task_specific_params.p_no

        # scenario setting
        self.user_no, self.user_distance, self.rate_req = self.scenarios(scenario_no)
        self.pathloss = np.asarray(self.user_distance) ** -self.pathloss_coeff
        self.pathloss_dB = fl.normal2db(self.pathloss)

        ####################################################
        # state related part
        ####################################################
        # channel partition
        self.h_partition = np.linspace(self.min_h, self.max_h, self.h_no - 1)

        # Lagrangian multiplier partition
        self.mu_partition = np.linspace(0, np.sqrt(self.max_mu), self.mu_no)
        self.mu_partition = self.mu_partition ** 2

        # power control partition
        self.p_partition = np.linspace(0, self.max_trans_pwr, self.p_no)

        self.feature_no = 2

        self.channel = None
        self.shadowing = None
        self.next_shadowing = self.channel_generator()

    # 채널 생성 (패스로스 + log normal shadowing)
    def channel_generator(self):
        shadowing = np.random.randn(self.user_no) * self.log_normal

        return shadowing

    # state == 0 선택 안되게끔 masking
    def mask_state_value(self, next_state_values, state, reshape=False):
        if not reshape:
            return next_state_values - np.tile((state == 0) * 1e20, (self.p_no, 1, 1)).flatten()
        else:
            next_state_values = torch.reshape(next_state_values, [-1, self.p_no, self.h_no, self.mu_no])
            next_state_values = next_state_values - np.tile((state == 0) * 1e20, (self.p_no, 1, 1))
            return next_state_values

    # random action 선택 함수
    def random_action(self, action_idx):
        # power 선택
        p_idx = np.random.randint(self.p_no)

        action = np.ravel_multi_index([p_idx] + action_idx, (self.p_no, self.h_no, self.mu_no))

        return action

    # 환경 update
    def update(self):
        self.shadowing = self.next_shadowing
        self.channel = self.pathloss * fl.db2normal(self.shadowing)
        self.next_shadowing = self.channel_generator()

    # 시나리오 설정
    def scenarios(self, scenario):
        if scenario == 0:
            user_no = 4
            user_distance = [20, 50, 50, 80]
            rate_req = list(np.ones(user_no)*1)

        elif scenario == 1:
            user_no = 9
            user_distance = [20, 20, 20, 50, 50, 50, 80, 80, 80]
            rate_req = list(np.ones(user_no)*0.5)

        elif scenario == 2:
            user_no = 20
            user_distance = np.zeros(user_no)
            user_distance[0:5] = 20
            user_distance[5:15] = 50
            user_distance[15:20] = 80
            rate_req = list(np.ones(user_no)*0.2)

        elif scenario == 3:
            user_no = 6
            user_distance = [20, 20, 50, 50, 80, 80]
            rate_req = list(np.ones(user_no)*0.7)

        elif scenario == 4:
            user_no = 12
            user_distance = np.zeros(user_no)
            user_distance[0:5] = 20
            user_distance[5:10] = 50
            user_distance[10:15] = 80
            rate_req = list(np.ones(user_no)*0.3)

        return user_no, user_distance, np.array(rate_req)


class RRASys:
    def __init__(self, env):
        self.arrive_iter = env.arrive_iter
        self.avg_rate = np.zeros((env.user_no, env.total_iterations))
        self.inst_rate = np.zeros((env.user_no, env.total_iterations))
        self.avg_pwr = np.zeros(env.total_iterations)
        self.pwr = np.zeros(env.total_iterations)
        self.mu = np.zeros((env.user_no, env.total_iterations))
        self.scheduled_no = np.zeros(env.user_no)
        self.env = env
        self.rate_req = env.rate_req
        self.stepsize = 0.1
        self.curr_mu = np.ones(env.user_no) * env.initial_mu

    # 시스템 업데이트
    def update(self, scheduling, trans_pwr, training=True):
        # inst rate 계산
        tmp_rate = scheduling * self.env.bandwidth \
                   * np.log2(1 + self.env.channel * trans_pwr / self.env.noise / self.env.bandwidth)
        # mu 계산
        if training is True:
            if self.env.iteration == 0:
                tmp_mu = np.maximum((self.stepsize * (self.rate_req - tmp_rate)), np.zeros(self.env.user_no))
            else:
                self.stepsize = max(min(0.1, 1 / (self.env.iteration + 1)), 1e-3)
                tmp_mu = np.maximum((self.curr_mu + self.stepsize * (self.rate_req - tmp_rate)),
                                    np.zeros(self.env.user_no))
                tmp_mu = np.minimum(tmp_mu, np.ones(self.env.user_no) * self.env.max_mu)
        else:
            tmp_mu = self.curr_mu
        # cost 계산 (W)
        cost = (trans_pwr/1e3 - np.sum(tmp_rate * self.curr_mu))/100

        # 스케줄링된 횟수 update
        self.scheduled_no = self.scheduled_no + (scheduling > 0)

        # 전송 전력, rate update
        self.pwr[self.arrive_iter + self.env.iteration] = trans_pwr
        self.inst_rate[:, self.arrive_iter + self.env.iteration] = tmp_rate
        if self.env.iteration == 0:
            self.avg_pwr[self.arrive_iter] = trans_pwr
            self.avg_rate[:,self.arrive_iter] = tmp_rate
        else:
            self.avg_pwr[self.arrive_iter + self.env.iteration] = \
                (self.avg_pwr[self.arrive_iter + self.env.iteration - 1] * self.env.iteration + trans_pwr) \
                / (self.env.iteration + 1)
            self.avg_rate[:, self.arrive_iter + self.env.iteration] = \
                (self.avg_rate[:, self.arrive_iter + self.env.iteration - 1] * self.env.iteration + tmp_rate) \
                / (self.env.iteration + 1)


        # mu update
        self.curr_mu = tmp_mu
        self.mu[:, self.arrive_iter + self.env.iteration] = tmp_mu

        return cost, tmp_mu

    # 현재 시스템 state 확인
    def get_state(self, next_state=False):
        # state 정의
        state = np.zeros((self.env.h_no, self.env.mu_no))
        # 각 state 해당하는 유저 저장할 리스트 생성
        corr_user = [[[] for _ in range(self.env.mu_no)] for _ in range(self.env.h_no)]

        # 각 유저별로 state 확인
        for user_idx in range(self.env.user_no):
            if next_state is True:
                h_idx, mu_idx = self.get_state_idx(self.env.next_shadowing[user_idx] + self.env.pathloss_dB[user_idx],
                                                   self.curr_mu[user_idx])
            else:
                h_idx, mu_idx = self.get_state_idx(self.env.shadowing[user_idx] + self.env.pathloss_dB[user_idx],
                                                   self.curr_mu[user_idx])

            # state point에 현재 유저 더해줌
            state[h_idx, mu_idx] = 1
            # state 해당하는 유저에 현재 유저 인덱스 추가
            corr_user[h_idx][mu_idx].append(user_idx)

        return state, corr_user

    # 주어진 channel, mu에 따른 state idx 확인
    def get_state_idx(self, shadowing, mu):
        # get channel idx
        h_idx = np.nonzero(self.env.h_partition <= shadowing)
        if h_idx[0].size:
            h_idx = np.max(h_idx) + 1
        else:
            h_idx = 0

        # get mu idx
        mu_idx = np.max(np.nonzero(self.env.mu_partition <= mu))

        return h_idx, mu_idx

    # descriptive action 변환
    def interpret_action(self, action, corr_user=None):
        action = list(np.unravel_index(action, (self.env.p_no, self.env.h_no, self.env.mu_no)))
        # action에 해당하는 전송 전력 반환
        trans_pwr = self.env.p_partition[action[0]]
        # action에 해당하는 유저들 중 하나 선택하여 user idx 반환
        scheduling = np.zeros(self.env.user_no)
        scheduling[random.sample(corr_user[action[1]][action[2]], 1)] = 1

        return scheduling, trans_pwr

    # 현재 시스템 action 수행
    def do_action(self, action, corr_user, training=True):
        # descriptive action 변환
        scheduling, trans_pwr = self.interpret_action(action, corr_user)
        cost, _ = self.update(scheduling, trans_pwr, training=training)

        return - cost  # reward (negative cost)

    def check_action(self, state, corr_user, action):
        action = list(np.unravel_index(action, (self.env.p_no, self.env.h_no, self.env.mu_no)))

        return corr_user[action[1]][action[2]]

    def getResults(self):
        return dict(avg_pwr=self.avg_pwr, avg_rate=self.avg_rate[-1,:], pwr=self.pwr)
        # return dict(avg_rate=self.avg_rate, avg_pwr=self.avg_pwr, mu=self.mu, pwr=self.pwr)