# agent.py
import numpy as np
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# descriptive policy agent
class DQNAgent:
    def __init__(self, idx, parent_edge, env, system, main_net, target_net):
        self.agent_name = idx
        self.parent_edge = parent_edge
        self.env = env
        self.system = system

        self.lr = 1e-5
        self.epsilon = 0.1
        self.batch_size = 16
        self.replay_limit = 500
        self.train_interval = 50
        self.target_update_interval = 100

        # main network 생성
        self.main_network = main_net
        # target network 생성
        self.target_network = target_net

        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.lr)

        # experience buffer 생성
        self.experience_buffer = ReplayMemory(self.replay_limit)

    def optimize_model(self):
        if len(self.experience_buffer) < self.batch_size:
            return
        transitions = self.experience_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        state_batch_stack = torch.stack(batch.state)

        # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
        # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
        state_action_values = self.main_network(state_batch).gather(1,action_batch)

        # 모든 다음 상태를 위한 V(s_{t+1}) 계산
        next_state_values = self.target_network(next_state_batch).detach()
        for i in range(self.batch_size):
            # state == 0을 action으로 선택하지 않기 위한 mask 수행
            next_state_values[i,:] = self.env.mask_state_value(next_state_values[i,:], state_batch_stack[i, :,:])
        next_state_values = next_state_values.max(1)[0]
        # 기대 Q 값 계산
        expected_state_action_values = (next_state_values.reshape((self.batch_size,1)) * self.env.discounted_factor) + reward_batch

        # Huber 손실 계산
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.main_network.parameters():
            param.grad.data.clamp_(-10, 10)
            # print(sum(param.grad.data))
        self.optimizer.step()

    def work(self, round, training=True):
        # 시스템의 descriptive state 확인
        curr_state, curr_corr_user = self.system.get_state()

        # DQN scheduling 수행
        action = self.scheduling(curr_state, round)

        # self.system.check_action(curr_state, curr_corr_user, action)

        # reward 확인 및 시스템 업데이트
        reward = self.system.do_action(action, curr_corr_user)
        next_state, _ = self.system.get_state(next_state=True)

        if training:
            # experience tuple 저장
            self.experience_buffer.push(torch.from_numpy(curr_state).float(),
                                        torch.tensor(action).reshape((-1,1)),
                                        torch.tensor(reward.reshape((-1,1))).float(),
                                        torch.from_numpy(next_state).float())

            # train interval 마다 training 수행
            if (self.env.iteration + 1) % self.train_interval == 0:
                self.optimize_model()

            # target update interval 마다 target network 업데이트
            if self.env.iteration % self.target_update_interval == 0:
                self.target_network.load_state_dict(self.main_network.state_dict())

    # Q network / e-greedy 기반 스케줄링
    def scheduling(self, state, round):
        if round <= self.parent_edge.parent_params.warm_up_period:
            rnd = 0
        else:
            rnd = np.random.rand(1)
        # e-greedy 스케줄링
        if rnd < self.epsilon:
            # 현재 해당 state에 유저 있는 action만 추출
            action_space = np.transpose(np.nonzero(state > 0))
            # 해당 action space에서 pwr control 제외한 action 선택
            action_idx = random.sample(action_space.tolist(), 1)[0]
            action = self.env.random_action(action_idx)
        # DQN 기반 스케줄링
        else:
            state = torch.tensor(state).float()
            # 현재 Q-network에서 Q-value 확인
            with torch.no_grad():
                Q_value = self.main_network(state)
                # 해당 Q-value에서 최적 action 선택
                action = self.get_action(Q_value, state)

        return action

    # 주어진 Q value 기반 최적 action 확인
    def get_action(self, Q_value, state):
        # state == 0을 action으로 선택하지 않기 위한 mask 수행
        Q_value = self.env.mask_state_value(Q_value, state, reshape=True)
        # 최적 action 추출
        action = np.argmax(Q_value)

        return action


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """transition 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)