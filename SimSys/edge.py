import copy
import numpy as np
from SimSys.edge_env import EdgeEnv
from DQN.model import DQN
from DQN.agent import DQNAgent


# task에 관계없이 공통 edge class 사용
class Edge:
    def __init__(self, idx, parent, params):
        self.idx = idx
        self.parent = parent
        self.parent_params = parent.params
        self.params = params
        self.task = params.task
        # task 별 다른 environment 사용
        self.edge_env = EdgeEnv(self, params.task, params.scenario, self.parent_params.task_params[self.task])

        # task parameter 따라 DQN 생성
        self.main_DQN = DQN(f'edge{idx}_main', self.parent_params.task_params[self.task])
        self.target_DQN = DQN(f'edge{idx}_target', self.parent_params.task_params[self.task])
        self.prev_DQN = DQN(f'edge{idx}_prev', self.parent_params.task_params[self.task])

        # Descriptive Policy Agent 생성
        self.agent = DQNAgent(idx, self, self.edge_env.env, self.edge_env.sys, self.main_DQN, self.target_DQN)

    # DQN 초기화
    def init_DQN(self, DQN):
        self.main_DQN.load_state_dict(DQN)
        self.target_DQN.load_state_dict(DQN)
        self.prev_DQN.load_state_dict(DQN)

    # edge FL 참가 여부
    def get_state(self, iter):
        if self.params.arrive_iter <= iter:
            if np.random.rand(1) < self.params.avail_prob:
                return 1
            else:
                return 0
        else:
            return 0

    # DQN 수행
    def do_DQN(self, round):
        for iteration in range(self.params.sub_iteration):
            # 환경 update (randomness)
            self.edge_env.env.update()

            self.agent.work(round)
            self.edge_env.env.iteration = self.edge_env.env.iteration + 1

    # DQN 주어진 model weight로 치환
    def update_model(self, model):
        self.main_DQN.load_state_dict(model)
        self.target_DQN.load_state_dict(model)

    # output 생성
    def getResults(self):
        return self.edge_env.sys.getResults()