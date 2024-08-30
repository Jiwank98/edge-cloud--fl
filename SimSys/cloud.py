import copy
import numpy as np
from tqdm import tqdm
from DQN.model import DQN
from SimSys.task_selectors import TaskSelector
from SimSys.edge import Edge


class Cloud:
    def __init__(self, params, task_net_init=None, output_name=None):
        self.params = params
        if output_name is None:
            self.output_name = params.selection_mode
        else:
            self.output_name = output_name

        self.selector = TaskSelector(params)

        self.edge_no = params.edge_no
        self.task_no = params.task_no
        self.schedule = np.zeros((self.task_no, self.params.iteration))
        self.edge_avail = np.zeros((self.edge_no, self.params.iteration))
        self.avg_participants = np.zeros((self.task_no, self.params.iteration))

        # define tasks and their DQNs
        self.Tasks = [set() for _ in range(self.task_no)]  # sets of edges for tasks
        self.Task_Nets = [DQN(f'task{i}', params.task_params[i]) for i in range(self.task_no)]

        if task_net_init is not None:
            for i in range(self.task_no):
                self.Task_Nets[i].load_state_dict(task_net_init[i])

        self.Edges = []
        for edge_idx in range(self.edge_no):
            self.Edges.append(Edge(edge_idx, self, params.edge_params[edge_idx]))
            self.Edges[edge_idx].init_DQN(self.Task_Nets[params.edge_params[edge_idx].task].state_dict())
            self.Tasks[params.edge_params[edge_idx].task].add(edge_idx)  # add edge index to its corresponding task set

    # federated learning scheduling
    def Run(self):
        for iter in tqdm(range(self.params.iteration), desc=f'TS for {self.params.selection_mode}'):
            state, bw, mem, comput, avail = self.get_state(iter)

            self.edge_avail[:, iter] = avail
            self.schedule[:, iter] = self.select_task(state, bw, mem, comput, iter)
            if iter == 0:
                self.avg_participants[:, iter] = state * self.schedule[:, iter]
            else:
                self.avg_participants[:, iter] = (self.avg_participants[:, iter-1] * iter + state * self.schedule[:, iter]) / (iter + 1)

    def select_task(self, state, bw, mem, comput, iter):
        return self.selector.do_select(state, bw, mem, comput, iter)

    # state for federated learning scheduling problem
    def get_state(self, iter):
        state = np.zeros(self.task_no)
        avail = np.zeros(self.edge_no)
        for edge_idx in range(self.edge_no):
            tmp_avail = self.Edges[edge_idx].get_state(iter)
            state[self.Edges[edge_idx].task] = state[self.Edges[edge_idx].task] + tmp_avail
            avail[edge_idx] = tmp_avail
        bw, mem, comput = np.zeros(self.task_no), np.zeros(self.task_no), np.zeros(self.task_no)
        for task_idx in range(self.task_no):
            bw[task_idx] = state[task_idx] * self.params.task_params[task_idx].task_specific_params.unit_bw
            mem[task_idx] = state[task_idx] * self.params.task_params[task_idx].task_specific_params.unit_mem
            comput[task_idx] = state[task_idx] * self.params.task_params[task_idx].task_specific_params.unit_comput

        return state, bw, mem, comput, avail

    # do federated learning based on the scheduling
    def do_FL(self, iter):
        # do DQN for all edges
        for edge_idx in range(self.params.edge_no):
            if self.Edges[edge_idx].params.arrive_iter <= iter:
                self.Edges[edge_idx].do_DQN(iter)

        # do federated learning only for the scheduled tasks
        for task_idx in range(self.task_no):
            if self.schedule[task_idx, iter] == 1:
                grads = []
                for edge_idx in self.Tasks[task_idx]:
                    if self.Edges[edge_idx].params.arrive_iter < iter:
                        if self.edge_avail[edge_idx, iter] == 1:
                            grads.append(self.get_grad(self.Edges[edge_idx].prev_DQN.state_dict(),
                            # grads.append(self.get_grad(self.Task_Nets[task_idx].state_dict(),
                                                       self.Edges[edge_idx].main_DQN.state_dict()))
                agg_model = self.aggregate_grads(self.Task_Nets[task_idx].state_dict(), grads, iter)
                self.Task_Nets[task_idx].load_state_dict(agg_model)
                for edge_idx in self.Tasks[task_idx]:
                    self.Edges[edge_idx].update_model(agg_model)

            for edge_idx in self.Tasks[task_idx]:
                self.Edges[edge_idx].prev_DQN.load_state_dict(self.Edges[edge_idx].main_DQN.state_dict())

    # calculate gradients
    def get_grad(self, w_ori, w):
        output = copy.deepcopy(w)
        for key in output.keys():
            output[key] = (w[key] - w_ori[key])
        return output

    # aggregate gradients to given model
    def aggregate_grads(self, w, grads, iter):
        w_avg = copy.deepcopy(w)
        for key in w_avg.keys():
            for i in range(len(grads)):
                w_avg[key] += grads[i][key]
        return w_avg