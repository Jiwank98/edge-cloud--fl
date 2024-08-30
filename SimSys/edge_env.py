from TaskEnv.radio_res_alloc import RRAEnv, RRASys
from TaskEnv.queue_sched import QueueEnv, QueueSys
from TaskEnv.data_aggregation import DataAggEnv, DataAggSys
from TaskEnv.wpt_sched import WPTEnv, WPTSys


class EdgeEnv:
    def __init__(self, parent, task, scenario, task_params):
        self.parent = parent
        self.task_params = task_params
        self.total_iterations = parent.params.sub_iteration * parent.parent_params.iteration
        # task에 해당하는 env 및 sys class 생성
        if type(task_params.task_specific_params).__name__ == 'QueueTaskParameters':
            self.env = QueueEnv(self, scenario, task_params)
            self.sys = QueueSys(self.env)
        elif type(task_params.task_specific_params).__name__ == 'DataAggTaskParameters':
            self.env = DataAggEnv(self, scenario, task_params)
            self.sys = DataAggSys(self.env)
        elif type(task_params.task_specific_params).__name__ == 'WPTTaskParameters':
            self.env = WPTEnv(self, scenario, task_params)
            self.sys = WPTSys(self.env)
        elif type(task_params.task_specific_params).__name__ == 'RRATaskParameters':
            self.env = RRAEnv(self, scenario, task_params)
            self.sys = RRASys(self.env)