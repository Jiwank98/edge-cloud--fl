import copy
from tqdm import tqdm
import datetime
import parameters
from SimSys.cloud import Cloud
from utils.output import OutputModule

save_time = datetime.datetime.today().strftime("%y%m%d_%H%M%S")
print(f'Save file name: {save_time}')

mat_path = './results/task_only/'
mat_name = f'run_{save_time}.mat'

# selection mode for federated learning
selection_modes = ['PF', 'Greedy', 'RR']
sel_params = [parameters.selection_base(multiplier_init=1, minimum_participants=[3, 3, 3]), [], []]


# settings for tasks
task0_params = parameters.DataAggTask_base()
task1_params = parameters.WPTTask_base()
task2_params = parameters.RRA_base(unit_bw=1, unit_mem=1, unit_comput=1)

task_params = [
    parameters.task_base(x_dim=task0_params.remain_no*task0_params.cap_no,
                         y_dim=task0_params.remain_no*task0_params.cap_no,
                         task_specific_params=task0_params),
    parameters.task_base(x_dim=task1_params.battery_no*2,
                         y_dim=task1_params.battery_no*2,
                         task_specific_params=task1_params),
    parameters.task_base(x_dim=task2_params.h_no*task2_params.mu_no,
                         y_dim=task2_params.h_no*task2_params.mu_no*task2_params.p_no,
                         task_specific_params=task2_params)
]
task_no = len(task_params)

# settings for edges in cloud
edge_tasks = [0] * 10 + [1] * 10 + [2] * 10  # task of each edge
edge_scenarios = [0,0,0,0,1,1,1,2,2,2] * 3  # task scenario of each edge
edge_arrives = [0.5] * 10 + [0.7] * 10 + [0.9] * 10
edge_arrive_iter = [0] * 30
edge_params = []
for i in range(len(edge_tasks)):
    edge_params.append(parameters.edge_base(task=edge_tasks[i],
                                            scenario=edge_scenarios[i],
                                            avail_prob=edge_arrives[i],
                                            arrive_iter=edge_arrive_iter[i]))

# Initialization
Clouds = []
Params = []
for i in range(len(selection_modes)):
    Params.append(parameters.cloud_base(
        iteration=5000,
        task_no=task_no,
        task_params=task_params,
        edge_no=len(edge_tasks),
        edge_params=edge_params,
        selection_mode=selection_modes[i],
        sel_params=sel_params[i]
    ))
    Clouds.append(Cloud(Params[i]))  # each cloud has same edges but different scheduling methods

# create output module
output = OutputModule(Params, Clouds)

# scheduling
for cloud in Clouds:
    cloud.Run()
output.getResults(edge_results=False)
output.saveMAT(mat_path, mat_name)
