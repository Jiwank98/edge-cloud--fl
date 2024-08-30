import copy
from tqdm import tqdm
import datetime
import parameters
from SimSys.cloud import Cloud
from utils.output import OutputModule

save_time = datetime.datetime.today().strftime("%y%m%d_%H%M%S")
print(f'Save file name: {save_time}')

mat_path = './results/'
mat_name = f'run_{save_time}.mat'

# selection mode for federated learning
selection_modes = ['Bench', 'Greedy', 'RR', 'PF', 'noFL']
# sel_params = [[],
#               [],
#               parameters.selection_base(multiplier_init=1, minimum_participants=[3, 3, 3]),
#               parameters.selection_base(multiplier_init=1, minimum_participants=[3, 3, 3]),
#               []]
sel_params = [[], [], [], parameters.selection_base(multiplier_init=1, minimum_participants=[4, 4, 4]), []]

# settings for tasks
task0_params = parameters.DataAggTask_base()
task1_params = parameters.WPTTask_base()
task2_params = parameters.RRA_base()

task_params = [
    parameters.task_base(x_dim=task0_params.remain_no*task0_params.cap_no,
                         y_dim=task0_params.remain_no*task0_params.cap_no,
                         task_specific_params=task0_params),
    parameters.task_base(x_dim=task1_params.battery_no*2,
                         y_dim=task1_params.battery_no*2*task1_params.power_no,
                         task_specific_params=task1_params),
    parameters.task_base(x_dim=task2_params.h_no*task2_params.mu_no,
                         y_dim=task2_params.h_no*task2_params.mu_no*task2_params.p_no,
                         task_specific_params=task2_params)
]
task_no = len(task_params)

# settings for edges in cloud
# edge_tasks = [0] * 10 + [1] * 10 + [2] * 10  # task of each edge
# edge_scenarios = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2] * 3  # task scenario of each edge
# edge_arrives = [0.9] * 10 + [0.7] * 10 + [0.5] * 10
# edge_arrive_iter = [0] * 30
# edge_params = []

edge_tasks = [0] * 20 + [1] * 20 + [2] * 20  # task of each edge
edge_scenarios = [0,0,0,0,1,1,1,2,2,2] * 6  # task scenario of each edge
edge_arrives = [0.7] * 20 + [0.5] * 20 + [0.3] * 20
edge_arrive_iter = [0] * 60
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
        iteration=500,
        task_no=task_no,
        task_params=task_params,
        edge_no=len(edge_tasks),
        edge_params=edge_params,
        selection_mode=selection_modes[i],
        sel_params=sel_params[i]
    ))
    if i == 0:
        Clouds.append(Cloud(Params[i]))  # each cloud has same edges but different scheduling methods
    else:
        Clouds.append(Cloud(Params[i], task_net_init=[Clouds[i-1].Task_Nets[task_idx].state_dict() for task_idx in range(task_no)]))

# create output module
output = OutputModule(Params, Clouds)

# scheduling
for cloud in Clouds:
    cloud.Run()
output.getResultsTS(plot=True)

# input("Please press the Enter key to proceed")

# federated learning
for iter in tqdm(range(Params[0].iteration), desc=f'FL'):
    for cloud in Clouds:
        cloud.do_FL(iter)
    output.getResults()
    output.saveMAT(mat_path, mat_name)
