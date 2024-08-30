import copy
from tqdm import tqdm
import datetime
import parameters
from SimSys.cloud import Cloud
from utils.output import OutputModule

save_time = datetime.datetime.today().strftime("%y%m%d_%H%M%S")
print(f'Save file name: {save_time}')

mat_path = './results/diff_edges/'
mat_name = f'run_{save_time}.mat'

# selection mode for federated learning
selection_modes = ['Bench']
sel_params = [[],
              [],
              parameters.selection_base(multiplier_init=1, minimum_participants=[3, 3, 3]),
              parameters.selection_base(multiplier_init=1, minimum_participants=[3, 3, 3]),
              []]

# settings for tasks
task0_params = parameters.DataAggTask_base()
task1_params = parameters.WPTTask_base()
task2_params = parameters.RRA_base()

task_params = [
    parameters.task_base(x_dim=task0_params.remain_no * task0_params.cap_no,
                         y_dim=task0_params.remain_no * task0_params.cap_no,
                         task_specific_params=task0_params),
    parameters.task_base(x_dim=task1_params.battery_no * 2,
                         y_dim=task1_params.battery_no * 2 * task1_params.power_no,
                         task_specific_params=task1_params),
    parameters.task_base(x_dim=task2_params.h_no * task2_params.mu_no,
                         y_dim=task2_params.h_no * task2_params.mu_no * task2_params.p_no,
                         task_specific_params=task2_params)
]
task_no = len(task_params)

# settings for edges in cloud
edge_tasks = [[0] * 3 + [1] * 3 + [2] * 3,
              [0] * 6 + [1] * 6 + [2] * 6,
              [0] * 9 + [1] * 9 + [2] * 9,
              [0] * 12 + [1] * 12 + [2] * 12,
              [0] * 15 + [1] * 15 + [2] * 15]  # task of each edge
edge_scenarios = [[0, 1, 2] * 3, [0, 1, 2] * 6, [0, 1, 2] * 9, [0, 1, 2] * 12,
                  [0, 1, 2] * 15]  # task scenario of each edge
edge_avail_prob = [[1.] * 9, [1.] * 18, [1.] * 27, [1.] * 36, [1.] * 45]
edge_arrive_iter = [[0] * 9, [0] * 18, [0] * 27, [0] * 36, [0] * 45]
edge_params = [[] for _ in range(6)]

# edge_tasks = [0] * 6  # task of each edge
# edge_scenarios = [0,0,1,1,2,2]  # task scenario of each edge
# edge_arrives = [1.] * 6
# edge_params = []

for scn in range(5):
    for i in range(len(edge_tasks[scn])):
        edge_params[scn].append(parameters.edge_base(task=edge_tasks[scn][i],
                                                     scenario=edge_scenarios[scn][i],
                                                     avail_prob=edge_avail_prob[scn][i],
                                                     arrive_iter=edge_arrive_iter[scn][i]))

# Initialization
Clouds = []
Params = []
for scn in range(5):
    Params.append(parameters.cloud_base(
        iteration=500,
        task_no=task_no,
        task_params=task_params,
        edge_no=len(edge_tasks[scn]),
        edge_params=edge_params[scn],
        selection_mode=selection_modes[0],
        sel_params=sel_params[0]
    ))
    Clouds.append(Cloud(Params[scn], output_name=f'edge_no_{scn}'))  # each cloud has same edges but different scheduling methods

# create output module
output = OutputModule(Params, Clouds)

# scheduling
for cloud in Clouds:
    cloud.Run()
output.getResultsTS()

# federated learning
for iter in tqdm(range(Params[0].iteration), desc=f'FL'):
    for cloud in Clouds:
        cloud.do_FL(iter)
    output.getResults()
    output.saveMAT(mat_path, mat_name)
