from collections import namedtuple

# parameters for cloud
cloud_base = namedtuple('CloudParameters',
                        ('iteration', 'task_no', 'task_params', 'edge_no', 'edge_params',
                         'bw_max', 'mem_max', 'comput_max',
                         'selection_mode', 'sel_params',
                         'warm_up_period'),
                        defaults=(100, 3, [], 10, [],
                                  20, 20, 20,
                                  'RR', None,
                                  0))

# parameters for edge
edge_base = namedtuple('EdgeParameters',
                       ('task',  # edge's task
                        'scenario',  # edge's scenario
                        'avail_prob',
                        'arrive_iter',
                        'sub_iteration'),
                       defaults=(0, 0, 0.5, 0, 100))

# parameters for task
task_base = namedtuple('TaskParameters',
                       ('fc_variable_no',
                        'x_dim',
                        'y_dim',
                        'task_specific_params'),
                       defaults=(300, None, None, None))

QueueTask_base = namedtuple('QueueTaskParameters',
                        ('discount_factor', 'p_no', 'x_no',
                         'unit_bw', 'unit_mem', 'unit_comput'),
                        defaults=(0.9, 4, 5,
                                  1, 1, 1),)

DataAggTask_base = namedtuple('DataAggTaskParameters',
                              ('discount_factor', 'max_buffer', 'remain_no', 'cap_no',
                               'unit_bw', 'unit_mem', 'unit_comput'),
                              defaults=(0.9, 200, 20, 20,
                                        1, 1, 1),)

WPTTask_base = namedtuple('WPTTaskParameters',
                          ('discount_factor', 'max_battery', 'battery_no', 'power_no',
                           'unit_bw', 'unit_mem', 'unit_comput'),
                              defaults=(0.9, 50, 10, 5,
                                        1, 1, 1),)

RRA_base = namedtuple('RRATaskParameters',
                        ('discount_factor',
                         'pathloss_coeff',
                         'log_normal',
                         'bandwidth',
                         'noise_density',
                         'max_trans_pwr',
                         'initial_mu',
                         'max_mu',
                         'min_channel',
                         'max_channel',
                         'h_no',
                         'mu_no',
                         'p_no',
                         'unit_bw', 'unit_mem', 'unit_comput'),
                        defaults=(0.9, 3.76, 10, 5, -106, 10e3, 1., 2, -50, -30, 15, 12, 20,
                                  1, 1, 1))

# parameters for edge selector (PF)
selection_base = namedtuple('SelectorParameters',
                            ('weights',
                             'multiplier_init',
                             'step_size',
                             'minimum_participants'),
                            defaults=([], 1, 0.01, []))
