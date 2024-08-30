import copy
import scipy.io as sio
from utils.function_library import *


class OutputModule:
    def __init__(self, Params, Clouds):
        self.Params = Params
        self.Clouds = Clouds
        self.selection_modes = []

        self.output = {}
        self.output['params'] = Params

        for Cloud in Clouds:
            self.output[Cloud.output_name] = dict(task_scheduling=None,
                                                  edge_results=[[] for _ in range(Cloud.params.edge_no)])
            self.selection_modes.append(Cloud.params.selection_mode)

    def getResults(self, edge_results=True):
        for Cloud in self.Clouds:
            if Cloud.selector.opportunistic_flag:
                self.output[Cloud.output_name]['multipliers_lambda'] = Cloud.selector.multipliers_lambda
                self.output[Cloud.output_name]['multipliers_mu'] = Cloud.selector.multipliers_mu
            if edge_results:
                for idx in range(self.Params[0].edge_no):
                    self.output[Cloud.output_name]['edge_results'][idx] = Cloud.Edges[idx].getResults()

    def getResultsTS(self, plot=False):
        for Cloud in self.Clouds:
            self.output[Cloud.output_name]['task_scheduling'] = Cloud.schedule
            self.output[Cloud.output_name]['avg_participants'] = Cloud.avg_participants
        if plot:
            fig = plt.figure()
            ax = []
            for i in range(self.Params[0].task_no):
                ax.append(fig.add_subplot(self.Params[0].task_no, 1, i+1))
                for Cloud in self.Clouds:
                    ax[i].plot(Cloud.avg_participants[i])
                    ax[i].legend(self.selection_modes)
            fig.tight_layout(pad=1.0)
            plt.show()

    def saveMAT(self, mat_path, mat_name):
        createFolder(mat_path)
        sio.savemat(mat_path + mat_name, self.output)
        # print('Output 저장 완료')