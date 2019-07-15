import numpy as np
import json
from os import path
from ..utils.utils import ensure_dir
import matplotlib.pyplot as plt
plt.switch_backend('agg') 

class TrainingMonitor():
    def __init__(self, fig_dir, arch,json_dir=None, start_at=0):
        self.start_at = start_at
        self.H = {}
        self.loss_path = path.sep.join([fig_dir,arch+'_loss.png'])
        self.json_path = path.sep.join([json_dir,arch+"_training_monitor.json"])
        self.use = 'on_epoch_end'

        ensure_dir(fig_dir)
        ensure_dir(json_dir)

    def _restart(self):
        if self.start_at > 0:
            if self.json_path is not None:
                if path.exists(self.json_path):
                    self.H = json.loads(open(self.json_path).read())
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_at]

    def step(self,logs={}):
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            if not isinstance(v,np.float):
                v = round(float(v),4)
            l.append(v)
            self.H[k] = l

        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()

