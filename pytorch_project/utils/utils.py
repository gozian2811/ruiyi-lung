import torch
import numpy as np

import visdom
import time



class Visualizer(object):
    """
    some usual visualizing utils of training
    """

    def __init__(self, env = 'default', **kwargs):
        self.vis = visdom.Visdom(env = env,**kwargs)
        self.index = {}
        self.log_text = ''

    def reint(self,env = 'default',**kwargs):
        self.vis = visdom.Visdom(env = env,**kwargs)
        return self
    def plot(self,name,y,**kwargs):
        x = self.index.get(name,0)
        self.vis.line(Y=np.array([y]),X=np.array([x]),
            win = name,
            opts=dict(title=name),
            update=None if x==0 else 'append',
            **kwargs
            )
        self.index[name]=x+1
    def log(self,info,win = 'log_text'):
        self.log_text += ('[{time}]{info} <br>'.format(
            time = time.strftime('%m%d_%H%M%S'),info = info))
        self.vis.text(self.log_text,win)
