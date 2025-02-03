from fastai.callback.core import Callback
import math

class CustomCosineAnnealing(Callback):
    def __init__(self, n_epochs, lr_max, lr_min=0):
        self.n_epochs = n_epochs
        self.lr_max = lr_max
        self.lr_min = lr_min
    
    def before_fit(self):
        self.n = self.n_epochs * len(self.learn.dls.train)
    
    def before_batch(self):
        # Calculate the current iteration
        pos = self.epoch * len(self.learn.dls.train) + self.iter
        # Calculate the new learning rate using cosine annealing
        self.learn.opt.lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (1 + math.cos(math.pi * pos / self.n))

# Define a custom callback to track learning rate
class CustomTrackLearningRate(Callback):
    def before_epoch(self):
        # Access the learning rate from the optimizer and print it
        lr = self.opt.hypers[0]['lr']
        print(f"Learning rate for epoch {self.epoch}: {lr}")


# callbacks for scheduling lr's (Batch and epoch wise)
class Callback(): order = 0

#|export
class BaseSchedCB(Callback):
    def __init__(self, sched): self.sched = sched
    def before_fit(self, learn): self.schedo = self.sched(learn.opt)
    def _step(self, learn):
        if learn.training: self.schedo.step()

#|export
class BatchSchedCB(BaseSchedCB):
    def after_batch(self, learn): self._step(learn)

#|export
class HasLearnCB(Callback):
    def before_fit(self, learn): self.learn = learn
    def after_fit(self, learn): self.learn = None

#|export
class EpochSchedCB(BaseSchedCB):
    def after_epoch(self, learn): self._step(learn)


# callback for recoding lr
#|export
class RecorderCB(Callback):
    def __init__(self, **d): self.d = d
    def before_fit(self, learn):
        self.recs = {k:[] for k in self.d}
        self.pg = learn.opt.param_groups[0]

    def after_batch(self, learn):
        if not learn.training: return
        for k,v in self.d.items():
            self.recs[k].append(v(self))

    def plot(self):
        for k,v in self.recs.items():
            plt.plot(v, label=k)
            plt.legend()
            plt.show()

