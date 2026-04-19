import torch.nn as nn
class Model(nn.Module):
    def __setstate__(self, state): self.__dict__.update(state)
    def state_dict(self): return self.__dict__
class Detect(nn.Module):
    def __setstate__(self, state): self.__dict__.update(state)

