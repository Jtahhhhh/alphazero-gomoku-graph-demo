from collections import deque
import random
class ReplayBuffer:
    def __init__(self,capacity): self.data=deque(maxlen=capacity)
    def extend(self,samples): self.data.extend(samples)
    def sample(self,n): return random.sample(self.data,min(n,len(self.data)))
    def __len__(self): return len(self.data)
