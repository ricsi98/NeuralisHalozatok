from models import *

q = BootstrappedDQN([10,15,8,7,6], 2, 3)
print('L',q.layers)
print('H',q.heads)