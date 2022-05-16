import numpy as np
import random
from nn import Brain


a = Brain(3,3)
x = [[1,0,0],[0,1,0],[0,0,1]]
y = [[1,0,0],[0,1,0],[0,0,1]]
a.train(x,y)
b = [[1,0,0]]
c = a.predict_one_sample(b)
print(c)
