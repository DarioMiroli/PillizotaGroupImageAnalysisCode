import numpy as np
import matplotlib.pyplot as plt

x = [i for i in range(10)]
y = [i*i + np.random.random()*10 for i in x ]

z= np.polyfit(x,y,3)

y2 = [z[0]*X**3 + z[1]*X**2 + z[2]*X + z[3] for X in x]

plt.plot(x,y)
plt.plot(x,y2)
plt.show()
