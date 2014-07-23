import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,1000)

fig = plt.figure()

ax1 = fig.add_subplot(2,1,1)
ax1.plot(x, np.sin(x))
ax1.set_title('sin(x)')
ax1.grid(True)

ax2 = fig.add_subplot(2,1,2)
ax2.plot(x, np.cos(x))
ax2.set_title('cos(x)')
ax2.grid(True)

plt.show()
