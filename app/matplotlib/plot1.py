import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,1000)

plt.subplot(2,1,1)
plt.plot(x, np.sin(x))
plt.grid()
plt.title('sin(x)')

plt.subplot(2,1,2)
plt.plot(x, np.cos(x))
plt.title('cos(x)')
plt.grid()
plt.xlabel('x')

plt.show()
