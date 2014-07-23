import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,2*np.pi, 300)

y = np.sin(x)
y2 = np.sin(x**2)

plt.plot(x,y, label=r'$\sin(x)$', linewidth=2, color='r')
plt.plot(x,y2, label=r'$\sin(x^2)$')

plt.title('Some functions')
plt.xlabel('x')
plt.ylabel('y')

plt.grid()
plt.legend(loc='lower left')

plt.show()
