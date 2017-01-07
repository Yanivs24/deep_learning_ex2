import matplotlib.pyplot as plt
import sys
import numpy as np

vec = np.loadtxt(sys.argv[1])
plt.plot(range(1,len(vec)+1),vec)
plt.axis([1,len(vec)+1,0,1])
plt.xlabel('Iterations amount')
plt.ylabel('%s' % sys.argv[2])
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title("Dev %s - POS" % sys.argv[2])

plt.show()
