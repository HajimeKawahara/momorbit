import numpy as np
import matplotlib.pyplot as plt
dat=np.load("sample_SB1AST.npz")
samples=dat["arr_0"]
plt.plot(samples[:,0])
plt.show()
