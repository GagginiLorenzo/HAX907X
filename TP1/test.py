n1=100
n2=100
sigma=0.1
sigma2=2
import numpy as np

nbp = int(np.floor(n1 / 8))
nbp
nbn = int(np.floor(n2 / 8))
nbn
xapp = np.reshape(np.random.rand((nbp + nbn) * 16), [(nbp + nbn) * 8, 2])
xapp
np.random.rand((nbp + nbn) * 16)
[(nbp + nbn) * 8, 2]
yapp = np.ones((nbp + nbn) * 8)
idx = 0
for i in range(-2, 2):
    for j in range(-2, 2):
        if (((i + j) % 2) == 0):
            nb = nbp
        else:
            nb = nbn
            yapp[idx:(idx + nb)] = [(i + j) % 3 + 1] * nb

        xapp[idx:(idx + nb), 0] = np.random.rand(nb)
        xapp[idx:(idx + nb), 0] += i + sigma * np.random.randn(nb)
        xapp[idx:(idx + nb), 1] = np.random.rand(nb)
        xapp[idx:(idx + nb), 1] += j + sigma * np.random.randn(nb)
        idx += nb

ind = np.arange((nbp + nbn) * 8)
np.random.shuffle(ind)
res = np.hstack([xapp, yapp[:, np.newaxis]])
#np.array(res[ind, :2]), np.array(res[ind, 2])

m=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
n=m.reshape(3,4)
n
