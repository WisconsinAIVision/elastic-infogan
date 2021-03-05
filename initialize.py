import numpy as np

a = np.zeros([50, 10, 10])
np.save('50_class_dist.npy', a)

a = np.zeros([50,])
np.save('50_no_unique.npy', a)

a = np.zeros([50,])
np.save('50_acc.npy', a)

a = np.zeros([50, 10])
np.save('obtained_priors.npy', a)
