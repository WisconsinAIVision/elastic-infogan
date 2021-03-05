import torch
import random
import sys
import argparse
import numpy as np
import os

base_path = os.getcwd()
data_path = '.'
parser = argparse.ArgumentParser(description='Determining the imbalance')
parser.add_argument('--ind',dest='ind', type=int, help='index for the random partitioning')
args = parser.parse_args()

print ("################### Experiment - ", args.ind, "####################")



sl = []
a,b = torch.load('./splits/70k.pt')
full_data = np.load('./splits/50_data_imbalance.npy')
exp_ind = args.ind - 1
for i in range(10):
    sl.append(int(full_data[exp_ind][i]))
print ("The data imbalance - ", sl)

l0 = []
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
l6 = []
l7 = []
l8 = []
l9 = []

for i in range(b.shape[0]):
    if b[i] == 0:
        l0.append(i)
    elif b[i] == 1:
        l1.append(i)
    elif b[i] == 2:
        l2.append(i)
    elif b[i] == 3:
        l3.append(i)
    elif b[i] == 4:
        l4.append(i)
    elif b[i] == 5:
        l5.append(i)
    elif b[i] == 6:
        l6.append(i)
    elif b[i] == 7:
        l7.append(i)
    elif b[i] == 8:
        l8.append(i)
    elif b[i] == 9:
        l9.append(i)

assert (len(l0) + len(l1) + len(l2) + len(l3) + len(l4) + len(l5) + len(l6) + len(l7) + len(l8) + len(l9)) == 70000
#min_points = min(len(l0), len(l1), len(l2), len(l3), len(l4), len(l5), len(l6), len(l7), len(l8), len(l9))
n0 = random.sample(l0, sl[0])
n1 = random.sample(l1, sl[1])
n2 = random.sample(l2, sl[2])
n3 = random.sample(l3, sl[3])
n4 = random.sample(l4, sl[4])
n5 = random.sample(l5, sl[5])
n6 = random.sample(l6, sl[6])
n7 = random.sample(l7, sl[7])
n8 = random.sample(l8, sl[8])
n9 = random.sample(l9, sl[9])

assert (len(n0) + len(n1) + len(n2) + len(n3) + len(n4) + len(n5) + len(n6) + len(n7) + len(n8) + len(n9)) == sum(sl)

fl = n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9
random.shuffle(fl)

c = torch.zeros([len(fl), 28, 28])
d = torch.zeros([len(fl),])

for i in range(len(fl)):
    c[i] = a[fl[i]]
    d[i] = b[fl[i]]

print ("Length of dataset", len(fl))
c = c.type(torch.ByteTensor)
d = d.type(torch.LongTensor)
q = c,d
torch.save(q, "./splits/training_split%d.pt" %(exp_ind))
