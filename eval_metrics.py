import torch
import torch.nn as nn
from mnist_train import Net
import sys
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
from scipy.stats import entropy
import os
from torchvision import utils as vutils

def compute_entropy(arr):

    rows = arr.shape[0]
    cols = arr.shape[1]
    l = []
    for i in range(rows):
        l.append(entropy(arr[i, :]))

    for i in range(cols):
        l.append(entropy(arr[:, i]))

    return np.mean(np.asarray(l))


class Generator(nn.Module):

    def __init__(self, input_dim=100, output_dim=1, input_size=32, len_discrete_code=10):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.len_discrete_code, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        #utils.initialize_weights(self)

    def forward(self, z, dist_code):
        x = torch.cat([z, dist_code], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

z_dim = 62
len_discrete_code = 10
output_dim = 1 # number of channels of the input images
input_size = 28 # image resolution
n_exp = 50 
sample_num = 100
repeat_checks = 100


temp_y = torch.zeros((sample_num, 1))
for i in range(sample_num):
    temp_y[i] = temp_y[i] + (i / (sample_num/len_discrete_code)) 
sample_y_ = torch.zeros((sample_num, len_discrete_code)).scatter_(1, temp_y.type(torch.LongTensor), 1).cuda()


# defining and loading the weights of the generator
G = Generator(input_dim = z_dim, output_dim = output_dim, input_size = input_size, len_discrete_code = len_discrete_code)
model_base_path = sys.argv[1] 

# defining and loading the weights of the pretrained MNSIT classifier
mnist_net = Net()
state_dict = torch.load('mnist_cnn.pth')
mnist_net.load_state_dict(state_dict)
mnist_net.cuda()

final_matrix = torch.zeros([n_exp, len_discrete_code, len_discrete_code])
for i in range(n_exp):
    temp_res = torch.zeros([len_discrete_code, len_discrete_code])
    print (i)
    netG_filename = os.path.join(model_base_path, 'netG%d.pth' %(i))
    state_dict = torch.load(netG_filename)
    G.load_state_dict(state_dict)
    G.eval()
    G.cuda()
    for u in range(repeat_checks):
        temp_sample_z_ = torch.randn((sample_num, z_dim)).cuda()
        qw = G(temp_sample_z_, sample_y_)
	#temp = (qw + 1)/2
        #vutils.save_image(temp.data, 'temp.png', nrow = 10, normalize = True)
        preds_gen = mnist_net(qw)
        top_ind_c1 = torch.argmax(preds_gen, dim = 1)
        top_ind_c1 = torch.reshape(top_ind_c1, [10, 10]) # shape - [10, 10]
        #print (top_ind_c1)
        #sys.exit(0)
        for v in range(len_discrete_code):
            for w in range(10):
                temp_res[v][top_ind_c1[v][w]] = temp_res[v][top_ind_c1[v][w]] + 1
		
    temp_res = temp_res/10
    final_matrix[i] = temp_res
    
final_matrix = final_matrix.cpu().numpy()
#np.save('final_matrix.npy', final_matrix)
#sys.exit(0)


# computation of entropy 
a = final_matrix
res = [] # List storing the final 'overall entropy' (across rows and columns) for each of the runs
for j in range(n_exp):
    if np.sum(a[j, :, :]) > 0: # Only test if the experiment has run
        res.append(compute_entropy(a[j, :, :]))


print ("# of experiments considered", len(res))
print ("Entropy mean -", np.mean(np.asarray(res)))
print ("Entropy std -", np.std(np.asarray(res)))




# computation of NMI
num_samples=1000

mat = final_matrix
all_nmi=[]
for run_it in range(n_exp):
    curr_run=mat[run_it]*10
    pred_label=np.zeros((num_samples*mat.shape[1]))
    gt_labels=np.zeros((num_samples*mat.shape[1]))
    tmp_offset=0
    for class_it in range(mat.shape[1]):
        pred_label[class_it*num_samples:(class_it+1)*num_samples]=class_it
        my_offset=0
        for tmp_it in range(mat.shape[1]):
            num_pred=int(round(curr_run[class_it][tmp_it]))
            if num_pred>0:
                gt_labels[class_it*num_samples+my_offset:class_it*num_samples+my_offset+num_pred]=tmp_it
                my_offset=my_offset+num_pred

        tmp_offset=my_offset
    curr_nmi=nmi(gt_labels,pred_label)

    if(tmp_offset>0):
        all_nmi.append(curr_nmi)

print("NMI mean - %f" %(np.mean(np.asarray(all_nmi))))
print("NMI std - %f" %(np.std(np.asarray(all_nmi))))

 
