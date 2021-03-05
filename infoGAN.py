
import utils, torch, time, os, pickle, itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader import dataloader
import sys
from torch.autograd import Variable
from mnist_train import Net
from nt_xent import NTXentLoss
import imageio


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x) * F.log_softmax(x)
        b = -1 * b.sum(dim = 1)
        return b.mean()

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
        utils.initialize_weights(self)

    def forward(self, z, dist_code):
        x = torch.cat([z, dist_code], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class Front_end(nn.Module):

    def __init__(self, input_dim=1, input_size=32):
        super(Front_end, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        a = self.fc(x)
        return a


class Discriminator(nn.Module):
    # Module which predicts real/fake for the image
    def __init__(self, output_dim=1):
        super(Discriminator, self).__init__()
        self.output_dim = output_dim
        
        self.fc = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid()
        )
        utils.initialize_weights(self)

    def forward(self, input):

        x = self.fc(input)
        return x


class Latent_predictor(nn.Module):
    # Module which reconstructs the latent codes from the fake images
    def __init__(self, len_discrete_code = 10):
        super(Latent_predictor, self).__init__()

        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)

        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            #nn.Linear(128, self.len_discrete_code)
        )
        self.fc1 =nn.Linear(128, self.len_discrete_code)
        utils.initialize_weights(self)

    def forward(self, input):

        a = self.fc(input)
        b = self.fc1(a)
        return a,b


class infoGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.mytemp = args.mytemp
        self.klwt = args.klwt
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.save_ind = args.ind
        self.z_dim = 62
        self.len_discrete_code = 10   # categorical distribution (i.e. label)
        self.exp_id = args.ind - 1
        self.sample_num = 100 
        temp = torch.tensor(self.len_discrete_code * [float(1)/ self.len_discrete_code]).cuda()  
        self.prior_parameters = Variable(temp, requires_grad = True)
        self.repeat_checks = 100
        self.mnist_net = Net().cuda()
        state_dict = torch.load('mnist_cnn.pth')
        self.mnist_net.load_state_dict(state_dict) 
        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size, exp_ind=self.exp_id)
        print ("Length of dataloader", len(self.data_loader))
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        self.G = Generator(input_dim = self.z_dim, output_dim = data.shape[1], input_size = self.input_size, len_discrete_code = self.len_discrete_code)
        self.FE = Front_end(input_dim = data.shape[1], input_size = self.input_size)
        self.D = Discriminator(output_dim=1)
        self.Q = Latent_predictor(len_discrete_code = self.len_discrete_code)

        self.G_optimizer = optim.Adam([{'params':self.G.parameters()}, {'params':self.Q.parameters()}, {'params':self.prior_parameters}], lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam([{'params':self.FE.parameters()}, {'params':self.D.parameters()}], lr=args.lrD, betas=(args.beta1, args.beta2))
        self.nt_xent_criterion = NTXentLoss('cuda', self.batch_size, self.mytemp, True)
        
        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.FE.cuda()
            self.Q.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
            self.MSE_loss = nn.MSELoss().cuda()
            self.entropy_loss = HLoss().cuda()
        else:
            ghj = 1 

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # Fixed noise
        self.sample_z_ = torch.randn((self.sample_num, self.z_dim))
        temp = torch.zeros((self.len_discrete_code, 1))
        for i in range(self.len_discrete_code):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.sample_num):
                temp_y[i] = temp_y[i] + (i / (self.sample_num/self.len_discrete_code)) 
        
        self.sample_y_ = torch.zeros((self.sample_num, self.len_discrete_code)).scatter_(1, temp_y.type(torch.LongTensor), 1)

        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = \
                self.sample_z_.cuda(), self.sample_y_.cuda()
                

    def sample_gumbel(self, shape, eps = 1e-20):

	    u = torch.FloatTensor(shape, self.len_discrete_code).cuda().uniform_(0, 1)
	    return -torch.log(-torch.log(u + eps) + eps)

    def gumbel_softmax_sample(self, logits, temp, batch_size):

	    y = logits + self.sample_gumbel(batch_size)
	    return torch.nn.functional.softmax( y / temp)

    def approx_latent(self, params):

	    params = F.softmax(params)
	    log_params = torch.log(params)
	    c = self.gumbel_softmax_sample(log_params, temp = 0.1, batch_size = self.batch_size) 
	    return c



    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['info_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.prior_denominator = 3 
        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1, device = "cuda"), torch.zeros(self.batch_size, 1, device = "cuda")
        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, xa_, y_) in enumerate(self.data_loader):

                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                z_ = torch.randn((self.batch_size, self.z_dim), device = "cuda")

                y_disc_= self.approx_latent(self.prior_parameters)

                if self.gpu_mode:
                    x_  = x_.cuda()
                    xa_ = xa_.cuda()
                # update D network
                self.D_optimizer.zero_grad()

                # real part
                real_intm = self.FE(x_)
                real_intm_aux = self.FE(xa_)
                real_logits = self.D(real_intm)
                D_real_loss = self.BCE_loss(real_logits, self.y_real_)

                # fake part
                fx = self.G(z_, y_disc_) 
                fake_intm_tmp = self.FE(fx.detach())
                fake_logits_tmp = self.D(fake_intm_tmp)
                D_fake_loss = self.BCE_loss(fake_logits_tmp, self.y_fake_)
                D_loss = D_real_loss + D_fake_loss

                D_loss.backward(retain_graph=True)
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                fake_intm = self.FE(fx)
                fake_logits = self.D(fake_intm)
                G_fake_loss = self.BCE_loss(fake_logits, self.y_real_)

                # information loss
                _,c_pred = self.Q(fake_intm)
                info_loss = self.CE_loss(c_pred, torch.max(y_disc_, 1)[1])

                
                # Augmentation similarity loss 
                real_c_pred,_ = self.Q(real_intm)
                real_aux_c_pred,_ = self.Q(real_intm_aux)
                real_c_pred = F.normalize(real_c_pred,dim=1)
                real_aux_c_pred = F.normalize(real_aux_c_pred,dim=1)
                kl_loss = self.nt_xent_criterion(real_c_pred, real_aux_c_pred)

                 
                G_loss = G_fake_loss + info_loss + self.klwt*kl_loss 
                G_loss.backward(retain_graph=True)
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, Info_loss: %.8f, KL_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_fake_loss.item(), info_loss.item(), kl_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))
            if epoch % 10 == 0:

                if not os.path.exists('./saved_models'):
                    os.makedirs('./saved_models')

                torch.save(self.G.state_dict(), os.path.join('./saved_models', 'netG%d.pth' %(self.exp_id)))
                torch.save(self.D.state_dict(), os.path.join('./saved_models', 'netD%d.pth' %(self.exp_id)))
                torch.save(self.FE.state_dict(), os.path.join('./saved_models', 'netFE%d.pth' %(self.exp_id)))
                torch.save(self.Q.state_dict(), os.path.join('./saved_models', 'netQ%d.pth' %(self.exp_id)))


        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()

        self.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        """ style by class """
        samples = self.G(self.sample_z_, self.sample_y_)
        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/disc_interpolation%d.png' %(self.save_ind))


    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

    def loss_plot(self, hist, path='Train_hist.png', model_name=''):
        x = range(len(hist['D_loss']))

        y1 = hist['D_loss']
        y2 = hist['G_loss']
        y3 = hist['info_loss']

        plt.plot(x, y1, label='D_loss')
        plt.plot(x, y2, label='G_loss')
        plt.plot(x, y3, label='info_loss')

        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        path = os.path.join(path, model_name + '_loss.png')

        plt.savefig(path)
