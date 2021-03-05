from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import Augmentor
import os
import torch
from PIL import Image

class New_MNIST(datasets.MNIST):

    def __init__(self, root, train = True, transform_1 = None, transform_2 = None, target_transform = None, download = False, exp_ind = None):

        self.root = os.path.expanduser(root)
        self.transform1 = transform_1
        self.transform2 = transform_2
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.data, self.labels = torch.load('./splits/training_split%d.pt' %(exp_ind))


    def __getitem__(self, index):

        if self.train:
            img, target = self.data[index], self.labels[index]
        else:
            img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform1 is not None:
            img1 = self.transform1(img)

        if self.transform2 is not None:
            #img_pert = []
            #for i in range(vis_count):
            #    img_pert.append(self.transform2(img))
            img2 = self.transform2(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target


def dataloader(dataset, input_size, batch_size, split='train', exp_ind = None):

    transform_org = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])  	
    p = Augmentor.Pipeline()
    p.rotate(probability=1, max_left_rotation = 20, max_right_rotation = 20)
    p.zoom(probability=1, min_factor=0.9, max_factor=1.1) #TODO PAY ATTENTION to zoom factor
    p.random_distortion(probability=1, grid_width=1, grid_height=1, magnitude=10)

    transform_aug = transforms.Compose([p.torch_transform(), transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

    if dataset == 'mnist':
        #data_loader = DataLoader(
        #    datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
        #    batch_size=batch_size, shuffle=True)
    
        data_loader = DataLoader(
            New_MNIST('data/mnist/processed', train=True, download=True, transform_1 = transform_org, transform_2 = transform_aug, exp_ind = exp_ind),
            batch_size=batch_size, shuffle=True)
        

    return data_loader
