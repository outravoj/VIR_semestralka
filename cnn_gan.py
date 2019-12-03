import argparse
import os
import os.path as osp
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.nn.init
import torch.optim as optim
import torch.distributions
import tracker
from torch.autograd import Variable

PATH = "/local/temporary/vir/hw01/pkl"

def load_data(batch_size, filename, device, shuffle=True, store_dir='./data'):
    print(osp.join(store_dir, filename))
    p = np.load(osp.join(store_dir, filename))
    data = torch.from_numpy(p.astype('float32'))  # squeeze into correct dtype and range [0-1]
    labels = torch.from_numpy(np.ones((data.shape[0],1)).astype("float32"))
    data = data.to(device)
    labels = labels.to(device)
    print(data.shape)
    dataset = torch.utils.data.TensorDataset(data, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv1d(2*68, 72, kernel_size=7)
        self.conv2 = nn.Conv1d(72, 48, kernel_size=5)
        self.conv3 = nn.Conv1d(48, 24, kernel_size=3)
        self.conv4 = nn.Conv1d(24,8, kernel_size=3)
        self.lin1 = nn.Linear((80-14) * 8, 1)
        self.seq = nn.Sequential(self.conv1, nn.LeakyReLU(0.2), nn.InstanceNorm1d(72),
                                 self.conv2, nn.LeakyReLU(0.2), nn.InstanceNorm1d(48),
                                 self.conv3, nn.LeakyReLU(0.2), nn.InstanceNorm1d(24),
                                 self.conv4)
        self.weights_initialization()

    def weights_initialization(self):
        for layer in self.named_parameters():
            if type(layer) in {nn.Conv1d, nn.Linear}:
                nn.init.xavier_uniform_(layer.weight.data, gain=torch.sqrt(2))

    def forward(self, x):
        x = self.seq(x)
        x = x.view(-1,66*8)
        x = torch.sigmoid(self.lin1(x.view(-1,(80-14) * 8)))
        return x


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lin = nn.Linear(100,7200)
        self.conv1t = nn.Conv1d(20, 40, kernel_size=4,stride=2)
        self.conv2t = nn.Conv1d(40, 60, kernel_size=4, stride=2)
        self.conv3t = nn.Conv1d(60,80, kernel_size=4)
        self.conv4t = nn.Conv1d(80, 100, kernel_size=4)
        self.conv5t = nn.Conv1d(100, 136, kernel_size=3)
        self.seq = nn.Sequential(
                        self.conv1t, nn.LeakyReLU(negative_slope=0.2), nn.InstanceNorm1d(40),
                        self.conv2t, nn.LeakyReLU(negative_slope=0.2), nn.InstanceNorm1d(60),
                        self.conv3t, nn.LeakyReLU(negative_slope=0.2), nn.InstanceNorm1d(80),
                        self.conv4t, nn.LeakyReLU(negative_slope=0.2), nn.InstanceNorm1d(100),
                        self.conv5t)
        self.weights_initialization()
        self.lr = nn.LeakyReLU(negative_slope=0.2)

    def weights_initialization(self):
        for layer in self.named_parameters():
            if type(layer) in {nn.Conv1d, nn.Linear}:
                nn.init.xavier_uniform_(layer.weight.data, gain=torch.sqrt(2))

    def forward(self, x):
        batch = x.shape[0]
        x = F.leaky_relu(self.lin(x),negative_slope=0.2)
        x = x.view(batch,20,360)
        return self.seq(x)


def sample_from_distribution(number_of_samples, dims):
    samples = torch.from_numpy(np.zeros((number_of_samples, 1, dims)))
    for i in range(number_of_samples):
        sample = torch.randn(samples[0].shape)
        sample = sample / torch.norm(sample)
        samples[i, 0,:] = sample
    return samples


def learn(discriminator, generator, opt_d, opt_g, bs, loader, epochs, device):
    discriminator.train()
    generator.train()
    fake_label = 0
    real_label = 1
    for i in range(epochs):
        print("epoch {}".format(i))
        j = 0
        last_d_loss = 6
        last_g_loss = 6
        for batch, _ in loader:
            #print(batch.shape, real_labels.shape)
            "trenink diskriminatoru"

            num = batch.shape[0]
            discriminator.zero_grad()

            real_pred = discriminator(batch)
            real_labels = torch.full(real_pred.shape, real_label) + (torch.rand(real_pred.shape) - 0.5) * 4
            real_labels.to(device)

            errD_real = F.binary_cross_entropy(real_pred, real_labels)
            errD_real.backward()
            D_x = real_pred.mean().item()

            noise = Variable(sample_from_distribution(num, 100).float()).to(device)
            generated = generator(noise)
            fake_pred = discriminator(generated)
            fake_labels = torch.full(fake_pred.shape, fake_label + 0.15) + (torch.rand(fake_pred.shape) - 0.5) * 3
            fake_labels.to(device)
            errD_fake = F.binary_cross_entropy(fake_pred, fake_labels)
            errD_fake.backward()
            D_G_z1 = fake_pred.mean().item()
            errD = errD_fake + errD_real
            opt_d.step()

            opt_g.zero_grad()
            noise = Variable(sample_from_distribution(num, 100).float()).to(device)
            generated = generator(noise)
            pred = discriminator(generated)
            labels = torch.full(pred.shape, real_label)
            labels.to(device)
            errG = F.binary_cross_entropy(pred,labels)
            D_G_z2 = pred.mean().item()
            opt_g.step()

            if (i+1)%20==0:
                f = str(i) + ".pt"
                torch.save(discriminator.state_dict(), "model_d" + f)
                torch.save(generator.state_dict(), "model_g" + f)
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (i, epochs,
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


def parse_args():
    parser = argparse.ArgumentParser('Simple MNIST classifier')
    parser.add_argument('--epochs', '-e', default=100, type=int)
    parser.add_argument('--batch_size', '-bs', default=128, type=int)
    parser.add_argument('--store_dir', '-sd', default='./', type=str)
    return parser.parse_args()


def main(path=""):
    args = parse_args()
    device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        torch.cuda.empty_cache()
    if path == "":
        path = PATH
    loader = load_data(args.batch_size, "input_data.npy", device=device, store_dir=path, shuffle=True)
    discriminator = Discriminator()
    generator = Generator()
    if device:
        generator.to(device)
        discriminator.to(device)


    lr = 0.0002

    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    learn(discriminator, generator, opt_d, opt_g, args.batch_size, loader, args.epochs, device)
    torch.save(discriminator.state_dict(), "model_d.pt")
    torch.save(generator.state_dict(), "model_g.pt")


def load_model(name, model_type):
    dir_path = osp.dirname(os.path.realpath(__file__))
    path = dir_path + "/" + name
    loaded = torch.load(path, map_location="cpu")
    if model_type == 1:
        model = Generator()
    else:
        model = Discriminator()
    model.load_state_dict(loaded)
    model.eval()
    return model

def sampler(typ, num):
    """
    funkce, vygeneruje data z nauceneho generatoru
    """
    model = load_model("model_g59.pt",typ)
    sample = sample_shit(model)
    sample = sample[0]
    visualise(sample,num)

def sample_shit(generator, device='cpu'):
    generator.eval()
    noise = Variable(sample_from_distribution(1, 100).float()).to(device)
    generated = generator(noise)
    print(generated.shape)
    return generated.detach().cpu().numpy()

def visualise(sample, num):
    tracker.visualise(sample, num)


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    #main(dir_path)
    sampler(1,5)
