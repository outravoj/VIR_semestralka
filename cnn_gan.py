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
import torch.distributions
import tracker
from torch.autograd import Variable

PATH = "/local/temporary/vir/hw01/pkl"

def load_data(batch_size, filename, device, shuffle=True, store_dir='./data'):
    """
    standardni nacitani
    """
    print(osp.join(store_dir, filename))
    p = np.load(osp.join(store_dir, filename))
    data = torch.from_numpy(p.astype('float32'))  # squeeze into correct dtype and range [0-1]
    labels = torch.from_numpy(np.zeros((data.shape[0],1)).astype("float32"))
    data = data.to(device)
    labels = labels.to(device)
    print(data.shape)
    dataset = torch.utils.data.TensorDataset(data, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=80, num_of_features=2*68):
        """
        diskriminace pomoci 1d konvoluce, instance normy, protoze zimmerman rikal ze jsou cool
        """
        super(Discriminator,self).__init__()
        self.num_of_f = num_of_features
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv1d(32,8, kernel_size=5)
        self.lin1 = nn.Linear((num_of_features-14) * 8, 1)
        self.seq = nn.Sequential(self.conv1, nn.LeakyReLU(0.2), nn.InstanceNorm1d(32),
                                 self.conv2, nn.LeakyReLU(0.2), nn.InstanceNorm1d(16),
                                 self.conv3, nn.LeakyReLU(0.2), nn.InstanceNorm1d(8))
        self.weights_initialization()

    def weights_initialization(self):
        for layer in self.named_parameters():
            if type(layer) in {nn.Conv1d, nn.Linear}:
                nn.init.xavier_uniform_(layer.weight.data, gain=torch.sqrt(2))

    def forward(self, x):
        #vymyslet format dat
        x = self.seq(x)
        x = torch.sigmoid(self.lin1(x.view(-1,(self.num_of_f-14) * 8)))
        return x


class Generator1(torch.nn.Module):
    def __init__(self):
        super(Generator1, self).__init__()
        self.lin = nn.Linear(20,30)
        self.conv1t = nn.ConvTranspose1d(1, 16, kernel_size=4, stride=2)
        self.conv2t = nn.ConvTranspose1d(16, 32, kernel_size=3, stride=1)
        self.conv3t = nn.ConvTranspose1d(32,64, kernel_size=4, stride=1)
        self.conv4t = nn.ConvTranspose1d(64, 80, kernel_size=4, stride=2)
        self.seq = nn.Sequential(
                        self.conv1t, nn.LeakyReLU(negative_slope=0.2), nn.InstanceNorm1d(16),
                        self.conv2t, nn.LeakyReLU(negative_slope=0.2), nn.InstanceNorm1d(32),
                        self.conv3t, nn.LeakyReLU(negative_slope=0.2), nn.InstanceNorm1d(64),
                        self.conv4t)
        self.weights_initialization()
        self.lr = nn.LeakyReLU(negative_slope=0.2)

    def weights_initialization(self):
        for layer in self.named_parameters():
            if type(layer) in {nn.ConvTranspose1d}:
                nn.init.xavier_uniform_(layer.weight.data, gain=torch.sqrt(2))

    def forward(self, x):
        x = F.leaky_relu(self.lin(x),negative_slope=0.2)
        return self.seq(x)


class Generator2(torch.nn.Module):
    def __init__(self):
        super(Generator2, self).__init__()
        self.lin = nn.Linear(20,2880)
        self.conv1t = nn.Conv1d(20, 40, kernel_size=3)
        self.conv2t = nn.Conv1d(40, 60, kernel_size=3)
        self.conv3t = nn.Conv1d(60,80, kernel_size=3)
        self.conv4t = nn.Conv1d(80, 80, kernel_size=3)
        self.seq = nn.Sequential(
                        self.conv1t, nn.LeakyReLU(negative_slope=0.2), nn.InstanceNorm1d(40),
                        self.conv2t, nn.LeakyReLU(negative_slope=0.2), nn.InstanceNorm1d(60),
                        self.conv3t, nn.LeakyReLU(negative_slope=0.2), nn.InstanceNorm1d(80),
                        self.conv4t)
        self.weights_initialization()
        self.lr = nn.LeakyReLU(negative_slope=0.2)

    def weights_initialization(self):
        for layer in self.named_parameters():
            if type(layer) in {nn.ConvTranspose1d}:
                nn.init.xavier_uniform_(layer.weight.data, gain=torch.sqrt(2))

    def forward(self, x):
        batch = x.shape[0]
        x = F.leaky_relu(self.lin(x),negative_slope=0.2)
        x = x.view(batch,20,144)
        return self.seq(x)


def sample_from_distribution(number_of_samples, dims, distro):
    samples = torch.from_numpy(np.zeros((number_of_samples, 1, dims)))
    for i in range(number_of_samples):
        sample = distro.sample((1,dims))[0].permute((1,0))
        sample = sample / torch.norm(sample)
        samples[i, 0,:] = sample
    return  samples


def accuracy(prediction, labels_batch, dim=-1):
    pred_index = prediction.argmax(dim)
    return (pred_index == labels_batch).float().mean()


def learn(discriminator, generator, opt_d, opt_g, bs, loader, epochs, device):
    discriminator.train()
    generator.train()
    distro = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    for i in range(epochs):
        print("epoch {}".format(i))
        j = 0
        avg_d_loss = 0
        for batch, real_labels in loader:
            #print(batch.shape, real_labels.shape)
            "trenink diskriminatoru"
            num = batch.shape[0]
            noise = Variable(sample_from_distribution(num, 20, distro).float()).to(device)
            fake_labels = Variable(torch.from_numpy(np.ones((num, 1)).astype("float32"))).to(device)
            generated = generator(noise)
            pred_real = discriminator(batch)
            pred_fake = discriminator(generated)
            opt_d.zero_grad()
            d1_loss = F.binary_cross_entropy(pred_real, real_labels)
            d2_loss = F.binary_cross_entropy(pred_fake, fake_labels)
            d_loss = d1_loss + d2_loss
            avg_d_loss += d_loss.detach().cpu().numpy()
            d_loss.backward()
            opt_d.step()
            j+=1
        print("d_loss = {}".format(avg_d_loss / j))

        "trenink generatoru"
        noise = sample_from_distribution(bs * 4, 20, distro).to(device).float()
        generated = generator(noise)
        labels = torch.from_numpy(np.ones((bs * 4, 1)).astype("float32")).to(device)
        pred = discriminator(generated)
        opt_g.zero_grad()
        g_loss = -F.binary_cross_entropy(pred, labels)
        g_loss.backward()
        print("g_loss = {}".format(g_loss.detach().cpu().numpy()))
        opt_g.step()


def evaluate(model, val_data):
    """neni zatim k nicemu"""
    with torch.no_grad():
        acc = 0
        for x_batch, y_batch in val_data:
            prediction = model(x_batch)
            acc += accuracy(prediction, y_batch) * x_batch.shape[0]
        print('Evaluate: accuracy: {:.5f}'.format(acc / len(val_data.dataset)))
        return acc / len(val_data.dataset)


def parse_args():
    parser = argparse.ArgumentParser('Simple MNIST classifier')
    parser.add_argument('--epochs', '-e', default=50, type=int)
    parser.add_argument('--batch_size', '-bs', default=64, type=int)
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
    loader = load_data(args.batch_size, "correct_data.npy", device=device, store_dir=path, shuffle=True)
    discriminator = Discriminator()
    generator = Generator2()
    if device:
        generator.to(device)
        discriminator.to(device)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.005, weight_decay=0.001)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, weight_decay=0.001)
    learn(discriminator, generator, opt_d, opt_g, args.batch_size, loader, args.epochs, device)
    torch.save(discriminator.state_dict(), "model_d.pt")
    torch.save(generator.state_dict(), "model_g.pt")


def load_model(name, model_type):
    dir_path = osp.dirname(os.path.realpath(__file__))
    path = dir_path + "/" + name
    loaded = torch.load(path, map_location="cpu")
    if model_type == 1:
        model = Generator1()
    elif model_type == 2:
        model = Generator1()
    else:
        model = Discriminator()
    model.load_state_dict(loaded)
    model.eval()
    return model

def sampler(typ):
    """
    funkce, vygeneruje data z nauceneho generatoru
    """
    model = load_model("model_g.pt",typ)
    sample = sample_shit(model)
    visualise(sample)

def sample_shit(generator, device='cpu'):
    generator.eval()
    distro = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    noise = Variable(sample_from_distribution(1, 20, distro).float()).to(device)
    generated = generator(noise)
    return generated.detach().cpu().numpy()

def visualise(sample, scale=2):
    tracker.visualise(sample[0], scale)


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    main(dir_path)
    #sampler(2)