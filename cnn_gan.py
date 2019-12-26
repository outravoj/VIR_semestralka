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

"ZATÍM JSEM NETRÉNOVAL NA SERVERECH === TŘEBA OTESTOVAT A DOLADIT"


def load_data(batch_size, filename, device, shuffle=True, store_dir='./data'):
    """
    data nactena jako batch velikosti batchsize reprezentuje ve formatu [B, C, F], kde B je batchsize
    část [1, C, F], kde C je 2*68 - pocet znaků v obrázku, a F je 80 - frames ve 4 sekundách je jeden vzorek
    load_data načítá reálná data z file input.npy
    znaky vystupují jako channely konvoluce, blízké framy jsou spojovány konvolučním jádrem
    """
    p = np.load(osp.join(store_dir, filename))
    data = torch.from_numpy(p.astype('float32'))  # squeeze into correct dtype and range [0-1]
    labels = torch.from_numpy(np.ones((data.shape[0],1)).astype("float32"))
    data = data.cuda()
    labels = labels.cuda()
    dataset = torch.utils.data.TensorDataset(data, labels)
    print(data.shape)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

class Discriminator(torch.nn.Module):
    """
    zatím 1D konvoluční síť, 4 vrstvy konvoluce, na konci lineárka a sigmoida - chceme na závěr 2 psti - pokud 1, pak si síť myslí, že
    obrázek je real, pokud 0, tak fake
    """
    def __init__(self, num_of_channels):
        super(Discriminator,self).__init__()
        self.num_of_channels =  num_of_channels
        self.conv1 = nn.Conv1d(self.num_of_channels, 72, kernel_size=5)
        self.conv2 = nn.Conv1d(72, 48, kernel_size=8)
        self.conv3 = nn.Conv1d(48, 24, kernel_size=4)
        self.conv4 = nn.Conv1d(24,8, kernel_size=4)
        self.lin1 = nn.Linear((63) * 8, 1)
        self.seq = nn.Sequential(self.conv1, nn.InstanceNorm1d(72),nn.LeakyReLU(0.2),
                                 self.conv2, nn.InstanceNorm1d(48),nn.LeakyReLU(0.2), nn.Dropout(),
                                 self.conv3, nn.InstanceNorm1d(24),nn.LeakyReLU(0.2),
                                 self.conv4, nn.InstanceNorm1d(8),nn.LeakyReLU(0.2))
        self.weights_initialization()

    def weights_initialization(self):
        for layer in self.named_parameters():
            if type(layer) in {nn.Conv1d, nn.Linear,nn.BatchNorm1d}:
                nn.init.xavier_normal_(layer.weight.data, gain=torch.sqrt(2))

    def forward(self, x):
        x = self.seq(x)
        x = torch.sigmoid(self.lin1(x.view(-1,(63) * 8)))
        return x

class Discriminator2(torch.nn.Module):
    """
    zatím 1D konvoluční síť, 4 vrstvy konvoluce, na konci lineárka a sigmoida - chceme na závěr 2 psti - pokud 1, pak si síť myslí, že
    obrázek je real, pokud 0, tak fake


    TODO PŘIDAT DALŠÍ VRSTVU, ASI NESTAČÍ, POUZIT DISC1 JAKO VZOR
    """
    def __init__(self, num_of_channels):
        super(Discriminator2,self).__init__()
        self.num_of_channels =  num_of_channels
        self.conv1 = nn.Conv1d(self.num_of_channels, 48, kernel_size=6)
        self.conv2 = nn.Conv1d(48, 24, kernel_size=6)
        self.conv3 = nn.Conv1d(24, 12, kernel_size=6)
        self.lin1 = nn.Linear((80-15) * 12, 1)
        self.seq = nn.Sequential(self.conv1, nn.InstanceNorm1d(48),nn.LeakyReLU(0.2),
                                 self.conv2, nn.InstanceNorm1d(24),nn.LeakyReLU(0.2),
                                 self.conv3, nn.InstanceNorm1d(12),nn.LeakyReLU(0.2))
        self.weights_initialization()


    def weights_initialization(self):
        for layer in self.named_parameters():
            if type(layer) in {nn.Conv1d, nn.Linear,nn.BatchNorm1d}:
                nn.init.xavier_normal_(layer.weight.data, gain=torch.sqrt(2))

    def forward(self, x):
        x = self.seq(x)
        x = torch.sigmoid(self.lin1(x.view(-1,(80-15) * 12)))
        return x


class Generator(torch.nn.Module):
    def __init__(self,num):
        """
        nepouzivat
        zatím vezmeme noise vektor délky 100, a roztáhneme ho linárkou, pak ho sypem do konvolucni site naladene tak, aby na vysledku
        byl pro jeden vektor 100x1 datovy format 2*68x80
        v mnoha textech doporucuji leaky relu, ale nekde jen v diskriminatoru, trena nastudovat a upravit
        možná bude lepší batchnorm misto instance norm - potřeba okouknout
        """
        super(Generator, self).__init__()
        self.lin = nn.Linear(100,7200)
        self.conv1t = nn.Conv1d(20, 40, kernel_size=8,stride=2)
        self.conv2t = nn.Conv1d(40, 60, kernel_size=8, stride=2)
        self.conv3t = nn.Conv1d(60,80, kernel_size=4)
        self.conv4t = nn.Conv1d(80, 100, kernel_size=4)
        self.seq = nn.Sequential(
                        self.conv1t, nn.LeakyReLU(negative_slope=0.2), nn.InstanceNorm1d(40),
                        self.conv2t, nn.LeakyReLU(negative_slope=0.2), nn.InstanceNorm1d(60),
                        self.conv3t, nn.LeakyReLU(negative_slope=0.2), nn.InstanceNorm1d(80),
                        self.conv4t, nn.LeakyReLU(negative_slope=0.2), nn.InstanceNorm1d(100),
                        self.conv5t)
        self.lr = nn.LeakyReLU(negative_slope=0.2)

    def weights_initialization(self):
        for layer in self.named_parameters():
            if type(layer) in {nn.Conv1d, nn.Linear,nn.BatchNorm1d}:
                nn.init.xavier_normal_(layer.weight.data, gain=torch.sqrt(2))

    def forward(self, x):
        batch = x.shape[0]
        x = F.leaky_relu(self.lin(x),negative_slope=0.2)
        x = x.view(batch,20,360)
        return self.seq(x)


class Generator2(torch.nn.Module):
    def __init__(self, channels):
        super(Generator2, self).__init__()
        self.conv1t = nn.ConvTranspose1d(100, channels, kernel_size=4,stride=4, bias=False)
        self.conv2t = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, bias=False)
        self.conv3t = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, bias=False)
        self.conv4t = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, bias=False)
        self.conv5t = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, bias=False)
        self.tanh = nn.Tanh()
        self.seq = nn.Sequential(
                        self.conv1t, nn.InstanceNorm1d(channels), nn.ReLU(),
                        self.conv2t, nn.InstanceNorm1d(channels),nn.ReLU(),
                        self.conv3t, nn.InstanceNorm1d(channels), nn.ReLU(), nn.Dropout(),
                        self.conv4t, nn.InstanceNorm1d(channels), nn.ReLU(),
                        self.conv5t)
        self.weights_initialization()

    def weights_initialization(self):
        for layer in self.named_parameters():
            if type(layer) in {nn.BatchNorm1d,nn.ConvTranspose1d, nn.Linear}:
                nn.init.xavier_normal_(layer.weight.data, gain=torch.sqrt(2))

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch,100,1)
        x =self.seq(x)
        x = x[:,:,:80]
        return x#5 * self.tanh(x / 5)




def sample_from_distribution(number_of_samples, dims_1, dims_2):
    """
    vytváří normalizované vektory délky dims, vrátí jich to number_of_samples
    number_of_samples jsou typicky batchsize, normální rozdělení
    """
    samples = torch.from_numpy(np.zeros((number_of_samples, dims_1, dims_2)))
    for i in range(number_of_samples):
        sample = torch.randn(samples[0].shape)
        for j in range(dims_1):
            sample[j] = sample[j] / torch.norm(sample[j])
        samples[i] = sample
    return samples


def learn(discriminator, generator, opt_d, opt_g, loader, epochs, device):
    discriminator.train()
    generator.train()
    fake_label = 0 #faky jako 0, realy jako 1
    real_label = 1
    diskr = []
    diskf = []
    gen = []
    for i in range(epochs):
        dr = 0
        df = 0
        g = 0
        j = 0
        gen_ = 8
        for batch, _ in loader:
            #print(batch.shape, real_labels.shape)
            #trenink diskriminatoru"
            """
            hyperparametry pro řízení iterací nad jednotlivými sítěmi
            """
            num = batch.shape[0]

            discriminator.zero_grad()
            # vytáhni batch z dat
            real_pred = discriminator(batch)
            """
            často je doporučováno zašumět labely, tedy ne čistě 1, ale třeba 0.8 - 1.2 - zkontrolovat
            """
            real_labels = torch.full(real_pred.shape, real_label) + (torch.rand(real_pred.shape) - 0.5) * 0.4
            real_labels = real_labels.cuda()


            errD_real = F.binary_cross_entropy(real_pred, real_labels)
            errD_real.backward()
            # bereme CL loss, chceme ji minimalizovat
            # backward, ale čekáme na fake data
            D_x = real_pred.mean().item()


            noise = Variable(sample_from_distribution(num, 1, 100).float()).cuda()  # sample fake dat podle batche
            generated = generator(noise)  # vygeneruj je na generatoru
            fake_pred = discriminator(generated)  # odhady diskriminatoru na fake datech
            fake_labels = torch.full(fake_pred.shape, fake_label + 0.1) + (torch.rand(fake_pred.shape) - 0.5) * 0.2  # opět šumím labely na 0.0 - 0.3
            fake_labels = fake_labels.cuda()
            errD_fake = F.binary_cross_entropy(fake_pred, fake_labels)  # fake loss
            errD_fake.backward()
            D_G_z1 = fake_pred.mean().item()
            dr += (float(errD_real.item()))
            df += (float(errD_fake.item()))
            opt_d.step()  # konečně step


            if errD_fake.item() > 0.8:
                k=0
            elif errD_fake.item() < 0.3:
                k = 3
            elif errD_fake.item() < 0.5:
                k=2
            else:
                k=1

            for s in range(k):
                opt_g.zero_grad()
                noise = Variable(sample_from_distribution(num, 1, 100).float()).cuda()  # sample novýho noisu a generace nových faků
                generated = generator(noise)
                pred = discriminator(generated)
                labels = torch.full(pred.shape,
                                    real_label)  # jako labely beru jedničky, i když jsou to faky - kvůli lepšímu trénování generátoru
                labels = labels.cuda()
                errG = F.binary_cross_entropy(pred, labels)
                errG.backward()
                D_G_z2 = pred.mean().item()
                opt_g.step()  # normálně backprop na generátoru
                gen_ = float(errG.item())
            if k == 0:
                pass
            else:
                gen_ = gen_ / k

            if (i+1)%20==0:
                f = str(i) + ".pt"
                torch.save(discriminator.state_dict(), "model_d" + f)
                torch.save(generator.state_dict(), "model_g" + f)
            print('[%d/%d] Loss_D_real: %.4f Loss_D_fake:  %.4f Loss_G: %.4f'
                  % (i, epochs,
                     errD_real.item(), errD_fake.item(), gen_))
            g += gen_
            j+=1
        diskr.append(dr/j)
        diskf.append(df/j)
        gen.append(g/j)
    return diskr, diskf, gen

def parse_args():
    parser = argparse.ArgumentParser('Simple MNIST classifier')
    parser.add_argument('--epochs', '-e', default=200, type=int)
    parser.add_argument('--batch_size', '-bs', default=128, type=int)
    parser.add_argument('--store_dir', '-sd', default='./', type=str)
    return parser.parse_args()


def main(path=""):
    args = parse_args()
    #device = torch.device('cuda')
    """if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        torch.cuda.empty_cache()"""
    if path == "":
        path = PATH
    loader = load_data(args.batch_size, "ultimate_dataset_plus_reduced.npy", device=2, store_dir=path, shuffle=True)
    discriminator = Discriminator(2*53)
    generator = Generator2(2*53)

    generator.cuda()
    discriminator.cuda()

    #lr zatím pro oba stejný, potřebujeme 2 optimizery, možná 2 lr?"
    lr_d = 0.0002
    lr = 0.0002

    #jeden optimizer pro diskriminator a druhy pro generator"
    opt_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    dr,df,g = learn(discriminator, generator, opt_d, opt_g, loader, args.epochs, 2)

    "vsechno uloz"
    torch.save(discriminator.state_dict(), "model_d.pt")
    torch.save(generator.state_dict(), "model_g.pt")
    mem = np.zeros((3,len(dr)))
    mem[0] = np.array(dr)
    mem[1] = np.array(df)
    mem[2] = np.array(g)
    np.save("lossy",mem)


def visualize_losses(name):
    data = np.load(name)
    tracker.plot_curves(data)

def load_model(name, model_type, chann=136):
    dir_path = osp.dirname(os.path.realpath(__file__))
    path = dir_path + "/" + name
    loaded = torch.load(path, map_location="cpu")
    if model_type == 1:
        model = Generator(2*68)
    elif model_type == 2:
        model = Generator2(2*53)
    else:
        model = Discriminator(chann)
    model.load_state_dict(loaded)
    model.eval()
    return model



def sampler(typ, num):
    """
    funkce, vygeneruje data z nauceneho generatoru
    """
    model = load_model("model_g899.pt",typ)
    sample = sample_shit(model)
    sample = sample[0]
    visualise(sample,num)

def sample_shit(generator, device='cpu'):
    generator.eval()
    noise = Variable(sample_from_distribution(1,1, 100).float()).to(device)
    generated = generator(noise)
    print(generated.shape)
    return generated.detach().cpu().numpy()

def visualise(sample, num):
    "funkce pro vizualizaci, z tracker.py, možná bude třeba zakomentovat nějaké importy"
    tracker.visualise(sample, num)

if __name__ == '__main__':
    """
    je třeba rozlišovat mezi input se 136 a 106 kanály - input resp reduced_input, vznikají oba v transform.py
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    #main(dir_path)
    sampler(2,3)
    #sample = np.load("ultimate_dataset_reduced.npy")
    #visualise(sample[666],12)
    #print(np.load("lossy.npy"))
    #visualize_losses("lossy.npy")