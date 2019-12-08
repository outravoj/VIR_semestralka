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
    data = data.to(device)
    labels = labels.to(device)
    dataset = torch.utils.data.TensorDataset(data, labels)
    print(data.shape)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

class Discriminator(torch.nn.Module):
    """
    zatím 1D konvoluční síť, 4 vrstvy konvoluce, na konci lineárka a sigmoida - chceme na závěr 2 psti - pokud 1, pak si síť myslí, že
    obrázek je real, pokud 0, tak fake
    """
    def __init__(self, num_of_features,batch_size):
        super(Discriminator,self).__init__()
        self.batch_size = batch_size
        self.lstm = nn.LSTM(num_of_features,30,5)
        self.lin1 = nn.Linear(30,1)

        self.weights_initialization()

    def weights_initialization(self):
        for layer in self.named_parameters():
            if type(layer) in {nn.LSTM, nn.Linear}:
                nn.init.xavier_uniform_(layer.weight.data, gain=torch.sqrt(2))

    def forward(self, x,batch_size):
        x,h = self.lstm(x)
        x = x[-1].view(batch_size,-1)
        x = torch.sigmoid(self.lin1(x))
        return x


class Generator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(100,106,6)
        self.lstm2 = nn.LSTM(106,106,5)
        self.weights_initialization()


    def forward(self, noise):
        x,_ = self.lstm(noise)
        x,_ = self.lstm2(x)
        return x

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def weights_initialization(self):
        for layer in self.named_parameters():
            if type(layer) in {nn.LSTM, nn.Linear}:
                nn.init.xavier_uniform_(layer.weight.data, gain=torch.sqrt(2))





def sample_from_distribution(number_of_samples, dims_1,dims2):
    """
    vytváří normalizované vektory délky dims, vrátí jich to number_of_samples
    number_of_samples jsou typicky batchsize, normální rozdělení
    """
    samples = (torch.from_numpy(np.random.rand(number_of_samples, dims_1,dims2))-25)*50
    return samples


def learn(discriminator, generator, opt_d, opt_g, loader, epochs, device):
    discriminator.train()
    generator.train()
    generator.double()
    discriminator.double()
    fake_label = 0 #faky jako 0, realy jako 1
    real_label = 1
    for i in range(epochs):
        print("EPOCH: ",i)
        for batch, _ in loader:
            #print(batch.shape, real_labels.shape)
            #trenink diskriminatoru"
            num = batch.shape[0]
            batch = batch.permute(2,0,1)
            """
            hyperparametry pro řízení iterací nad jednotlivými sítěmi
            """

            discriminator.zero_grad()
            # vytáhni batch z dat
            real_pred = discriminator(batch.double(),num)

            """
            často je doporučováno zašumět labely, tedy ne čistě 1, ale třeba 0.8 - 1.2 - zkontrolovat
            """
            real_labels = torch.full(real_pred.shape, real_label) + (torch.rand(real_pred.shape) - 0.5) * 0.4
            real_labels = real_labels.to(device)

            errD_real = F.binary_cross_entropy(real_pred.float(), real_labels.float())
            print("Loss na real: ",errD_real)
            #print("Real_predictions: ",real_pred)
            # bereme CL loss, chceme ji minimalizovat
            # backward, ale čekáme na fake data
            D_x = real_pred.mean().item()


            noise = Variable(sample_from_distribution(80,60,100).double()).to(device)  # sample fake dat podle batche
            generated = generator(noise)


            fake_pred = discriminator(generated.double(),60)  # odhady diskriminatoru na fake datech

            fake_labels = torch.full(fake_pred.shape, fake_label + 0.15) + (
                        torch.rand(fake_pred.shape) - 0.5) * 0.3  # opět šumím labely na 0.0 - 0.3
            fake_labels = fake_labels.to(device)

            errD_fake = F.binary_cross_entropy(fake_pred.float(), fake_labels.float())  # fake loss
            print("Loss na fake: ", errD_fake)
            D_G_z1 = fake_pred.mean().item()
            errD = errD_fake + errD_real
            errD.backward()
            opt_d.step()  # konečně step

#    -------------        GENERATOR TRAINNG PART       -----------------

            opt_g.zero_grad()

            noise = Variable(sample_from_distribution(80, 60, 100).double()).to(device)  # sample fake dat podle batche
            generated = generator(noise)

            pred = discriminator(generated.double(),60)

            labels = torch.full(pred.shape,
                                real_label)  # jako labely beru jedničky, i když jsou to faky - kvůli lepšímu trénování generátoru
            labels = labels.to(device)
            errG = F.binary_cross_entropy(pred.float(), labels.float())
            #print("Labels of generaed data: ",pred)
            errG.backward()
            D_G_z2 = pred.mean().item()
            opt_g.step()  # normálně backprop na generátoru

            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (i, epochs,
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


        if (i+1)%20==0:
            f = str(i) + ".pt"
            torch.save(discriminator.state_dict(), "models/model_d" + f)
            torch.save(generator.state_dict(), "models/model_g" + f)


def parse_args():
    parser = argparse.ArgumentParser('Simple MNIST classifier')
    parser.add_argument('--epochs', '-e', default=100, type=int)
    parser.add_argument('--batch_size', '-bs', default=128, type=int)
    parser.add_argument('--store_dir', '-sd', default='./', type=str)
    return parser.parse_args()


def main(path=""):
    args = parse_args()
    device = torch.device('cuda')
    if torch.cuda.is_available():
        print("Cuda available, running on gpu!")
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        torch.cuda.empty_cache()
    if path == "":
        path = PATH
    loader = load_data(args.batch_size, "datasets/ultimate_dataset_reduced.npy", device=device, store_dir=path, shuffle=True)
    discriminator = Discriminator(2*53,128)
    generator = Generator(100,106,106)
    if device:
        print("Both models on: !",device)
        generator.to(device)
        discriminator.to(device)

    #lr zatím pro oba stejný, potřebujeme 2 optimizery, možná 2 lr?"
    lr_d = 0.0005
    lr = 0.001

    #jeden optimizer pro diskriminator a druhy pro generator"
    opt_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    learn(discriminator, generator, opt_d, opt_g, loader, args.epochs, device)
    torch.save(discriminator.state_dict(), "models/model_d.pt")
    torch.save(generator.state_dict(), "models/model_g.pt")


def load_model(name, model_type, chann=136):
    dir_path = osp.dirname(os.path.realpath(__file__))
    path = dir_path + "/" + name
    loaded = torch.load(path, map_location="cpu")
    if model_type == 1:
        model = Generator(100,106,106)
    else:
        model = Discriminator(chann)
    model.load_state_dict(loaded)
    model.eval()
    return model

def sampler(typ, num):
    """
    funkce, vygeneruje data z nauceneho generatoru
    """
    generator = load_model("models/model_g59.pt",typ)
    device = torch.device('cuda')
    generator.to(device)
    generator.double()
    noise = Variable(sample_from_distribution(80, 1, 100).double()).to(device)  # sample fake dat podle batche
    generated = generator(noise)
    generated = generated.permute(1,2,0)[0,:,:]
    visualise(generated,num)

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
    sampler(1, 5)
