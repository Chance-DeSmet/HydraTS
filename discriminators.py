import torch 
import torch.nn as nn
from torch_wass import torch_wasserstein_loss, torch_reid_loss, torch_single_point_loss
from networks import Generator, Standard_Discriminator, SinglePoint_Discriminator, Privacy_Discriminator
from hydra_utilities import import_data, create_noise, generate, calc_wass_dist,save_generation
from hydra_utilities import calc_even, import_data_batch, calc_quick, score, create_noise_shaped
from hydra_utilities import import_data_all, batch_from_dat_tens, import_data_spec
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NORMALIZE_INPUT = 0 #0 to not normalize, 1 to normalize
SHUFFLE = 1  #0 to shuffle, 1 to not shuffle
MULTI_CSV = True  #Do we pull from multiple CSVS?


class AccuracyDiscriminatorClass:
    '''
    Standard discriminator, analyzing a group of data
    '''
    def __init__(self, num_batches, batch_length, features, data, nz,lr=0.0001, masked=0, inp_channels=0):
        self.name = "_Distribution_Realism"
        self.disc = Standard_Discriminator(batch_length,features, masked, inp_channels).to(device)
        self.optim = torch.optim.SGD(self.disc.parameters(), lr=lr)
        self.disc_scheduler = ReduceLROnPlateau(self.optim, 'min', factor=0.05)
        self.loss = torch.nn.MSELoss()
        self.own_loss_chart = []
        self.gen_loss_chart = []
        self.batch_length = batch_length
        self.num_batches = num_batches
        self.features = features
        self.data = data
        self.last_disc_loss = 0
        self.last_gen_loss = 0
        self.nz = nz
        self.inp_channels = inp_channels
        if(MULTI_CSV):
            self.data_stored = import_data_spec(mode=2,path=self.data,normalize=NORMALIZE_INPUT).to(device)
        else:
            self.data_stored = import_data_all(data, norm=NORMALIZE_INPUT)
    def train(self, gen, g_optim, g_loss, not_collapsed=1,gen_collapsed=0,scalar=1, reps=1):
        noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
        gen_out = gen(noise)
        gen_labs = torch.ones(self.num_batches, device=device)[:,None]

        gen_disc_out = self.disc(gen_out)
        gen_loss = g_loss(gen_disc_out,gen_labs)

        self.last_gen_loss = gen_loss
        self.gen_loss_chart.append(gen_loss)

        gen_loss.backward()
        
        i = 0
        while(i < reps and not_collapsed):
            self.optim.zero_grad()
            
            noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
            
            gen_out = gen(noise)

            new_noise = torch.cuda.FloatTensor(gen_out.shape).uniform_(0, 1)
            if(MULTI_CSV):
                real_data = self.data_stored[torch.randint(self.data_stored.shape[0], (self.num_batches,), device=device),:,:,:] 
                shift_amount = int(real_data.shape[3]/2)
                rand_shift = torch.randint((-1*shift_amount)+1, shift_amount-1, (1,), device=device)
                real_data = torch.roll(real_data, int(rand_shift), 3)
            else:
                real_data = batch_from_dat_tens(self.data_stored,self.num_batches, self.batch_length, non_shuffle=SHUFFLE)
                real_data = torch.Tensor(real_data, device=device)

            
            new_noise = (((torch.cuda.FloatTensor(real_data.shape).uniform_(0, 1)) *0.2) +0.9)
            real_data = real_data * new_noise


            disc_labels_pos = torch.ones(self.num_batches, device=device) 
            disc_labels_neg = torch.zeros(self.num_batches, device=device)
            
            disc_labs_all = torch.cat((disc_labels_pos, disc_labels_neg))
            data_in_all = torch.cat((real_data.float(),gen_out.float()))

            disc_out_all = self.disc(data_in_all)

            disc_loss_all = self.loss(disc_out_all.flatten(), disc_labs_all.flatten())
            
            disc_loss_all.backward()

            self.optim.step()

            self.own_loss_chart.append(disc_loss_all.detach())
            self.last_disc_loss = disc_loss_all.detach()


            i += 1
        return(self.last_gen_loss)
    def test_loss(self):
        return(self.own_loss_chart)
    def get_disc(self):
        return(self.disc)
    
    
class PrivacyDiscriminatorClass:
    '''
    Try to reidentify sensitive attribute(s) from other
    info
    '''
    def __init__(self, num_batches, batch_length, features, data, nz,scalar=1, lr=0.0001, masked=0, inp_channels=0):
        self.name = "_Reidentification"
        self.disc = Privacy_Discriminator().to(device)
        self.optim = torch.optim.SGD(self.disc.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        self.own_loss_chart = []
        self.gen_loss_chart = []
        self.batch_length = batch_length
        self.num_batches = num_batches
        self.features = features
        self.data = data
        self.last_disc_loss = 0
        self.last_gen_loss = 0
        self.nz = nz
        self.set_g_loss = torch_reid_loss()
        if(MULTI_CSV):
            self.data_stored = import_data_spec(mode=2,path=self.data,normalize=NORMALIZE_INPUT).to(device)
        else:
            self.data_stored = import_data_all(data, norm=NORMALIZE_INPUT)
    def train(self, gen, g_optim, g_loss,not_collapsed=1, gen_collapsed=0,scalar=1, reps=1):
        noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
        
        gen_out = gen(noise)
        gen_out = torch.mean(gen_out, 3)[:,3,:][:,None,:]
        gen_labs = torch.zeros(self.num_batches, device=device)[:,None]
        gen_disc_out = self.disc(gen_out)
        gen_loss = g_loss(gen_disc_out,gen_labs)
        self.last_gen_loss = gen_loss
        self.gen_loss_chart.append(gen_loss)
        gen_loss.backward()

        i = 0
        while i < reps:
            self.optim.zero_grad()
            
            noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
            
            gen_out = gen(noise)

            new_noise = torch.cuda.FloatTensor(gen_out.shape).uniform_(0, 1)
            if(MULTI_CSV):
                real_data = self.data_stored[torch.randint(self.data_stored.shape[0], (self.num_batches,), device=device),:,:,:] 
                shift_amount = int(real_data.shape[3]/2)
                rand_shift = torch.randint((-1*shift_amount)+1, shift_amount-1, (1,), device=device)
                real_data = torch.roll(real_data, int(rand_shift), 3)
            else:
                real_data = batch_from_dat_tens(self.data_stored,self.num_batches, self.batch_length, non_shuffle=SHUFFLE)
                real_data = torch.Tensor(real_data, device=device)

            
            new_noise = (((torch.cuda.FloatTensor(real_data.shape).uniform_(0, 1)) *0.2) +0.9)
            real_data = real_data * new_noise
            disc_labels_pos = torch.ones(self.num_batches, device=device) 
            disc_labels_neg = torch.zeros(self.num_batches, device=device) 
            
            disc_labs_all = torch.cat((disc_labels_pos, disc_labels_neg))

            data_in_all = torch.cat((real_data.float(),gen_out.float()))
            data_in_all = torch.mean(data_in_all, 3)[:,3,:][:,None,:]
            disc_out_all = self.disc(data_in_all)
            disc_loss_all = self.loss(disc_out_all.flatten(), disc_labs_all.flatten())
            disc_loss_all.backward()

            self.optim.step()
            self.own_loss_chart.append(disc_loss_all.detach())
            self.last_disc_loss = disc_loss_all.detach()


            i += 1
        return(self.last_gen_loss)
    def test_loss(self):
        return(self.own_loss_chart)
    def get_disc(self):
        return(self.disc)
    

class SinglePointAccuracyDiscriminatorClass:
    def __init__(self, num_batches, batch_length, features, data, nz, lr=0.0001, masked=0, inp_channels=0):
        self.name = "_Point-Wise_Realism"
        self.disc = SinglePoint_Discriminator(batch_length,features,masked, inp_channels).to(device)
        self.optim = torch.optim.SGD(self.disc.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        self.own_loss_chart = []
        self.gen_loss_chart = []
        self.batch_length = batch_length
        self.num_batches = num_batches
        self.features = features
        self.data = data
        self.last_disc_loss = 0
        self.last_gen_loss = 0
        self.nz = nz
        self.single_g_loss = torch_single_point_loss()
        if(MULTI_CSV):
            self.data_stored = import_data_spec(mode=2,path=self.data,normalize=NORMALIZE_INPUT).to(device)
        else:
            self.data_stored = import_data_all(data, norm=NORMALIZE_INPUT).to(device)
    def train(self, gen, g_optim, g_loss,not_collapsed=1, gen_collapsed=0, scalar=1,reps=1):
        g_optim.zero_grad()
        noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
        gen_out = gen(noise)

        gen_labs = torch.ones(self.num_batches,gen_out.shape[1], device=device)
        gen_disc_out = self.disc(gen_out)
        gen_loss = self.single_g_loss(gen_labs.flatten(), gen_disc_out.flatten())

        self.last_gen_loss = gen_loss.detach()
        self.gen_loss_chart.append(gen_loss.detach())
        gen_loss.backward()
        i = 0
        while i < reps:
            self.optim.zero_grad()
            noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
            
            gen_out = gen(noise)
            if(MULTI_CSV):
                real_data = self.data_stored[torch.randint(len(self.data_stored), (self.num_batches,), device=device),:,:,:] 
                shift_amount = int(real_data.shape[3]/2)
                rand_shift = torch.randint((-1*shift_amount)+1, shift_amount-1, (1,), device=device)
                real_data = torch.roll(real_data, int(rand_shift), 3)
            else:
                real_data = batch_from_dat_tens(self.data_stored,self.num_batches, self.batch_length, non_shuffle=SHUFFLE)
                real_data = torch.Tensor(real_data, device=device)

            new_noise = (((torch.cuda.FloatTensor(real_data.shape).uniform_(0, 1)) *0.2) +0.9)
            real_data = real_data * new_noise

            disc_labels_pos = torch.ones(self.num_batches, gen_out.shape[1], device=device) 
            disc_labels_neg = torch.zeros(self.num_batches, gen_out.shape[1], device=device) 
            disc_labs = torch.cat((disc_labels_pos, disc_labels_neg))
            disc_dat = torch.cat((real_data.float(), gen_out.float()))
            disc_out = self.disc(disc_dat)
            disc_loss = self.loss(disc_out.flatten(), disc_labs.flatten())
            disc_loss.backward()
            self.optim.step()


            self.own_loss_chart.append(disc_loss.detach())
            self.last_disc_loss = disc_loss.detach()

            i += 1
        return(self.last_gen_loss)
    def test_loss(self):
        return(self.own_loss_chart)
    def get_disc(self):
        return(self.disc)    




class AvoidanceDiscriminatorClass:
    def __init__(self, num_batches, batch_length, features, data, nz,lr=0.0001, masked=0, inp_channels=0):
        self.name = "_Avoided_Data"

        self.disc = Standard_Discriminator(batch_length,features, masked, inp_channels).to(device)
        self.optim = torch.optim.SGD(self.disc.parameters(), lr=lr)
        self.disc_scheduler = ReduceLROnPlateau(self.optim, 'min', factor=0.05)
        self.loss = torch.nn.MSELoss()
        self.own_loss_chart = []
        self.gen_loss_chart = []
        self.batch_length = batch_length
        self.num_batches = num_batches
        self.features = features
        self.data = data
        self.last_disc_loss = 0
        self.last_gen_loss = 0
        self.nz = nz
        self.inp_channels = inp_channels
        if(MULTI_CSV):
            self.data_stored = import_data_spec(mode=2,path=self.data,normalize=NORMALIZE_INPUT).to(device)
        else:
            self.data_stored = import_data_all(data, norm=NORMALIZE_INPUT)
    def train(self, gen, g_optim, g_loss, not_collapsed=1,gen_collapsed=0,scalar=1, reps=1):
        noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
        
        gen_out = gen(noise)
        gen_labs = torch.zeros(self.num_batches, device=device)[:,None]

        gen_disc_out = self.disc(gen_out)
        gen_loss = g_loss(gen_disc_out,gen_labs)

        self.last_gen_loss = gen_loss.detach()
        self.gen_loss_chart.append(gen_loss.detach())
        
        gen_loss.backward()
        i = 0
        while(i < reps and not_collapsed):
            self.optim.zero_grad()
            
            noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
            
            gen_out = gen(noise)
            new_noise = (((torch.cuda.FloatTensor(gen_out.shape).uniform_(0, 1)) *0.2) +0.9)
            if(MULTI_CSV):
                real_data = self.data_stored[torch.randint(self.data_stored.shape[0], (self.num_batches,), device=device),:,:,:] 
                shift_amount = int(real_data.shape[3]/2)
                rand_shift = torch.randint((-1*shift_amount)+1, shift_amount-1, (1,), device=device)
                real_data = torch.roll(real_data, int(rand_shift), 3)
            else:
                real_data = batch_from_dat_tens(self.data_stored,self.num_batches, self.batch_length, non_shuffle=SHUFFLE)
                real_data = torch.Tensor(real_data, device=device)

            
            new_noise = (((torch.cuda.FloatTensor(real_data.shape).uniform_(0, 1)) *0.2) +0.9)
            real_data = real_data * new_noise
            disc_labels_pos = torch.ones(self.num_batches, device=device) 
            disc_labels_neg = torch.zeros(self.num_batches, device=device) 
            
            disc_labs_all = torch.cat((disc_labels_pos, disc_labels_neg))
            data_in_all = torch.cat((real_data.float(),gen_out.float()))
            disc_out_all = self.disc(data_in_all)

            disc_loss_all = self.loss(disc_out_all.flatten(), disc_labs_all.flatten())
            disc_loss_all.backward()

            self.optim.step()

            self.own_loss_chart.append(disc_loss_all.detach())
            self.last_disc_loss = disc_loss_all.detach()


            i += 1
        return(self.last_gen_loss)
    def test_loss(self):
        return(self.own_loss_chart)
    def get_disc(self):
        return(self.disc)