import torch.nn as nn
import torch
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device, torch.cuda.is_available())
print("Command line arguments:", sys.argv, flush=True)
print(torch.cuda.get_device_name())
print(torch.__version__)           # 1.9.0
print(torch.version.cuda)           # 11.1
print(torch.cuda.is_available())     #True
def expanded_sigmoid(x):
    absv= torch.abs(x)
    sign = torch.div(x,torch.add(absv,0.000001))
    ret = torch.log(torch.add(absv, 1))
    retu = torch.mul(ret,sign)
    return(retu)

def t_sin(x):
    retu = torch.sin(x)
    return(retu)
    

class Generator(nn.Module):
    
    def __init__(self,nz,out_length,time_bins, num_sensors):
        
        super(Generator, self).__init__()
        self.nz = nz
        self.freq_bins = out_length
        self.time_bins = time_bins
        self.second_hard_t = nn.Hardtanh(0,1)
        self.multihead_attn = nn.MultiheadAttention(1,1, batch_first=True)        
        self.first_hard_t = nn.ReLU()
        self.num_sensors = num_sensors
        self.lin_upsample = nn.Sequential(
            nn.Linear(16,32),
            nn.SELU(),
            nn.Linear(32,64),
            nn.SELU(),
            
            )
        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (2,8), stride=(1)),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, (2,8), stride=(2)),  
            nn.BatchNorm2d(16),
            nn.SELU(),
            nn.ConvTranspose2d(16,8, (4,12), stride=(2)),  
            nn.BatchNorm2d(8),
            nn.SELU(),
            nn.ConvTranspose2d(8,self.num_sensors, (5,31), stride=(2)), 
            )
        
    def forward(self, input):
        out = input
        out = self.lin_upsample(out)
        out = out.reshape(out.shape[0],64,1,1)
        out = self.conv_trans(out)
        out = torch.sin(out)
        out = 100*out
        return(out)


class Standard_Discriminator(nn.Module):
    def __init__(self,freq_bins,time_bins, masked,inp_channels):
        super(Standard_Discriminator, self).__init__()
        self.freq_bins = freq_bins
        self.time_bins = time_bins - masked
        self.masked = masked
        self.multihead_attn = nn.MultiheadAttention(1,1, batch_first=True)
        self.inp_channels = inp_channels
        self.conv_layer = nn.Sequential(
            nn.Conv2d(self.inp_channels,8,(8,32),stride=2,padding=(2),padding_mode='circular'),
            nn.SELU(),
            nn.Conv2d(8,8,(4,16),stride=2,padding=(2),padding_mode='circular'),
            nn.SELU(),
            nn.Conv2d(8,8,(3,8),stride=1,padding=0,padding_mode='circular'),
            nn.SELU(),
            nn.Conv2d(8,8,(3,3),stride=1,padding=0,padding_mode='circular'),
            nn.SELU(),
            nn.Flatten(start_dim=1),
            )
        self.lin_layer = nn.Sequential(
            nn.Linear(64,32),
            nn.SELU(),
            nn.Linear(32,16),
            nn.SELU(),
            nn.Linear(16,16),
            nn.Linear(16,1),
            )
    def forward(self, input):
        bound_1 = torch.randint(low=0, high=self.masked, size=(1,))
        bound_2 = self.masked - bound_1
        inp = input[:,:,:,bound_1:-bound_2].clone()
        #print("Input shape is:", inp.shape)
        out = self.conv_layer(inp)
        out = self.lin_layer(out)
        out = torch.sin(out)
        out = torch.add(out,1)
        out = torch.divide(out,2)
        return(out)
    
class SinglePoint_Discriminator(nn.Module):
    def __init__(self,freq_bins,time_bins, masked, inp_channels):
        super(SinglePoint_Discriminator, self).__init__()
        self.freq_bins = freq_bins
        self.time_bins = time_bins - masked
        self.masked = masked
        self.inp_channels = inp_channels
        self.conv_layer = nn.Sequential(
            nn.Conv2d(self.inp_channels,8,(8,32),stride=2,padding=(2),padding_mode='circular'),
            nn.SELU(),
            nn.Conv2d(8,8,(4,16),stride=2,padding=(2),padding_mode='circular'),
            nn.SELU(),
            nn.Conv2d(8,16,(3,8),stride=1,padding=0,padding_mode='circular'),
            nn.SELU(),
            nn.Conv2d(16,16,(3,2),stride=1,padding=0,padding_mode='circular'),
            nn.Conv2d(16,self.inp_channels,(2,5),stride=1,padding=0,padding_mode='circular'),
            nn.Flatten(start_dim=1),
            )
    def forward(self, input):
        bound_1 = torch.randint(low=0, high=self.masked, size=(1,))
        bound_2 = self.masked - bound_1
        inp = input[:,:,:,bound_1:-bound_2].clone()
        out = self.conv_layer(inp)
        out = torch.sin(out)
        out = torch.add(out,1)
        out = torch.divide(out,2)
        return(out) 
    
class Privacy_Discriminator(nn.Module):
    def __init__(self):
        super(Privacy_Discriminator, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Softmax(dim=2),
            nn.Conv1d(1, 4, 3),            
            nn.SELU(),
            nn.Conv1d(4, 4, 3),   
            nn.Conv1d(4, 4, 3),  
            nn.Conv1d(4, 4, 3, stride=2),            
            nn.SELU(),
            nn.Conv1d(4, 4, 3, stride=2),            
            nn.SELU(),
            nn.Conv1d(4, 1, 3, stride=1),
            )
    def forward(self, gen):
        out = self.conv_layer(gen)
        out = torch.flatten(out, start_dim=1)
        out = torch.sin(out)
        out = torch.add(out,1)
        out = torch.divide(out,2)
        return(out) 
    

    

if __name__=='__main__':  
    freq_bins = 23
    time_bins = 137
    num_filtered=37
    nz = 20
    big_nz = 16
    batches=16
    num_sensors = 4
    gen_conv = Generator(nz,freq_bins,time_bins, num_sensors).to(device)
    singlepoint_disc = SinglePoint_Discriminator(freq_bins,time_bins,num_filtered,num_sensors).to(device)
    standard_disc = Standard_Discriminator(freq_bins,time_bins,num_filtered,num_sensors).to(device)
    priv_disc = Privacy_Discriminator().to(device)
    gen_params_conv = sum(p.numel() for p in gen_conv.parameters())
    disc_params_single = sum(p.numel() for p in singlepoint_disc.parameters())
    disc_params_standard = sum(p.numel() for p in standard_disc.parameters())
    disc_params_priv = sum(p.numel() for p in priv_disc.parameters())
    
    print("parameters for this pair (gen, single, batch, priv)", gen_params_conv, disc_params_single, disc_params_standard, disc_params_priv)
    
    inp = torch.randn(batches,nz)[:,None, None,:].to(device)
    
    inp_conv = torch.randn(batches,big_nz)[:,None, None,:].to(device)
    print("Generator conv input:", inp_conv.shape)
    g_out = gen_conv(inp_conv)
    
    real = torch.rand((g_out.shape)).to(device)
    print("Output from Generator", g_out.shape)    
    d_out_single = singlepoint_disc(g_out)
    print("Discriminator shape singlepoint is", d_out_single.shape)
    d_out_standard = standard_disc(g_out)
    print("Discriminator shape standard is", d_out_standard.shape)
    g_priv = torch.mean(g_out, 3)[:,3,:][:,None,:].clone()
    #print("Privacy shape is:", g_priv.shape)
    d_out_priv = priv_disc(g_priv)
    print("Discriminator shape privacy is", d_out_priv.shape)
    
    
    
