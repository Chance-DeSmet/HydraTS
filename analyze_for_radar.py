import torch
import torch.nn as nn
import numpy as np
import torchaudio.transforms as T
from tslearn.metrics import dtw, dtw_path
from scipy.stats import wasserstein_distance
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import librosa
from cosine_sim import pull_model, test_similarity
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_perf(orig_tens, gen_tens, comp_list, comp_names, cat, participant, model):

    bootstrap_amount = 16
    orig_tens = orig_tens[torch.randint(orig_tens.shape[0], (bootstrap_amount,))].clone()
    gen_tens = gen_tens[torch.randint(gen_tens.shape[0], (bootstrap_amount,))].clone()
    comp_list[0] = comp_list[0][torch.randint(comp_list[0].shape[0], (bootstrap_amount,))].clone()
    comp_list[1] = comp_list[1][torch.randint(comp_list[1].shape[0], (bootstrap_amount,))].clone()

    over_dict = make_perf_dict()
    sub_dict = over_dict[cat]
    val_list = sub_dict[participant]

    griffin_lim_watch = T.GriffinLim(45)
    init_sense = 0
    while(init_sense < orig_tens.shape[1]):
        if(init_sense == 0):            
            waveform_orig = griffin_lim_watch(librosa.db_to_power(orig_tens[:,init_sense,:,:].detach().cpu()))[:,None,:]
            waveform_gen = griffin_lim_watch(librosa.db_to_power(gen_tens[:,init_sense,:,:].detach().cpu()))[:,None,:]
        else:
            w_temp = griffin_lim_watch(librosa.db_to_power(orig_tens[:,init_sense,:,:].detach().cpu()))[:,None,:]
            waveform_orig = np.concatenate((waveform_orig,w_temp),axis=1)
            g_temp = griffin_lim_watch(librosa.db_to_power(gen_tens[:,init_sense,:,:].detach().cpu()))[:,None,:]
            waveform_gen = np.concatenate((waveform_gen,g_temp),axis=1)
        init_sense += 1
    dtw_val = 0
    i_sens = 0
    while(i_sens < 4):
        i = 0
        while i < waveform_orig.shape[0]:
            j = 0
            while j < waveform_gen.shape[0]:
                dtw_val += dtw(waveform_orig[i,i_sens], waveform_gen[j,i_sens])
                j += 1
            i += 1
        i_sens += 1
    dtw_val = dtw_val / (waveform_gen.shape[0]*waveform_orig.shape[0]*4)
    wass_val = 0
    i_sens = 0
    while(i_sens < 4):
        i = 0
        while i < waveform_orig.shape[0]:
            j = 0
            while j < waveform_gen.shape[0]:
                wass_val += wasserstein_distance(waveform_orig[i, i_sens], waveform_gen[j, i_sens])
                j += 1
            i += 1
        i_sens += 1
    wass_val = wass_val / (waveform_gen.shape[0]*waveform_orig.shape[0]*4)
    bound_val = test_similarity(model, orig_tens, gen_tens, comp_list[0], comp_list[1])

    summed_gen = torch.sum(gen_tens, 3)[:,3,:].clone()
    summed_orig = torch.sum(orig_tens, 3)[:,3,:].clone()
    sm = nn.Softmax(dim=1)

    sm_gen= sm(summed_gen)
    sm_orig = sm(summed_orig)
    mse = nn.MSELoss()
    diff = mse(sm_gen, sm_orig)
    soft_val = torch.mean(diff).detach().cpu().numpy()

    total_val = dtw_val + wass_val + bound_val + soft_val 
    print("Total Val is:", total_val)
    print("Individual Scores: Mean DTW, Mean EM, Cosine Similarity, Privacy (distance) movement")
    print(total_val, ',', dtw_val, ',', wass_val, ',', bound_val, ',', soft_val)
    if(len(val_list) > 0):
        print("Original from Time GAN:")
        print(val_list)
        
        dtw_diff = val_list[1] - dtw_val
        dtw_perc_diff = dtw_diff/(val_list[1] + 1e-4)
        wass_diff = val_list[2] - wass_val
        wass_perc_diff = wass_diff/(val_list[2] + 1e-4)
        bound_diff =  bound_val - val_list[3] 
        b_perc_diff = bound_diff/(val_list[3] +1e-4)
        priv_diff = soft_val - val_list[4]
        priv_perc_diff = priv_diff/(val_list[4] + 1e-4)
        tot = 0.1*dtw_diff + wass_diff + bound_diff + priv_diff
        out_list = [dtw_diff, wass_diff, bound_diff, priv_diff]
        out_list = [-10 if val < 0 else val for val in out_list]
        if(tot < -10):
            tot = -10 
        return(tot, out_list)
    else:
        return(total_val, [dtw_val, wass_val, bound_diff, soft_val])


    
