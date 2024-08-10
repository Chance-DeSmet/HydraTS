# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:53:10 2022

@author: Chance
"""

#import tracemalloc 
import gc

import torch
import torch.nn as nn
import os
from hydra_utilities import import_data, create_noise, create_noise_shaped
from discriminators import AccuracyDiscriminatorClass, AvoidanceDiscriminatorClass
from discriminators import SinglePointAccuracyDiscriminatorClass, PrivacyDiscriminatorClass
from analyze_for_radar import get_perf
from networks import Generator
import pandas as pd
import matplotlib.pyplot as plt
from process_generations import generate_report_pred
import datetime
torch.autograd.set_detect_anomaly(True)
import numpy as np
from cosine_sim import pull_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device, torch.cuda.is_available())

def attribute_dict():
    '''
    dictionary index:
        0 -> sensitive attribute -- int
        1 -> classification attribute -- int
        2 -> diversity attribute -- int
        3 -> diversity weights used for comparisons -- list 
    '''
    att_dict = {"heart":[0,1,13,[1,1]], "cervical_cancer":[0,1,33,[1,1]], "health_insurance":[1,14,19,[1,1]], "electric_grid":[4,0,13,[1,1]],
                "smarthome_wsu":[9,12,1,[1,1]], "iris":[0,1,4,[1,1]], "yes_spec":[0,1,4,[1,1]], "yes_1":[0,1,4,[1,1]], "yes_w":[0,1,4,[1,1]],
                "exercise":[0,1,4,[1,1]],"relax":[0,1,4,[1,1]],"housework":[0,1,4,[1,1]],"errands":[0,1,4,[1,1]],"hobby":[0,1,4,[1,1]],"hygiene":[0,1,4,[1,1]]}
    
    return(att_dict)

def make_selections(num_els):
    all_lists = []
    i = 0 
    full = []
    while i < num_els:
        full.append(i)
        i += 1
    all_lists.append(full)
    j = 1
    while j < num_els:
        new_missing = full.copy()
        new_missing.remove(j)
        all_lists.append(new_missing)
        j += 1
    k = 1
    while k < num_els:
        new_duo = [0, k]
        all_lists.append(new_duo)
        k += 1
    return(all_lists)

def run_through(data_list, num_epochs, gen_lr, disc_lr,sample_interval, load=0, dat_path=None, weight_path=None, participant='sttr001'):
  
    att_dict = attribute_dict()
    i = 0
    epochs = num_epochs
    nz = 16

    batch_len = 23 #Freq bins
    batch_width = 137 #Time bins


    masked = 37 #randomly mask out portions (in this case to get 100 time bins)
    inp_channels = 4 #how many features collected on

    compliment_list = ['exercise', 'errands', 'work']
    compliment_list.remove(data_list)

    num_batches = 8
    lr = gen_lr
    disc_lr = disc_lr
    correcting = 0
    #dat_samp = import_data(batch_len, data_list[i])
    data_name = data_list
    data_list = dat_path + participant + '/' 
    print("new loading path is:",data_list)
    '''
    call generator functions
    '''
    #print("Att is", att_dict[data_name][0])
    if(participant != 'grouped'):
        gold_path = "./saved_weights/gold/" + data_name + '_pretrained_gen.pt' #Create path to load from saved weights
    else:
        gold_path = "./saved_weights/gold/" + participant + '_' + data_name + '_pretrained_gen.pt'
    weight_path = "./saved_weights/bootstrapped/" + participant + '_' + data_name + '_pretrained_gen.pt'
    if(load == 0):
        gen = Generator(nz, batch_len, batch_width,inp_channels).to(device)
    else:
        gen = Generator(nz, batch_len, batch_width,inp_channels).to(device)
        try:
            gen.load_state_dict(torch.load(gold_path))
            print("GOLDEN Generator weights successfully loaded")
        except FileNotFoundError:
            print("No golden weights found, trying initial bootstrapped...")
            try:
                gen.load_state_dict(torch.load(weight_path))
                print("Generator weights successfully loaded")
            except FileNotFoundError:
                print(f"Error: Weights file '{weight_path}' not found.")
            except Exception as e:
                print(f"Error loading weights: {e}")
    gen_optim = torch.optim.SGD(gen.parameters(), lr=lr)
    gen_scheduler = 0
    gen_loss = torch.nn.MSELoss()
    '''
    make_discs
    '''
    list_of_nets = init_nets(num_batches, batch_len, batch_width, data_list, disc_lr, nz, att_dict, data_name, masked, inp_channels, participant)
    
    data_sample = train_nets(epochs, gen, gen_optim, gen_loss, list_of_nets, nz, data_list, gen_scheduler,batch_len,batch_width,sample_interval,att_dict['yes_w'],load,data_name,participant,weight_path,compliment_list)
    
    
    process_output(data_sample, data_list[i]) 
        
def init_nets(num_batches, batch_len, features, data, disc_lr, nz, att_dic, dat_name, masked, inp_channels, participant):
    '''
    First make a list of every disc we are using
    '''
    avoidance_targets = ['./data_folder/exercise/'+participant+'/', './data_folder/errands/'+participant+'/', './data_folder/work/'+participant+'/']
    ret_list = []
    
    
    Acc_disc = AccuracyDiscriminatorClass(num_batches, batch_len, features, data, nz, lr=disc_lr, masked=masked, inp_channels=inp_channels)
    acc_path = data+"_acc_disc.pt"
    
    try:
        Acc_disc.disc.load_state_dict(torch.load(acc_path))
        print("General Discriminator weights successfully loaded")
    except FileNotFoundError:
        print(f"Error: Weights file '{acc_path}' not found.")
    except Exception as e:
        print(f"Error loading weights: {e}")
    
    ret_list.append(Acc_disc)

    Single_Acc_disc = SinglePointAccuracyDiscriminatorClass(num_batches, batch_len, features, data, nz, lr=disc_lr, masked=masked, inp_channels=inp_channels)#.to(device)
    point_path = data+"_sing_disc.pt"
    
    try:
        Single_Acc_disc.disc.load_state_dict(torch.load(point_path))
        print("Channel discriminator weights successfully loaded")
    except FileNotFoundError:
        print(f"Error: Weights file '{point_path}' not found.")
    except Exception as e:
        print(f"Error loading weights: {e}")
    
    ret_list.append(Single_Acc_disc)
    i = 0
    av_list = ['_av_1_disc.pt', '_av_2_disc.pt', '_av_3_disc.pt', '_av_4_disc.pt', '_av_5_disc.pt', '_av_6_disc.pt', '_av_7_disc.pt']
    for items in avoidance_targets:
        if(data != avoidance_targets[i]):
            print("Avoiding data from "+ avoidance_targets[i], flush=True)
            av_disc = AvoidanceDiscriminatorClass(num_batches, batch_len, features, avoidance_targets[i], nz, lr=disc_lr, masked=masked, inp_channels=inp_channels)#.to(device)
            av_path = data + av_list[i]
            try:
                av_disc.disc.load_state_dict(torch.load(av_path))
                print("Weights successfully loaded from boundary " +str(i))
            except FileNotFoundError:
                print(f"Error: Weights file '{av_path}' not found.")
            except Exception as e:
                print(f"Error loading weights: {e}")
            ret_list.append(av_disc)
        else:
            print("Not avoiding data from "+ avoidance_targets[i]) 
        i += 1
    
    priv_disc = PrivacyDiscriminatorClass(num_batches, batch_len, features, data, nz, lr=disc_lr, masked=masked, inp_channels=inp_channels)
    ret_list.append(priv_disc)
    return(ret_list)
    

def train_single(disc_element, number,gen, gen_opt, gen_loss,not_collapsed,gen_collapsed):
    i = 0
    while i < number:
        disc_element.train(gen, gen_opt, gen_loss,not_collapsed,gen_collapsed,reps=1)
        i += 1

def train_nets(epochs, gen, gen_opt, gen_loss, list_of_nets, nz, dat_source, gen_scheduler,batch_len,features,interval,att_dict,load, dat_name, participant, save_path,compliment_list):
    '''
    TODO: 
        Fix multi vs single spec runs - default to finding what ever is in folder
    '''

    validation_spec = torch.load("./data_folder/" + dat_name + '/' + participant + '/' + dat_name + "_val.pt").to(device)
    comp_1_spec = torch.load("./data_folder/" + compliment_list[0] + '/' + participant + '/' + compliment_list[0] + "_val.pt").to(device)
    comp_2_spec = torch.load("./data_folder/" + compliment_list[1] + '/' + participant + '/' + compliment_list[1] + "_val.pt").to(device)
    perf_model = pull_model().to(device)
    generative_combined_loss = []
    channel_vals =  ['yaw', 'pitch', 'roll', 'distance']
    i = 0
    not_collapsed=1
    gen_collapsed=0
    EM_chart_list = []
    curr_EM = 1
    gen_learning_rate_chart = []
    disc_learning_rate_chart = []
    mean_dist_chart = []
    all_disc_to_gen_losses = []
    all_disc_names = []
    name_tracker = 0
    post_train = 0
    start_time = datetime.datetime.now()
    for names in list_of_nets:
        new_list = []
        all_disc_to_gen_losses.append(new_list)
        all_disc_names.append(list_of_nets[name_tracker].name[1:])
        name_tracker += 1
    mean_losses = [0]*len(list_of_nets)
    test_i = 0  #here is where we generate the intial avg F1 for the data without augmentation
    test_f1s = []
    while(test_i < 5):
        test_f1s.append(generate_report_pred("None_Testing_Mode", participant))
        test_i += 1
    print("List of test scores are:", test_f1s)
    orig_f1 = np.mean(test_f1s) 
    f1_std = np.std(test_f1s)   
    gen_attempts = 0
    while i < (epochs + post_train):
        j = 0        
        gen_opt.zero_grad()
        comb_losses = 0
        for items in list_of_nets:
            curr_loss = list_of_nets[j].train(gen, gen_opt, gen_loss,not_collapsed,gen_collapsed,mean_losses[j],reps=1)
            if(curr_loss > 0.9 and j == 0 and i > 2000 and gen_attempts < 2000):
                not_collapsed = 0
                gen_opt.step()
                gen_opt.zero_grad()
                gen_attempts += 1
                continue
            else:
                not_collapsed = 1
            all_disc_to_gen_losses[j].append(curr_loss)
            
            comb_losses += curr_loss
        
            j += 1
        gen_opt.step()
        i += 1
        generative_combined_loss.append(comb_losses)
        if(i % 500 == 0):
            print("Done with step: " + str(i), flush=True) #I just want a heartbeat to make sure training progress
            end_time = datetime.datetime.now()
            runtime = end_time - start_time
            print(f"Runtime: {runtime}", flush=True)
        if((i % interval == 0 or i == 1) and not_collapsed == 1):
            
            print("Tracking at epoch: ", i, " Number of Generator attempts:", gen_attempts, flush=True)
            gen_attempts = 0
            num_specs = validation_spec.shape[0] #gen as many as there are validation specs
            noise = create_noise_shaped(num_specs,batch_len, features, nz)
            
            nout = gen(noise)
            if(i == 1):
                print("We are generating " + str(num_specs) + " spectrograms")
                print("Gen outut shape is ", str(nout.shape))
            
            torch.save(nout, dat_source+"aug_analysis.pt")
            if(i < 10):
                MYDIR = ("./saved_output/" + str(dat_name))
                CHECK_FOLDER = os.path.isdir(MYDIR)


                if not CHECK_FOLDER:
                    os.makedirs(MYDIR)
                    print("created folder : ", MYDIR)

                else:
                    print(MYDIR, "folder already exists.")

                MYDIR = ("./reports/" + str(dat_name))
                CHECK_FOLDER = os.path.isdir(MYDIR)


                if not CHECK_FOLDER:
                    os.makedirs(MYDIR)
                    print("created folder : ", MYDIR)

                else:
                    print(MYDIR, "folder already exists.")

            F1_pred = generate_report_pred(dat_name, participant)  
            if(F1_pred > 0.45): #If we are higher than terrible, record it
                dat_loc = "./saved_output/"+str(dat_name)+"/generator_out_"+dat_name+'_'+str(i)+"_f1_"+str(F1_pred)+'_load_status_'+str(load)+".pt"
                torch.save(nout, dat_loc)   
                torch.save(gen.state_dict(), "./saved_weights/bootstrapped/"+ participant + '_' + dat_name + "_pretrained_gen.pt")
            if(F1_pred > 0.55):
                print("&&&&&&&&&&&&&&&&&&&&&&&&")
                print("@@@@@@@@@@@@@@@@@@@@@@@@")
                print("&&&&&&&&&&&&&&&&&&&&&&&&")
                print("GOOD RESULT!!!")
                print("&&&&&&&&&&&&&&&&&&&&&&&&")
                print("@@@@@@@@@@@@@@@@@@@@@@@@")
                print("&&&&&&&&&&&&&&&&&&&&&&&&")
                dat_loc_good = "./saved_output/"+str(dat_name)+"/generator_BEAT_"+dat_name+'_'+str(i)+"_"+participant+'_'+str(F1_pred)+'_load_status_'+str(load)+".pt"
                torch.save(nout, dat_loc_good)   
                torch.save(gen.state_dict(), dat_source + "_" + str(F1_pred) + "_" + str(datetime.datetime.now()) + "_BEAT_gen.pt")
                torch.save(list_of_nets[0].disc.state_dict(), dat_source+"BEAT_acc_disc.pt")
                torch.save(list_of_nets[1].disc.state_dict(), dat_source+"BEAT_sing_disc.pt") 
            print("########################")
            print("########################")
            print("########################")
            print("Eval score at " +str(i)+ " epochs: " + str(F1_pred))
            print("########################")
            print("########################")
            print("########################")
            EM_chart_list.append(F1_pred)
            mean_dist_chart.append(F1_pred)
            curr_g_loss = comb_losses
            curr_real_d_loss = list_of_nets[0].last_disc_loss
            print("Generative Loss is:", curr_g_loss)            
            print("Discriminative Loss is:", curr_real_d_loss)
            plt.clf()
            plt.plot(EM_chart_list, label='Eval Score with augmented data')
            plt.hlines(y=orig_f1, xmin=0, xmax=len(EM_chart_list)-1, colors='aqua', linestyles='--', lw=2, label='Mean score of non-augmented data')
            plt.hlines(y=f1_std+orig_f1, xmin=0, xmax=len(EM_chart_list)-1, colors='salmon', linestyles='--', lw=2, label='1 Standard deviation above the mean')
            plt.legend()
            plt.title("Observed Eval Distance through training")
            plt.savefig("./plots/updating/"+dat_name+"/Eval_score_"+participant+'_load_status_'+str(load)+"_tracker_updating.png")

            plt.clf()
            plt.plot(torch.tensor(generative_combined_loss).detach().cpu().numpy())
            plt.title("Total loss to generator")
            plt.ylim(0,7)
            plt.savefig("./plots/updating/"+dat_name+"/gen_loss_"+participant+'_load_status_'+str(load)+"_tracker_updating.png")

            plt.clf()
            plt.plot([tensor.detach().cpu().numpy() for tensor in list_of_nets[0].own_loss_chart])
            plt.title("First disc, distribution, loss")
            plt.savefig("./plots/updating/"+dat_name+"/disc_0"+participant+'_load_status_'+str(load)+"_tracker.png")

            plt.clf()
            plt.plot([tensor.detach().cpu().numpy() for tensor in list_of_nets[1].own_loss_chart])
            plt.title("Second disc, singlepoint loss")
            plt.savefig("./plots/updating/"+dat_name+"/disc_1"+participant+'_load_status_'+str(load)+"_tracker.png")

            plt.clf()
            plt.plot([tensor.detach().cpu().numpy() for tensor in list_of_nets[2].own_loss_chart])
            plt.title(compliment_list[0] + " avoidance loss")
            plt.savefig("./plots/updating/"+dat_name+"/disc_2"+participant+'_avoiding_'+compliment_list[0]+'_load_status_'+str(load)+"_tracker.png")

            plt.clf()
            plt.plot([tensor.detach().cpu().numpy() for tensor in list_of_nets[3].own_loss_chart])
            plt.title(compliment_list[1] + " avoidance loss")
            plt.savefig("./plots/updating/"+dat_name+"/disc_3"+participant+'_avoiding_'+compliment_list[1]+'_load_status_'+str(load)+"_tracker.png")

            plt.clf()
            plot_it = 0
            for lines in all_disc_to_gen_losses:
                plt.plot([tensor.detach().cpu().numpy() for tensor in all_disc_to_gen_losses[plot_it]], label=all_disc_names[plot_it])
                plt.legend()
                plot_it += 1            
            plt.title("Individual losses seen by generator")
            plt.ylim(-0.1,1.75)
            plt.savefig("./plots/updating/"+dat_name+"/indiv_gen_loss_"+participant+'_load_status_'+str(load)+"_tracker_updating.png")

        
            nout_save = nout[0,:,:,:].clone()
            nout_i = 0
            while nout_i < nout_save.shape[0]:
                plt.clf()
                fig, axs = plt.subplots(1, 1)
                axs.set_title("Visualization of recent synthetic spectorgram (db), " + channel_vals[nout_i])
                axs.set_ylabel("Frequency bins")
                axs.set_xlabel("Time bins")
                im = axs.imshow(nout_save[nout_i,:,:].detach().cpu().numpy(), origin="lower", aspect="auto")
                fig.colorbar(im, ax=axs)
                plt.show(block=False)
                plt.savefig("./plots/updating/"+dat_name+"/synthetic_"+participant+'_load_status_'+channel_vals[nout_i]+"_tracker.png")
                nout_i += 1
            if(i >  1000):
                torch.save(gen.state_dict(), save_path)
                torch.save(list_of_nets[0].disc.state_dict(), dat_source+"_acc_disc.pt")
                torch.save(list_of_nets[1].disc.state_dict(), dat_source+"_sing_disc.pt")
            
            
            plt.clf()
            plt.close(fig='all')
            
            collected = gc.collect()

            print("Garbage collector: collected",
                "%d objects." % collected)
            '''
            print("Beginning memory trace:")
            snapshot = tracemalloc.take_snapshot() 
            top_stats = snapshot.statistics('lineno')             
            for stat in top_stats[:10]: 
                print(stat)
            print("Done with memory trace", flush=True)
            '''
        
            
            
            
            
            
            
    plt.clf()
    plt.plot(torch.tensor(generative_combined_loss).detach().cpu().numpy())
    plt.title("Total loss to generator")
    plt.savefig("./plots/static/"+dat_name+"/gen_loss_"+participant+str(datetime.datetime.now())+'_load_status_'+str(load)+"_tracker.png")
    
    plt.clf()
    plt.plot(EM_chart_list)
    plt.title("Observed EM Distance through training")
    plt.savefig("./plots/static/"+dat_name+"/Val_tracker_"+'_load_status_'+str(load)+"_tracker.png")

    plt.clf()
    plt.plot([tensor.detach().cpu().numpy() for tensor in list_of_nets[0].own_loss_chart])
    plt.title("First discriminator, overall accuracy loss")
    plt.savefig("./plots/static/"+dat_name+"/disc_0_"+'_load_status_'+str(load)+"_tracker.png")

    plt.clf()
    plt.plot([tensor.detach().cpu().numpy() for tensor in list_of_nets[1].own_loss_chart])
    plt.title("Second discriminator, individual channel accuracy loss")
    plt.savefig("./plots/static/"+dat_name+"/disc_1_"+'_load_status_'+str(load)+"_tracker.png")

    plt.clf()
    plt.plot([tensor.detach().cpu().numpy() for tensor in list_of_nets[2].own_loss_chart])
    plt.title(compliment_list[0] + "discriminator, avoidance loss")
    plt.savefig("./plots/static/"+dat_name+"/disc_"+participant+compliment_list[0]+'_'+'_load_status_'+str(load)+"_tracker.png")

    plt.clf()
    plt.plot([tensor.detach().cpu().numpy() for tensor in list_of_nets[3].own_loss_chart])
    plt.title(compliment_list[1] + "discriminator, avoidance loss")
    plt.savefig("./plots/static/"+dat_name+"/disc_"+participant+compliment_list[1]+'_'+'_load_status_'+str(load)+"_tracker.png")
    
    plt.close(fig='all')
    
    return(F1_pred)

    
    
            
        
    
    

