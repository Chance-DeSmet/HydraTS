# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:10:16 2022

@author: sl1mc
"""
from hydraTS_training import run_through
import sys
import torch


# total arguments
n = len(sys.argv)
print("Total arguments passed:", n)
 
# Arguments passed
print("\nName of Python script:", sys.argv[0])



if(__name__ == '__main__'):

    
    participant = 'grouped'

    data_errands = torch.rand((20,4,23,137))
    torch.save(data_errands, "./data_folder/errands/"+participant+"/errands_train.pt")
    torch.save(data_errands, "./data_folder/errands/"+participant+"/errands_val.pt")
    data_work = torch.rand((20,4,23,137))
    torch.save(data_work, "./data_folder/work/"+participant+"/work_train.pt")
    torch.save(data_work, "./data_folder/work/"+participant+"/work_val.pt")
    data_exercise = torch.rand((20,4,23,137))
    torch.save(data_work, "./data_folder/exercise/"+participant+"/exercise_train.pt")
    torch.save(data_work, "./data_folder/exercise/"+participant+"/exercise_val.pt")

    data_names = 'errands'
    epochs = 55000
    gen_lr = 0.00005
    disc_lr = 0.00001
    interval= 100
    load= 1
    dat_path = './data_folder/errands/'
    weight_path = './data_folder/errands/errands_gen.pt'
    print("Running with following conditions:", data_names, epochs, gen_lr, disc_lr, load, dat_path, weight_path, participant)
    run_through(data_names, epochs, gen_lr, disc_lr,interval, load=load, dat_path=dat_path, weight_path=weight_path, participant=participant)
