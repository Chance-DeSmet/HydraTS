import torch
import pandas as pd
from new_radar import make_radar_chart
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def area_and_name_from_df(df,name):
    h_auc = get_area(df.iloc[0,1:])    
    df.at[0, 'Name'] =  df.at[0, 'Name']  + name + " AuRC:" +str(h_auc)
    t_auc = get_area(df.iloc[1,1:])
    df.iloc[1,0] =  df.iloc[1,0] + name + " AuRC:" +str(t_auc)
    o_auc = get_area(df.iloc[2,1:])
    df.iloc[2,0] =  df.iloc[2,0] + name + " AuRC:" +str(o_auc)

    return(df)

def area_from_df(df, name):
    h_auc = get_area(df.iloc[0,1:])    
    df.iloc[0,0] =  df.iloc[0,0]  + name + " AuRC:" +str(h_auc)
    t_auc = get_area(df.iloc[1,1:])
    df.iloc[1,0] =  df.iloc[1,0]  + name + " AuRC:" +str(t_auc)
    o_auc = get_area(df.iloc[2,1:])
    df.iloc[2,0] =  df.iloc[2,0]  + name + " AuRC:" +str(o_auc)

    return(df)

def get_area(MCC_pred):
    j = 0
    num = len(MCC_pred)
    auc = 0
    for items in MCC_pred:
        #print("i is:", i)
        if(j == 0):
            auc += 0.5 * MCC_pred[0]*MCC_pred[-1]*np.sin(360.0/num)
        else:
            auc += 0.5 * MCC_pred[j]*MCC_pred[j-1]*np.sin(360.0/num)
        j += 1
    try:
        return(round(float(auc.numpy()), 2))
    except:
        return(round(float(auc), 2))
def call_radar(gen, time, orig, name, catagories, divisors):
    i = 0


    new_time = []
    new_gen = []
    new_orig = []
    for items in orig:
        if(i == 0 or i == 1):
            new_time.append(divisors[i]/time[i])
            new_gen.append(divisors[i]/gen[i])
            new_orig.append(divisors[i]/orig[i])
        else:
            new_time.append(time[i]/divisors[i])
            new_gen.append(gen[i]/divisors[i])
            new_orig.append(orig[i]/divisors[i])
        i += 1

    time_auc = get_area(new_time)
    gen_auc = get_area(new_gen)
    orig_auc = get_area(new_orig)
    df= pd.DataFrame([new_orig,new_time,new_gen], columns=catagories)
    print(df)
    path = "./Hydra_TS_radar_" + name 
    make_radar_chart(df, name, path)



