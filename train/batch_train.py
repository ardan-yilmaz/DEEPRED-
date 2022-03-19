from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy as np

import torch
from torch.autograd import Variable
import numpy as np
import math
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import math
from csv import reader
import csv
import matplotlib.pyplot as plt


from classes import *
from thresholds import find_all_thresholds
#from classes import test_dataset



#df = pd.read_csv("level_6_after_filtering_stats_more_than_50.csv")
#annot_nums = df[" NUMBER_OF_ANNOTS "].to_list()

import subprocess
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import pandas as pd

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def print_scores(TN, FP, FN, TP, epoch, batch_size):

      #FINAL SCORES:
      print("SCORES:::::")
      print("total TN: ", TN)
      final = ""
      final += "total TN: " + str(TN) +"\n"

      print("total FP: ", FP)
      final += "total FP: " + str(FP) +"\n"

      print("total FN: ", FN)
      final += "total FN: " + str(FN) +"\n"


      print("total TP: ", TP)
      final += "total TP: " + str(TP) +"\n"

      total_pred = TP + FP + FN + TN
      print("total pred: ", total_pred)
      final += "total pred: " + str(total_pred) +"\n"

      acc = (TP + TN)/(total_pred)
      print("accuracy: ", acc)
      final += "accuracy: " + str(acc) +"\n"


      recall = TP / (TP + FN)
      print("recall: ", recall)
      final += "recall: " + str(recall) +"\n"

      prec = TP/(TP+FP)
      print("precision: ", prec)
      final += "precision: " + str(prec) +"\n"

      f1 = 2*recall*prec / (recall+prec)
      print("f1-score: ", f1)
      final += "f1-score: " + str(f1) +"\n"


      MCC = 0
      try:
        #calcualting the multiplication first then sqrt causes floating pnt overflow although more efficient :(
        MCC = (TP*TN - FP*FN) / (math.sqrt(TP+FP)* math.sqrt(TP+FN)* math.sqrt(TN+FP)* math.sqrt(TN+FN))
      except: pass

      print("MCC: ", MCC)
      final += "MCC: " + str(MCC) 

      

      ################## WRITE THE RESULTING SCORES TO A FILE            
      final_score = os.path.join(out_path, "scores_on_" +str(epoch) + "_epochs_"+ str(batch_size)+ "batches"+ ".txt")
      with open(final_score, "a+") as out_s:
        out_s.write(final)  

      ################## WRITE THE TERM BASED PERFORMANCE TO A FILE 
      df = None
      term_based_perf_path = os.path.join(out_path, "term_based_on_"+str(epoch) + "_epochs_"+ str(batch_size)+ "batches"+ ".csv")
      df = pd.DataFrame(class_based_perfs[1:], columns = class_based_perfs[0])   
      df = df.sort_values(by=['annot num'])  
      df.to_csv(term_based_perf_path, index = False)
      


def class_weights():
  go_terms = list()
  f = open(go_names_and_lengths_path, 'r')
  lines = f.readlines()
  lll = len(lines)
  ctr  =0
  for i in range(0, lll):
    if i == 0: continue
    if ctr == num_GO_Terms: break
    ll = (lines[i]).split(',')
    a1 = ll[0]
    a2  = ll[1]
    go_terms.append([a1, a2])
    ctr +=1

  

  total_num_of_proteins=0
  for i in range(0,len(go_terms)):
    total_num_of_proteins+= int(go_terms[i][1])

  print("total_num_of_proteins=",total_num_of_proteins)

  number_of_proteins_in_each_class=[]

  weights=[]
  weights_log2=[]
  weights_log10 = []
  for i in range(0,len(go_terms)):
    number_of_proteins_in_each_class.append(int(go_terms[i][1]))

  for i in range(0,len(go_terms)):
    weights_log10.append( math.log(  number_of_proteins_in_each_class[i]  ,10))
    weights_log2.append( math.log2( number_of_proteins_in_each_class[i] ) )

  print("go_temrs=", go_terms[0:5])
  print("total_num_of_proteins =", total_num_of_proteins)
  print("weights = ", weights)


  weights=number_of_proteins_in_each_class
  weights = torch.tensor(weights, dtype=torch.float32)
  weights_log2 = torch.tensor(weights_log2, dtype=torch.float32)
  weights_log10 = torch.tensor(weights_log10, dtype=torch.float32)


  weights_log2 = weights_log2 / weights_log2.sum()
  return weights_log2
  


def train(batch_size):
  
  global device
  train_losses = []  
  
  #dataloaders
  train_set = DEEPRedDataset(train_data_path, train_labels_path)
  train_set_loader = DataLoader(train_set, batch_size=batch_size, shuffle =True)

  model = torch.nn.Sequential(

    torch.nn.Linear(400, 1024),
    torch.nn.BatchNorm1d(num_features=1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(DropOut_param),

    torch.nn.Linear(1024, 512),
    torch.nn.BatchNorm1d(num_features=512),
    torch.nn.ReLU(),
    torch.nn.Dropout(DropOut_param), 


    torch.nn.Linear(512, num_GO_Terms),
    torch.nn.Sigmoid()    

  ).to(device)

  print(model)
  num_epoch =250
  print("number of epochs ===== ", num_epoch)
  loss_function = torch.nn.BCELoss(weights_log2)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



  #optimizer.zero_grad()
  for epoch in range(num_epoch):    
    print("Epoch :{}".format(epoch))
    accum_train_loss = 0

    #TRAINING
    model.train()
    print("Training:", model.training)    

    #read the batches
    for i, (data, label) in enumerate(train_set_loader):
      #data, labels = data.to(device), label.to(device)
       
      input_train, target = data.to(device), label.to(device)
      #forward
      out = model(input_train.float()).to(device)
      #print(out)
      loss = loss_function(out, target.float())
      #total_training_loss += float(loss.item())
      #backward
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      accum_train_loss += loss.item()
      # print("Epoch {} training loss:".format(epoch), total_training_loss)
      #print("Epoch {} training loss:".format(epoch), loss.item())   
    
    print(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i}') 
    train_losses.append(accum_train_loss / i)  
    
    

    if (epoch+1) % 50==0:   

      ### SAVE THE MODEL HERE ###

      model.eval()   
      with torch.no_grad():

        ### SAVE THE MODEL
        saved_path = str(level) +"_" +str(epoch)+"_epochs_"+str(batch_size)+".pt"
        saved_path = os.path.join(out_path, saved_path)
        torch.save(model.state_dict(), saved_path)           

        test_data =pd.read_csv(test_data_path, header=None)
        sc = StandardScaler()
        test_data = sc.fit_transform(test_data.values)
        test_data = torch.tensor(test_data).to(device) 

        test_labels = torch.from_numpy(pd.read_csv(test_labels_path, header=None).to_numpy())


        test_preds = model(test_data.float())
        test_preds = test_preds.cpu().detach().numpy()

        num_test_prots, num_of_classes = test_preds.shape
        class_based_perfs = []
        TN, FP, FN, TP, optimal_thresholds = find_all_thresholds(test_preds, test_labels, num_test_prots, num_of_classes, class_based_perfs)

        ## save the train loss img
        plt.plot(train_losses) 
        loss_img_path = os.path.join(out_path, "loss_on_"+str(epoch)+"_epochs_"+str(batch_size)+"_batches.png")
        plt.savefig(loss_img_path)

        ####SAVE THE DETERMINED THRESHOLD VALS####
        import csv
        print("THRESHOLDSS:: \n", optimal_thresholds)
        thr_path = os.path.join(out_path, "thresholds_on_"+str(epoch)+"_epochs_"+str(batch_size)+"_batches.csv")
        with open(thr_path, "a+") as thr_f:
          write = csv.writer(thr_f)
          write.writerow(optimal_thresholds) 
               

        print_scores(TN, FP, FN, TP, epoch, batch_size)





#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
###################################                  MAIN        ################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################





if __name__ == "__main__":
  ################
  # CHANGE THESE #
  ################
  data_path = "dataset/model2"
  out_path = "results_batches"
  num_GO_Terms=57
  level = 2
  print("LEVEL ===== ", level)
  #hyperparams
  batch_sizes = [64, 256]
  #batch_size = 64
  learning_rate = 0.001
  DropOut_param = 0.2 
  ################
  ################
  ################

  

  cur_dir = os.path.join(os.getcwd(), data_path)
  go_names_and_lengths_path= os.path.join(cur_dir, "annots.csv")

  device = get_free_gpu()
  #device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print("device::: ", device)	  

  ##FIND THE CLASS WEIGHTS
  weights_log2 = class_weights()
  weights_log2 = weights_log2.to(device)

  out_path = os.path.join(cur_dir, out_path)
  if not os.path.exists(out_path):
    os.mkdir(out_path)
  

  #dataset paths
  train_data_path = os.path.join(cur_dir,"train_data.csv")
  train_labels_path = os.path.join(cur_dir,"train_labels.csv")
  test_data_path = os.path.join(cur_dir,"test_data.csv")  
  test_labels_path =  os.path.join(cur_dir,"test_labels.csv") 

  for batch in batch_sizes:
    ######## class based performance results ########
    class_based_perfs = list()
    class_based_perfs.append(["annot num", "TN", "FP", "FN", "TP", "acc", "recall", "prec", "f1", "MCC"]) 
    train(batch)  

  """
  class_based_perfs = list()
  class_based_perfs.append(["annot num", "TN", "FP", "FN", "TP", "acc", "recall", "prec", "f1", "MCC"]) 
  train(batch_size)

 
  """

  
 
 



