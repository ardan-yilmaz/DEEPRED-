import pandas as pd
import numpy as np
import os
import torch
from torch.autograd import Variable
import csv

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

from classes import train_dataset
from classes import test_dataset


#returns the true labels per given class
def true_labels_for_class(term, y_test):
  true_labels = list()    
  for x in y_test:
      true_labels.append(int(x[term]))
      
  return true_labels 


#returns initial prediction scores per given class
def init_pred_per_class(term, y_pred):
  terms = list()    
  for x in y_pred:
      terms.append(x[term])
      
  return terms





#calculates f1 score for the given class with the given threshold val
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
def score_per_class(term, threshold, true_labels, num_of_prots, y_pred):     

  preds = init_pred_per_class(term, y_pred) 
  #print(preds)   
  for i in range(0, num_of_prots):
      #print("iiiiiiii: ",i)
      if preds[i] > threshold:
          preds[i] = 1
      else:
          preds[i] = 0   
          
  

  #print("LEN PREDS: ", len(preds))
  #print("LEN true_labels: ", len(true_labels))
      

  F1_score = f1_score(true_labels, preds, average="binary")   
  m_score = matthews_corrcoef(true_labels, preds) 


  return F1_score, m_score, preds





#to find the threshold values for the given class:term using f1-score
from sklearn.metrics import confusion_matrix

def treshold_per_class_f1_score(term, y_test, num_of_prots, y_pred):   
  
  threshold = 0
  max_f1_score = -1
  corresponding_m_score = -1
  optimal_threshold = 1
  true_labels = true_labels_for_class(term, y_test)    
  preds_opt = np.array(num_of_prots) 
  
  while(threshold < 0.25):        
      f1_score, m_score, preds = score_per_class(term, threshold, true_labels, num_of_prots, y_pred)  
      if f1_score > max_f1_score:    
          max_f1_score = f1_score
          corresponding_m_score = m_score
          optimal_threshold = threshold   
          preds_opt = preds
          
      threshold += 0.01 

  if max_f1_score == 0:
    threshold = 1
    f1_score, m_score, preds_opt = score_per_class(term, threshold, true_labels, num_of_prots, y_pred)

  #print("true_labels: ", true_labels)
  #print("preds_opt ", preds_opt) 
  cf_m = confusion_matrix(true_labels, preds_opt, labels=[0, 1])
  print("FOR TERM ", term, "threshold: ", optimal_threshold, "w/ score: ", max_f1_score)
  print("conf matrix for class")
  print(cf_m)   
  print()
  TN, FP, FN, TP = cf_m.ravel()

  #print("unraveling cnfmtrx: ", TN, FP, FN, TP)

  return TN, FP, FN, TP, optimal_threshold




def treshold_per_class_matth(term, y_test, num_of_prots, y_pred):   
  
  threshold = 0
  corr_f1_score = -1
  max_m_score = -1
  optimal_threshold = 1
  true_labels = true_labels_for_class(term, y_test)    
  preds_opt = np.array(num_of_prots) 
  
  while(threshold < 0.25):        
      f1_score, m_score, preds = score_per_class(term, threshold, true_labels, num_of_prots, y_pred)  
      if m_score > max_m_score :    
          max_m_score = m_score
          corr_f1_score = f1_score
          optimal_threshold = threshold   
          preds_opt = preds
          
      threshold += 0.01 


  cf_m = confusion_matrix(true_labels, preds_opt, labels=[0, 1])
  print("FOR TERM ", term, "threshold: ", optimal_threshold, "w/ score: ", corr_f1_score, " and matthews_corrcoef: ", max_m_score)
  print("conf matrix for class")
  print(cf_m)   
  print()
  TN, FP, FN, TP = cf_m.ravel()

  #print("unraveling cnfmtrx: ", TN, FP, FN, TP)

  return TN, FP, FN, TP, optimal_threshold








def find_all_thresholds(y_pred, y_test, num_of_prots, num_of_classes):
  TN = 0
  FP = 0
  FN = 0
  TP = 0
  optimal_thresholds = []
  for i in range(0, num_of_classes):
      tn, fp, fn, tp, thrshld = treshold_per_class_matth(i, y_test, num_of_prots, y_pred)
      #print("found conf matrix entries: ", tn, fp, fn, tp)
      TN += tn
      FP += fp
      FN += fn
      TP += tp
      optimal_thresholds.append(thrshld)
      #print("after addition: ", TN, FP, FN, TP)

  return TN, FP, FN, TP, optimal_thresholds




if __name__ == "__main__":

	num_GO_Terms = 151
	all_preds = dict()
	level = 2

	test_path = os.path.join(os.getcwd(), "new_data")
	test_data_path = os.path.join(test_path, str(level)+"_test_data")
	test_labels_path = os.path.join(test_path, str(level)+"_test_labels")


	model_path = os.path.join(test_path, str(level)+".pt")


	test_data_set = test_dataset(test_data_path, test_labels_path, level)	 
	num_test_files = len(os.listdir(test_data_path))

	use_gpu = torch.cuda.is_available()
	def get_device():
	    device = "cpu"
	    if use_gpu:
	        print("GPU is available on this device!")
	        device = "cuda"
	    else:
	        print("only CPU is available on this device :(")
	    return device
	device = get_device()
	if torch.cuda.is_available():
	    map_location=lambda storage, loc: storage.cuda()
	else:
	    map_location='cpu'





	model2 = torch.nn.Sequential(

	    torch.nn.Linear(20000, 1000),
	    torch.nn.BatchNorm1d(num_features=1000),
	    torch.nn.ReLU(),
	    torch.nn.Dropout(0.2),
	    torch.nn.Linear(1000, num_GO_Terms),
	    torch.nn.Softmax(1)    

	  ).to(device)

	y_test_all = test_labels_path + "/"+ str(level) +"_test_labels_whole.csv" 
	y_test_all =pd.read_csv(y_test_all, header=None) 
	y_test_all = torch.tensor(y_test_all.values)
	print("y test labels all read w/ shape: ", y_test_all.shape)	


	##LOAD THE MODEL
	
	model2.load_state_dict(torch.load(model_path,map_location=map_location))
	print("model loaded")
	model2.eval()

	print("number of test files: ", num_test_files)
	first = 1

	for i in range(0, num_test_files):

		x_test, y_test = test_data_set.__getitem__(i)
		print(" ", i, " th test file loaded")

		input = Variable(x_test).to(device)

		y_pred = (model2(input.float())).to(device)
		print(" ", i, " th test file predicted")

		if not torch.is_tensor(y_test): y_test = torch.from_numpy(y_test)

		y_pred = Variable(y_pred, requires_grad=True)
		y_test = Variable(y_test)

		y_pred=y_pred.cpu()
		y_test=y_test.cpu()

		y_pred=y_pred.detach().numpy()
		y_test=y_test.detach().numpy()

		#num_of_prots, num_of_classes = y_pred.shape
		#num_test_prots += num_of_prots
		print("type of y_pred: ", type(y_pred))

		if first:
			all_preds = np.array(y_pred)
			first=0
			continue

		all_preds = np.concatenate((all_preds, y_pred), axis=0)
		print("all preds shape after concat: ", all_preds.shape)



		
	


	print("all preds shape: ",all_preds.shape)
	num_test_prots, num_of_classes = all_preds.shape
	TN, FP, FN, TP, optimal_thresholds = find_all_thresholds(all_preds, y_test_all, num_test_prots, num_of_classes)

	####SAVE THE DETERMINED THRESHOLD VALS####
	import csv
	print("THRESHOLDSS:: \n", optimal_thresholds)
	thr_path = os.path.join(test_path, "mfo_thresholds_matth_level_"+str(level) +".csv")
	with open(thr_path, "w+") as thr_f:
		write = csv.writer(thr_f)
		write.writerow(optimal_thresholds)



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


	final_score = os.path.join(test_path, "mfo_level_"+str(level)+"_matth_scores.txt")
	with open(final_score, "w+") as out_s:
		out_s.write(final)




      






