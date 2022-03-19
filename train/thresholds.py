from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import numpy as np


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


def score_per_class(term, threshold, true_labels, num_of_prots, y_pred): 
  preds = init_pred_per_class(term, y_pred) 
  for i in range(0, num_of_prots):
      if preds[i] > threshold:
          preds[i] = 1
      else:
          preds[i] = 0    

  F1_score = f1_score(true_labels, preds, average="binary")   
  m_score = matthews_corrcoef(true_labels, preds) 


  return F1_score, m_score, preds

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

  cf_m = confusion_matrix(true_labels, preds_opt, labels=[0, 1])
  print("FOR TERM ", term, "threshold: ", optimal_threshold, "w/ score: ", max_f1_score)
  print("conf matrix for class")
  print(cf_m)   
  print()
  TN, FP, FN, TP = cf_m.ravel()


  return TN, FP, FN, TP, optimal_threshold

def treshold_per_class_matth(term, y_test, num_of_prots, y_pred, num_of_classes, class_based_perfs):   
  
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

  total_pred = TP + FP + FN + TN
  print("total pred: ", total_pred)


  acc = (TP + TN)/(total_pred)
  recall = TP / (TP + FN)
  prec = TP/(TP+FP)
  f1 = 2*recall*prec / (recall+prec)
  #["annot num", "TN", "FP", "FN", "TP", "acc", "recall", "prec", "f1", "MCC]
  row= [FN+TP, TN, FP, FN, TP, acc, recall, prec, f1, max_m_score]
  class_based_perfs.append(row) 

  return TN, FP, FN, TP, optimal_threshold

def find_all_thresholds(y_pred, y_test, num_of_prots, num_of_classes, class_based_perfs):
  TN = 0
  FP = 0
  FN = 0
  TP = 0
  optimal_thresholds = []
  for i in range(0, num_of_classes):
      tn, fp, fn, tp, thrshld = treshold_per_class_matth(i, y_test, num_of_prots, y_pred, num_of_classes, class_based_perfs)
      TN += tn
      FP += fp
      FN += fn
      TP += tp
      optimal_thresholds.append(thrshld)

  return TN, FP, FN, TP, optimal_thresholds