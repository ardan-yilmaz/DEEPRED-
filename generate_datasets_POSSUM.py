import os
import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split


##CHANGE THESE
start_GO = 0 # index-2 of the term to include first OR the previous end_GO. STARTS @ 0
end_GO =  8 # index-1 of the item to include last
model_no = 1
dataset = "dataset1"
level = 8
graph = "MFO"
########################

num_go_terms = end_GO - start_GO 


########################
##CHANGE THE FOLLOWING PARAMETERS
cur_dir = "C:\\Users\\LENOVO\\Desktop\\ardan\\DEEPred"
cur_dir = os.path.join(cur_dir, graph)
level_path = os.path.join(cur_dir, "level_"+str(level))

model_path = os.path.join(level_path, dataset)
out_path = os.path.join(model_path, "model"+str(model_no))
#out_path = "C:\\Users\\LENOVO\\Desktop\\ardan\\DEEPred\\BPO\\level_3\\dataset1\\model1"

#GO_path = "C:\\Users\\LENOVO\\Desktop\\ardan\\DEEPred\\BPO\\level_3\\level_3_filtered"
GO_path = os.path.join(level_path, "level_" + str(level) +"_filtered")
#GO_annot_path = "C:\\Users\\LENOVO\\Desktop\\ardan\\DEEPred\\BPO\\level_3\\level_3_90_filtered_more_than_100.csv"
GO_annot_file = "level_"+str(level)+"_90_filtered_more_than_100.csv"
GO_annot_path = os.path.join(level_path, GO_annot_file)






train_data_path = os.path.join(out_path, "train_data.csv")
train_labels_path = os.path.join(out_path, "train_labels.csv")
test_data_path = os.path.join(out_path, "test_data.csv")
test_labels_path = os.path.join(out_path, "test_labels.csv")



k_sep_path = "C:\\Users\\LENOVO\\Desktop\\ardan\\DEEPred\\k-seperated-biagrams\\k_seperated_biagrams.csv"
#k_sep_path  = "C:\\Users\\LENOVO\\Desktop\\k_sep_trial\\ksep.csv"




class prot_data:
  def __init__(self,  feature_vector):
  	#self.id = pid
    self.feature_vector = feature_vector
    self.labels = [0]*num_go_terms

#load the k_seperated biagrams dataset

k_sep = dict()

print("ksep to load")
with open(k_sep_path, mode='r') as inp:
	reader = csv.reader(inp)
	for row in reader:
		#print(row)
		try:
			val  = [float(x) for x in row[1:]]
			k_sep[row[0]] = prot_data(val)
		except Exception:
			print("PROBLEM HERE: ", row)
			#print("PROBLEM HERE: ", row[0])
			#print("PROBLEM HERE: ", row[1])
			#print("PROBLEM HERE: ", row[1:])				
			pass
    	
    	



print("ksep loaded")



go_terms = pd.read_csv(GO_annot_path)
go_terms_names = go_terms.GO_TERM.to_list()
all_prots = dict()
all_train_data = []
all_train_labels  =[]
num_not_found = 0
total_prot = 0

#print("go_terms \n", go_terms.iloc[start_GO:end_GO])
print("first go: ", go_terms.iloc[start_GO])
print("last go: ", go_terms.iloc[end_GO-1])

cnt  =0
for i, go in enumerate(go_terms_names):

	if i < start_GO: continue
	if i >= end_GO : break

	#print(go_terms.iloc[i])
	cnt += 1
	go += ".csv"
	path = os.path.join(GO_path, go)
	#print("curr index", i)
	with open(path, mode='r') as inp:
		reader = csv.reader(inp)
		prot_list = [rows[0] for rows in reader]

		#check for each prot in k_sep
		for prot_id in prot_list:
			total_prot += 1

			prot = k_sep.get(prot_id)
			if prot == None:
				num_not_found += 1
				#print(prot_id, " NOT FOUND!")
			else:
				#print(prot_id, " is annotated w/ ", go)
				prot.labels[i-start_GO] = 1					
				#print(prot.labels)				
				#if prot in dict.keys():
				all_prots[prot_id] = prot


print("total prots: ", total_prot)
print("not found prot num: ", num_not_found)	

print(cnt)			


for prot in all_prots.values():

	all_train_data.append(prot.feature_vector)
	all_train_labels.append(prot.labels)


X_train, X_test, y_train, y_test = train_test_split(all_train_data, all_train_labels, test_size=0.25, random_state=42)


				
 

  	
#train_data
with open(train_data_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(X_train)

#train_labels
with open(train_labels_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(y_train)

#test data
with open(test_data_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(X_test)  

#test labels
with open(test_labels_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(y_test)    



# write annots file
new_data_frame = go_terms.iloc[start_GO: end_GO]
print(new_data_frame)
out_path = os.path.join(out_path, "annots.csv")
new_data_frame.to_csv(out_path, index =0)
