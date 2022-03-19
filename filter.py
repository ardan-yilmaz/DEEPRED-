import os 
import csv
import sys
       

all_annot_files_path = "C://Users//LENOVO//Desktop//ardan//DEEPred//annotations"

maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

all_clusters = dict()



def create_cluster_dict(cluster_path):
    with open(cluster_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row[0].startswith("Cluster"): continue
            prot_l = row[4:][0].split(';')

            #list to be preprocessed as items contains ' ' at beginning
            num_of_prots = row[3]
            for i in range(0, int(num_of_prots)):
                if (prot_l[i])[0] == ' ':
                    prot_l[i] = (prot_l[i])[1:]
            
            all_clusters[row[0]] = prot_l


def write_filtered(prots, path, GO_Term, level):
    new_file_name = GO_Term + "_filtered_90.csv"
    new_path = os.path.join(path, new_file_name)
    print("path being written: ", new_path)

    with open(new_path, 'w+') as csv_file:  
        writer = csv.writer(csv_file, delimiter = ',', lineterminator='\n')
        for key, value in prots.items():
            writer.writerow([key, ' '.join(map(str, value))])


            
def apply_filtering(prots):
    
    for cluster_prots in all_clusters.values():
        flag = -1
        representative_prot = cluster_prots[0]
        if representative_prot in prots.keys():
            #prot_to_keep may be deleted later. keeping 4 debugging purposes for now
            prot_to_keep = representative_prot
            flag = 1
        
        for cluster_prot in cluster_prots:
            if cluster_prot in prots.keys():
                if flag == 1: 
                    if cluster_prot == representative_prot: continue
                    prots.pop(cluster_prot)
                
                else:
                    prot_to_keep = cluster_prot
                    flag = 1






def annot_reader(path, file_name, level, level_path):
    
    print("term", file_name, "being read")

    prots = dict()
    with open(path, 'r') as f:
        reader = csv.reader(f)  
        for row in reader:
            prots[row[0]] = row[1:]

        print("filtering on", file_name[:-4])    
        apply_filtering(prots)
        print("filtering complete on", file_name[:-4])
        
        level_name = "level_"+ str(level)

        out_path = os.path.join(level_path, level_name+"_filtered")

        if not os.path.exists(out_path): os.mkdir(out_path)
        
        #print("out_path::", out_path)

        write_filtered(prots, out_path, file_name[:-4], level)       
        



def level_terms_reader(f, level, level_path):

    print("level:: ", level)
    all_termms_in_level = list()

    reader = csv.reader(f)  
    for row in reader:
        if(row[0] == "GO_TERM"): continue
        row_l = row[0].split(':')
        to_append = row_l[0] + '_' + row_l[1]
        all_termms_in_level.append(to_append)
            
    print("level ", level, "read")            

    for term in all_termms_in_level:
        file_name = term + '.csv'
        term_path = os.path.join(all_annot_files_path, file_name)
        print("term path::", term_path)


        annot_reader(term_path, file_name, level, level_path)






if __name__ == "__main__":

    #should be BPO, CCO or MFO
    graph = "CCO"
    level = sys.argv[1]

    cluster_path = "C://Users//LENOVO//Desktop//ardan//DEEPred//filtre//uniref-uniprot_(reviewed_yes)+identity_0.9.tab"
    #cluster_path_ex = "C://Users//LENOVO//Desktop//ardan//DEEPred//filtre//trial//clusters_meee.tab"
    create_cluster_dict(cluster_path)
    print("clusters dict created") 	

    main_path = "C:\\Users\\LENOVO\\Desktop\\ardan\\DEEPred"
    main_path = os.path.join(main_path, graph)	

    level_path = os.path.join(main_path, "level_"+str(level))
    file_to_filter_name = "level_" + str(level) + "_to_be_filtered.csv"
    file_to_filter = os.path.join(level_path, file_to_filter_name)

    try:
        f = open(file_to_filter)

    except FileNotFoundError:
        print(file_to_filter)
        print("FileNotFoundError 11111")        
    else:
        level_terms_reader(f, level, level_path)



    """
    levels = os.listdir(main_path)
	for level in levels:
		print("curr_level: ", level)
		cur_path = os.path.join(main_path, level)
		file_name = level + "_to_be_filtered.csv"
		cur_path = os.path.join(cur_path, file_name)
		print("cur_path: ", cur_path)
		try:
			f = open(cur_path)

		except FileNotFoundError:
			print("FileNotFoundError 11111")		
		else:
			level_terms_reader(f, level, main_path)
			print("level ", level, "done\n")	
    """		





