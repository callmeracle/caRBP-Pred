import re
import sys

pos_pep_lst=[]
with open(sys.argv[1],'r') as pos:#### positive peptide
    for line in pos:
        line = line.strip().split(",")
        pos_pep_lst.append(line[1])

neg_pep_lst=[]
with open(sys.argv[2],'r') as neg:
    for line in neg:
        line = line.strip().split(",")
        neg_pep_lst.append(line[1])

window_size=50
#pos_file=open(sys.argv[4],"w")
#neg_file=open(sys.argv[5],"w")

with open(sys.argv[3], 'r') as f:#### fasta
        for line in f:
                line = line.strip().split(',')
                if len(line[1]) <= window_size:
                        print ("error")
                for i in range(len(line[1]) - window_size + 1):
                        for p in pos_pep_lst:
                                if p in line[1][i:i+window_size] and p not in neg_pep_lst:####### must be perfectly match
                                        print (line[0])
#                               else:
#                                       print (">sp|"+line[0],line[1],sep="\n",file=neg_file)
