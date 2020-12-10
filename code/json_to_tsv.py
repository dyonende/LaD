import csv
import json
import glob
import sys

out_name = input("name the output file: ")
path_to_data = input("give path to data folder: ")
data = list()
data.append("link\tdate\twebsite\ttext\n")
with open(out_name, "w") as outfile:
    for filepath in glob.glob(path_to_data+"/*.json"):
        with open(filepath) as infile:
            json_data = json.load(infile)
            
            link = json_data["link"].replace('\t', ' ')
            link = link.replace('\n', ' ')
            
            date = json_data["date"].replace('\t', ' ')
            date = date.replace('\n', ' ')
            
            website = json_data["website"].replace('\t', ' ')
            website = website.replace('\n', ' ')
            
            text = json_data["text"].replace("\t", " ")
            text = text.replace('\n', " ")
            
            data.append(link+'\t'+date+'\t'+website+'\t'+text+'\n')
    
    outfile.writelines(data)


