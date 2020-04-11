import re
import os

os.remove("./data.txt")

#same pattern
datasets = ["botwiki-2019.tsv","botometer-feedback-2019.tsv",
"celebrity-2019.tsv","cresci-rtbust-2019.tsv","cresci-stock-2018.tsv",
"gilani-2017.tsv","pronbots.tsv","vendor-purchased-2019.tsv",
"midterm-2018.tsv"]



for dataset in datasets:
    with open(f'./data/{dataset}', 'r') as f:
        lines = f.readlines()
    lines = [dataset + ',' + re.sub("\t", ",", line) for line in lines]
    with open('./data.txt', 'a') as f:
        f.writelines(lines)

#varol-2017.dat is messy, create a proper file first with sed
os.chdir("./data")
os.system("sed -e 's/ /\t/g' varol-2017.dat > varol-2017.new")
os.system("sed -e 's/\t\t/\t/g' varol-2017.new > varol-2017.new2")
os.system("sed -e 's/\t\t/\t/g' varol-2017.new2 > varol-2017.new3")
os.chdir("..")

datasets = ["varol-2017.new3"]
for dataset in datasets:
    with open(f'./data/{dataset}', 'r') as f:
        lines = f.readlines()
    lines = [dataset + ',' + re.sub("\t", ",", line) for line in lines]
    lines_sep = [line.split(',',2) for line in lines]
    teste = lines_sep[0]
    lines = [line[0] + "," + line[1] + "," + re.sub("1","bot",re.sub("0","human",line[2])) for line in lines_sep]
    with open('./data.txt', 'a') as f:
        f.writelines(lines)
