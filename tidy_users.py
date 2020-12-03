import csv
import re

#compare the end of line with the expected pattern to correct for descriptions with new lines that are breaking the CSV
#pattern is the last column, format: ,2018-12-05 20:14:33
with open("users.csv","r") as infile:
    lines = infile.readlines()
lines = [re.sub("\n", "", line) for line in lines]
lines = [re.sub(r"(,[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})", r"\1\n", line) for line in lines]
with open('users2.csv', 'w') as outfile:
        outfile.writelines(lines)

        