# read a csv file "all_seasons.csv"
# and print the first 5 rows

import csv

# open the file
with open('all_seasons.csv', 'r') as f:
    # read the file
    reader = csv.reader(f)
    # print the first 5 rows
    for row in reader:
        print(row)
        break

# close the file
f.close()
