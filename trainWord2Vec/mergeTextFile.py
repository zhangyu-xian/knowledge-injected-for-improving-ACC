#coding=utf-8
import os

filedir = os.getcwd()+'./corpus'
filenames=os.listdir(filedir)
f=open('result.txt','w', encoding='utf-8')

for filename in filenames:
    filepath = filedir+'/'+filename
    
    for line in open(filepath, encoding='utf-8'):
        f.writelines(line)
    # f.write('\n')

f.close()
