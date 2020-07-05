# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:50:09 2020

@author: Administrator
"""
__author__ = 'group12'
__version__ = 0.1

from subprocess import Popen, PIPE
import re,time,unicodedata,sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

commit = re.compile('^commit [0-9a-z]{40}$', re.IGNORECASE)
fixes  = re.compile('^\W+Fixes: [a-f0-9]{8,40} \(.*\)$', re.IGNORECASE)

def get_Commit(kernelRange, repo):
    '''get total  commit'''
    commit_num = 0
    commit_count = []
    cmd = ["git", "log", "-p", "--no-merges", kernelRange]
    p = Popen(cmd, cwd=repo, stdout=PIPE)
    data, res = p.communicate()
    data = unicodedata.normalize(u'NFKD', data.decode(encoding="utf-8", errors="ignore"))
    for line in data.split("\n"):
        if(commit.match(line)):
            cur_commit = line
            commit_count.append(cur_commit[7:19])
            commit_num += 1

    print("total found commit:",commit_num)
    return commit_count


def get_fix(kernelRange, repo):
    '''get commit which has a fix tag'''
    fixes_num = 0
    fix_commit = []
    bug_commit = []
    cmd = ["git", "log", "-p", "--no-merges", kernelRange]
    p = Popen(cmd, cwd=repo, stdout=PIPE)
    data, res = p.communicate()
    data = unicodedata.normalize(u'NFKD', data.decode(encoding="utf-8", errors="ignore"))
    for line in data.split("\n"):
        if(commit.match(line)):
            cur_commit = line
        if(fixes.match(line)):
            fixes_num += 1
            fix_commit.append(cur_commit[7:19])
            bug_commit.append(line.strip()[7:15])
            #print(cur_commit[7:19],",",line.strip()[7:16],sep="")
    #print(fix_commit, bug_commit)
    print("total found fixes:",fixes_num)
    return fix_commit, bug_commit

def Commit_year(kernelRange,repo):
    commit_year = []
    cmd = ["git", "log", "--pretty=format:\"%ad\"", kernelRange]
    p = Popen(cmd, cwd=repo, stdout=PIPE)
    data, res = p.communicate()
    data = unicodedata.normalize(u'NFKD', data.decode(encoding="utf-8", errors="ignore"))
    for line in data.split("\n"):
        commit_year.append(line[-11:-6])
    return commit_year

def CommitDate(kernelRange,repo):
    commitdate = []
    cmd = ["git", "log", "--pretty=format:\"%ad\"", kernelRange]
    p = Popen(cmd, cwd=repo, stdout=PIPE)
    data, res = p.communicate()
    data = unicodedata.normalize(u'NFKD', data.decode(encoding="utf-8", errors="ignore"))
    for line in data.split("\n"):
        commitdate.append(line[1:-1])
    return commitdate

def get_timezone(kernelRange,repo):
    '''get the time zone of each commit'''
    timezone = []
    cmd = ["git", "log", "--pretty=format:\"%ad\"", kernelRange]
    p = Popen(cmd, cwd=repo, stdout=PIPE)
    data, res = p.communicate()
    data = unicodedata.normalize(u'NFKD', data.decode(encoding="utf-8", errors="ignore"))
    for line in data.split("\n"):
        timezone.append(line[-6:-1])
    return timezone

def get_author(kernelRange,repo):
    '''get author name'''
    author = []
    cmd = ["git", "log", "--pretty=format:\"%an\"", kernelRange]
    p = Popen(cmd, cwd=repo, stdout=PIPE)
    date, res = p.communicate()
    date = unicodedata.normalize(u'NFKD', date.decode(encoding="utf-8", errors="ignore"))
    for line in date.split("\n"):
        author.append(line[1:-1])
    return author

def get_email(kernelRange,repo):
    '''get email of author '''
    email = []
    cmd = ["git", "log", "--pretty=format:\"%ae\"", kernelRange]
    p = Popen(cmd, cwd=repo, stdout=PIPE)
    date, res = p.communicate()
    date = unicodedata.normalize(u'NFKD', date.decode(encoding="utf-8", errors="ignore"))
    for line in date.split("\n"):
        email.append(line[1:-1])
    return email


def get_content(kernelRange,repo):
    '''get content of commit '''
    content = []
    cmd = ["git", "log", "--pretty=format:\"%ae\"", kernelRange]
    p = Popen(cmd, cwd=repo, stdout=PIPE)
    date, res = p.communicate()
    date = unicodedata.normalize(u'NFKD', date.decode(encoding="utf-8", errors="ignore"))
    for line in date.split("\n"):
        content.append(line[1:-1])
    return content

def write_json(jlist,f_name):
    filename = f_name + '.json'
    # 将bx列表写入json文件
    with open(filename, 'w') as file:  
         json.dump(jlist, file)
        

if __name__ == "__main__":
    repo = "D:\Tencent Files\linux-stable"
    kernelRange = "v4.7..v4.7.10"
    total_commit = get_Commit(kernelRange, repo)
    fix_commit, bug_commit = get_fix(kernelRange, repo)
    author = get_author(kernelRange,repo)
    email =  get_email(kernelRange,repo)
    
    #get total commit
    commit_list=[]
    count=0
    for i in total_commit: #commit has been fixed
        if i in fix_commit:
            commit_list.append(0)
        else:  ##commit has not been fixed
            commit_list.append(1)
        count+=1
    
    #get the timezone of the commit
    timezone = get_timezone(kernelRange, repo)
    commit_timezone = []
    for i in timezone:
        if i[0] == '-':
            commit_timezone.append(-int(i[2]))
        else:
            commit_timezone.append(int(i[2]))
    
    #View data    
    dic = {'Time_zone':commit_timezone,'Commit':commit_list}
    df = pd.DataFrame(dic)
    
    #Draw a scatterplot to view the data distribution
    x = df['Time_zone']
    y = df['Commit']
    plt.scatter(x,y,color='b',label='Commit')
    plt.xlabel('Time_zone')
    plt.ylabel('Commit')
    plt.show()
    
    #Split the training set and use scatter plot observation
    X = np.array(commit_timezone).reshape(-1, 1)
    Y = np.array(commit_list).reshape(-1, 1)
    train_X,test_X,train_y,test_y=train_test_split(X,Y,train_size=0.75,random_state=0)
    plt.scatter(train_X,train_y,color='b',label='train data')
    plt.scatter(test_X,test_y,color='r',label='test data')
    plt.legend(loc=5)
    plt.xlabel('Time_zone')
    plt.ylabel('Commit')
    plt.show()
    
    #Import model
    modelLR=LogisticRegression()
    
    #Train model
    modelLR.fit(train_X,train_y)
    print(modelLR.score(test_X,test_y))
    
    #Specify the forecast at a certain point
    #The probability that the value of commit is 0 or 1
    print(modelLR.predict_proba(np.array(-8).reshape(-1, 1)))
    print(modelLR.predict(np.array(-8).reshape(-1, 1)))
    
    #Log out the regression function and draw the curve
    b=modelLR.coef_
    a=modelLR.intercept_
    print('Regression function:1/(1+exp-(%f+%f*x))'%(a,b))
    
    #Draw the corresponding logistic regression curve
    plt.scatter(train_X,train_y,color='b',label='train data')
    plt.scatter(test_X,test_y,color='r',label='test data')
    plt.plot(test_X,1/(1+np.exp(-(a+b*test_X))),color='r')
    plt.plot(X,1/(1+np.exp(-(a+b*X))),color='y')
    plt.legend(loc=2)
    plt.xlabel('Time_zone')
    plt.ylabel('Commit')
    plt.show()
        

