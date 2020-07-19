# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:14:59 2020

@author: Administrator
"""

__author__ = 'ZhangYi'
__version__ = 0.1

import subprocess, re
import matplotlib.pyplot as plt

def get_version(repo):
    '''
    get the versions. 
    '''
    git_tag = subprocess.Popen(["git", "tag"], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True)
    git_tag = re.findall('v[0-9].[0-9]', str(git_tag.communicate()[0]))
    git_versions = []
    for i in git_tag:
        if i not in git_versions:
            git_versions.append(i)
    git_versions.pop(0) 
    git_versions.pop()
    return git_versions

def get_modify_time(repo):
    '''
    get the modify time of each version.
    '''
    seconds_times = []
    for i in range(0,len(get_version(repo))):
        git_tag = "git log -1 --pretty=format:\"%ct\" " + get_version(repo)[i]
        git_rev_list = subprocess.Popen(git_tag, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True)
        tag_counts = git_rev_list.communicate()[0]
        if i == 0:
            seconds_times.append(int(tag_counts))
        else:
            seconds_times.append((int(tag_counts) - seconds_times[0])//24//3600)
    seconds_times[0] = 0
    return seconds_times

def draw_plot(repo):
    '''
    draw the plot.
    '''
    plt.scatter(get_modify_time(repo),get_version(repo))
    plt.title("Modification time of different versions")
    plt.xlabel("days")
    plt.ylabel("version")
    plt.show()


if __name__ == "__main__":
    repo = "D:\Tencent Files\linux-stable"
    print('Git versions: ', get_version(repo))
    print('Modify time: ', get_modify_time(repo))
    print(draw_plot(repo))