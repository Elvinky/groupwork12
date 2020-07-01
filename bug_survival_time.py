# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:43:04 2020

@author: Administrator
"""

from subprocess import Popen, PIPE
import re,time,unicodedata,sys

commit = re.compile('^commit [0-9a-z]{40}$', re.IGNORECASE)
fixes  = re.compile('^\W+Fixes: [a-f0-9]{8,40} \(.*\)$', re.IGNORECASE)

def get_Date(commit_id,repo):
    cmd = ["git","log","--pretty=format:\"%ad\"","-1", commit_id]
    p = Popen(cmd, cwd=repo, stdout=PIPE)
    date, res = p.communicate()
    date = unicodedata.normalize(u'NFKD', date.decode(encoding="utf-8", errors="ignore"))
    dateStamp = time.mktime(time.strptime(date[:-7].strip('"'),"%a %b %d %H:%M:%S %Y"))
    return dateStamp

def get_Commits(kernelRange, repo):
    nr_fixes = 0
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
            nr_fixes += 1
            fix_commit.append(cur_commit[7:19])
            bug_commit.append(line.strip()[7:15])
    #print(fix_commit, bug_commit)
    print("total found fixes:",nr_fixes)
    return fix_commit, bug_commit

def Bug_Survival_Time(fix_commit, bug_commit):
    fix_time_stamp = []
    bug_time_stamp = []
    for commit_id in fix_commit:
        try:
            FixDateStamp = get_Date(fix_commit_id, repo)
            fix_time_stamp.append(FixDateStamp)
        except ValueError:
            pass
    for commit_id in bug_commit:
        try:
            BugDateStamp = get_Date(bug_commit_id, repo)
            bug_time_stamp.append(BugDateStamp)
        except ValueError:
            pass
    #print(fix_date_stamp, bug_date_stamp)
    try:
        date_diff = list(map(lambda x: x[0]-x[1], zip(fix_time_stamp, bug_time_stamp)))
    except len(bug_time_stamp) != len(fix_time_stamp):
        print("The number of fix date is not equal to bug date.")
        sys.exit()
    finally:
        return date_diff