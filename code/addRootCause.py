import csv
import copy
import time
from datetime import datetime
from pandas import DataFrame
import os
import math
import numpy as np
import random

# 读取文件
def readcsv(eventlog):
    csvfile = open(eventlog, 'r', encoding='utf-8')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    sequence = []
    header = next(spamreader)
    # next(spamreader, None)  # skip the headers
    for line in spamreader:
        sequence.append(line)
    return sequence, header

def makeTime(data, index):
    convert = {}
    dayS = 60 * 60 * 24
    ind = 0
    front = data[ind]
    t2 = time.strptime(front[index-1], "%Y/%m/%d %H:%M")
    t = time.strptime(front[index], "%Y/%m/%d %H:%M")
    temp = (datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(t2))).total_seconds() / dayS#
    # data[ind].append(temp) #当前事件的执行时间
    convert[data[ind][3]] = [temp]
    for line in data[1:]:
        ind += 1
        t = time.strptime(line[index], "%Y/%m/%d %H:%M")#%d/%m/%Y %H:%M:%S
        if line[0] == front[0]:
            t2 = time.strptime(front[index], "%Y/%m/%d %H:%M")
            temp = (datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(t2))).total_seconds() / dayS
            if temp < 0:
                print(line[3],temp)
        else:
            t2 = time.strptime(line[index - 1], "%Y/%m/%d %H:%M")
            t = time.strptime(line[index], "%Y/%m/%d %H:%M")
            temp = (datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(t2))).total_seconds() / dayS#0.  #
        # data[ind].append(temp)
        if data[ind][3] in convert.keys():
            convert[data[ind][3]].append(temp)
        else:
            convert[data[ind][3]] = [temp]
        front = line
    return convert

def distinMD(dataset,convert):
    dayS = 60 * 60 * 24
    convert['case'] = []
    # 按轨迹分开
    orginal_trace = list()
    trace_temp = list()
    flag = dataset[0][0]
    for line in dataset:
        if flag == line[0]:
            trace_temp.append(line)
        else:
            t2 = time.strptime(trace_temp[-1][2], "%Y/%m/%d %H:%M")
            t = time.strptime(trace_temp[0][1], "%Y/%m/%d %H:%M")
            temp = (datetime.fromtimestamp(time.mktime(t2)) - datetime.fromtimestamp(
                time.mktime(t))).total_seconds() / dayS
            convert['case'].append(temp)
            # trace_temp[-1].append(temp)
            orginal_trace.append(trace_temp)
            trace_temp = list()
            trace_temp.append(line)
        flag = line[0]
    return orginal_trace

def addCase(trace,convert,type,att,delay,probD=None,probN=None):
    delayTime = delay * max(convert['case'])
    if type == 'cla':
        for case, duration in zip(trace, convert['case']):
            randomNum = random.random()
            if duration > delayTime:
                for i in range(len(probD)):
                    if randomNum < probD[i]:
                        break
            else:
                randomNum = random.random()
                for i in range(len(probN)):
                    if randomNum < probN[i]:
                        break
            for event in case:
                event.append(att[i])
    else:
        for case, duration in zip(trace, convert['case']):
            attValue = (duration / max(convert['case']))*(att[1]-att[0])+att[0]
            for event in case:
                event.append(attValue)

def addEvent(trace,convert,name,type,att,delay,probD=None,probN=None):
    delayTime = delay * max(convert[name])
    dayS = 60 * 60 * 24
    if type == 'cla':
        for case in trace:
            front = []
            for event in case:
                if event[3] == name:
                    t2 = time.strptime(event[2], "%Y/%m/%d %H:%M")
                    if front == []:
                        t = time.strptime(event[1], "%Y/%m/%d %H:%M")
                    else:
                        t = time.strptime(front[2], "%Y/%m/%d %H:%M")
                    duration = (datetime.fromtimestamp(time.mktime(t2)) - datetime.fromtimestamp(
                        time.mktime(t))).total_seconds() / dayS
                    randomNum = random.random()
                    if duration > delayTime:
                        for i in range(len(probD)):
                            if randomNum < probD[i]:
                                break
                    else:
                        randomNum = random.random()
                        for i in range(len(probN)):
                            if randomNum < probN[i]:
                                break
                else:
                    i = 0
                event.append(att[i])
                front = event
    else:
        for case in trace:
            front = []
            for event in case:
                if event[3] == name:
                    t2 = time.strptime(event[2], "%Y/%m/%d %H:%M")
                    if front == []:
                        t = time.strptime(event[1], "%Y/%m/%d %H:%M")
                    else:
                        t = time.strptime(front[2], "%Y/%m/%d %H:%M")
                    duration = (datetime.fromtimestamp(time.mktime(t2)) - datetime.fromtimestamp(
                        time.mktime(t))).total_seconds() / dayS
                    attValue = (duration / max(convert[name])) * (att[1] - att[0]) + att[0]
                else:
                    attValue = (att[1] - att[0]) / 2
                event.append(attValue)
                front = event

eventlog = 'hd'
# 读取文件
data, header = readcsv('../dataset/'+eventlog + '.csv')
# 事件的执行时间
convert = makeTime(data, 2)  # 时间下标
# header.append('duration')
orginal_trace = distinMD(copy.deepcopy(data), convert)
# 添加案例级别根因
header.append('case1')
addCase(orginal_trace,convert,'cla',['a','b','c'],0.8,[0.1,0.9,1],[0.4,0.6,1])  # 类别属性
# header.append('case2')
# addCase(orginal_trace,convert,'num',[0, 10],0.8)  # 数值属性

# 添加事件级别根因
header.append('event1')
addEvent(orginal_trace,convert,'Take in charge ticket','cla',['a','b','c'],0.8,[0.1,0.9,1],[0.4,0.6,1])  # 类别属性
# header.append('event2')
# addEvent(orginal_trace,convert,'Resolve ticket','num',[0, 10],0.8)  # 数值属性

Dict = {}
for i in range(len(header)):
    alist = []
    for case in orginal_trace:
        for line in case:
            alist.append(line[i])
    Dict[header[i]] = alist
new = DataFrame(Dict)
new.to_csv(eventlog+"_RCS.csv", index=False)

print('end')