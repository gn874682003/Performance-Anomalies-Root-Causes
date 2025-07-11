import csv
import copy
import time
from datetime import datetime
from pandas import DataFrame
import os
import math
import numpy as np

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

# 计算时间特征
def makeTime(data, index):
    dayS = 60 * 60 * 24
    ind = 0
    # front = data[ind]
    # # t2 = time.strptime(front[index-1], "%Y/%m/%d %H:%M:%S")
    # t = time.strptime(front[index], "%Y/%m/%d %H:%M")
    # front[index] = t
    # front[index-1] = time.strptime(front[index-1], "%Y/%m/%d %H:%M")
    # temp = 0. #(datetime.fromtimestamp(time.mktime(temp1)) - datetime.fromtimestamp(time.mktime(temp2))).seconds / dayS#
    # count = temp
    # data[ind].append(temp) #当前事件的执行时间
    # # data[ind].append(count) #总执行时间
    # maxR = temp
    # maxA = count
    # data[ind].append(t.tm_mon) #月/12
    # data[ind].append(t.tm_mday) #日/30
    # data[ind].append(t.tm_wday) #周/7
    # data[ind].append(t.tm_hour) #时/24
    # monD = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}
    # for i in monD.keys():
    #     if t.tm_mon in monD[i]:
    #         data[ind].append(i)  # 'month:'+
    # dayD = {1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    #         3: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]}
    # for i in dayD.keys():
    #     if t.tm_mday in dayD[i]:
    #         data[ind].append(i)
    # weekD = {1: [0, 1, 2, 3, 4], 2: [5, 6]}
    # for i in weekD.keys():
    #     if t.tm_wday in weekD[i]:
    #         data[ind].append(i)  # 'week:'+
    # hourD = {1: [6, 7, 8, 9, 10, 11, 12], 2: [13, 14, 15, 16, 17, 18], 3: [19, 20, 21, 22, 23, 24],
    #          4: [0, 1, 2, 3, 4, 5]}
    # for i in hourD.keys():
    #     if t.tm_hour in hourD[i]:
    #         data[ind].append(i)  # 'hour:'+
    # data[ind].append(t.tm_year-2000) #年-2000 增量
    for line in data:
        t = time.strptime(line[index], "%d/%m/%Y %H:%M:%S")#%Y/%m/%d %H:%M
        # if line[0] == front[0]:
        #     t2 = front[index]  # time.strptime(front[index], "%Y/%m/%d %H:%M")
        #     line[index] = t
        #     temp = (datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(t2))).total_seconds() / dayS
        #     count += temp
        #     line[index-1] = front[index]
        # else:
        t2 = time.strptime(line[index - 1], "%d/%m/%Y %H:%M:%S")
        temp = (datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(
            time.mktime(t2))).total_seconds() / dayS  # 0.  #
        count = temp
        line[index] = time.strptime(line[index], "%d/%m/%Y %H:%M:%S")
        line[index - 1] = time.strptime(line[index - 1], "%d/%m/%Y %H:%M:%S")
        data[ind].append(temp)
        # data[ind].append(count)
        # for i in monD.keys():
        #     if t.tm_mon in monD[i]:
        #         data[ind].append(i)
        # for i in dayD.keys():
        #     if t.tm_mday in dayD[i]:
        #         data[ind].append(i)
        # for i in weekD.keys():
        #     if t.tm_wday in weekD[i]:
        #         data[ind].append(i)
        # for i in hourD.keys():
        #     if t.tm_hour in hourD[i]:
        #         data[ind].append(i)
        data[ind].append(t.tm_mon)  # 月/12
        data[ind].append(t.tm_mday)  # 日/30
        data[ind].append(t.tm_wday)  # 周/7
        data[ind].append(t.tm_hour)  # 时/24
        # data[ind].append(t.tm_year - 2000)
        # front = line
        ind += 1
    return

#对字符属性进行标号
def makeVocabulary(data, index, co=None):
    temp = list()
    for line in data:
        temp.append(line[index])#原去掉int()
    temp_temp = set(temp)#删除重复数据
    if 'null' in temp_temp:
        temp_temp.remove('null')
    if co == None:
        vocabulary = {str(sorted(list(temp_temp))[i]): i + 1 for i in range(len(temp_temp))}
        vocabulary['null'] = 0
    else:
        vocabulary = {co[i]: i for i in range(len(co))}
        for line in temp_temp:
            if line not in vocabulary.keys():
                vocabulary[line] = len(vocabulary)
    for i in range(len(data)):
        data[i][index] = vocabulary[data[i][index]]
    voc = {vocabulary[i]: i for i in vocabulary.keys()}
    return voc

# 计算标签
def makeLable(data, ti, ei, maxA, maxR):
    ind = 0
    front = data[ind]
    for line in data[1:]:
        if line[0] == front[0]:
            data[ind].append(line[ei]) #下一事件
            data[ind].append(line[ti]) #下一事件持续时间*maxR
        else:
            data[ind].append(0)
            data[ind].append(0.)
        front = line
        ind += 1
    data[ind].append(0)
    data[ind].append(0.)
    data[ind].append(0.) #剩余时间
    count = 0.
    for line in reversed(data[0:-1]):
        ind -= 1
        if line[0] == front[0]:
            count += front[ti]#*maxR
        else:
            count = 0.
        front = line
        data[ind].append(count)  # 剩余时间

# 日志预处理
def LogC(eventlog,convertI):
    # 读取文件
    data, header = readcsv('../dataset/'+eventlog + '.csv')
    convertReflact = {}
    for i in range(3,len(data[0])):
        if i in convertI:
            convertReflact[header[i]] = makeVocabulary(data, i)
            # convertReflact.append(makeVocabulary(data, i))
        else:
            if data[0][i] == '':
                data[0][i] = 0
            min = float(data[0][i])
            max = float(data[0][i])
            for j in range(len(data)):
                if data[j][i] == '':
                    data[j][i] = 0
                data[j][i] = float(data[j][i])
                if min > data[j][i]:
                    min = data[j][i]
                if max < data[j][i]:
                    max = data[j][i]
            for j in range(len(data)):
                data[j][i] = data[j][i]/max

    # 时间特征，总执行时间，当前事件的执行时间，月日周时
    makeTime(data, 2)  # 时间下标
    header.append('duration')
    # header.append('allDuration')
    header.append('month')
    header.append('day')
    header.append('week')
    header.append('hour')
    # header.append('year')  # 增量更新
    convertReflact['month'] = {1:'1-3',2:'4-6',3:'7-9',4:'10-12'}
    convertReflact['day'] = {1:'1-10', 2:'11-20', 3:'21-31'}
    convertReflact['week'] = {1:'1-5', 2:'6-7'}
    convertReflact['hour'] = {1:'6-12', 2: '13-18', 3: '19-24', 4: '0-5'}
    return data, header, convertReflact

# 动态、静态属性区分
def distinMD(dataset, State):
    # 按轨迹分开
    orginal_trace = list()
    trace_temp = list()
    flag = dataset[0][0]
    for line in dataset:
        if flag == line[0]:
            trace_temp.append(line)
        else:
            orginal_trace.append(trace_temp)
            trace_temp = list()
            trace_temp.append(line)
        flag = line[0]
    # 输入属性类别编号
    for j in range(1, len(State)):
        for line1 in orginal_trace:
            for i in range(1,len(line1)):
                if line1[0][j] != line1[i][j]:
                    if State[j] == 1 or State[j] == 3:
                        State[j] += 1
                        # break
    return orginal_trace

# 数值属性离散化处理，可给定某分类属性，分别对各类进行离散化处理
def numAttrDiscre(numeric, category=None):
    numI = header.index(numeric)
    if category == None:
        numList = []
        for line in Convert:
            numList.append(line[numI])
        numList.sort()
        num41 = numList[math.floor(len(numList) / 4)]
        num43 = numList[math.floor(len(numList) / 4 * 3)]
        for line in Convert:
            if line[numI] < num41:
                line[numI] = 1
            elif line[numI] > num43:
                line[numI] = 2
            else:
                line[numI] = 3
        convertReflact[numeric] = {1:'<' + str(num41),2:'>' + str(num43),3:str(num41) + 'to' + str(num43)}
    else:
        cateI = header.index(category)
        numDict = {}
        convertReflact[numeric] = {}
        for line in convertReflact[category]:
            numDict[line] = []
        for line in Convert:
            numDict[line[cateI]].append(line[numI])
        for line in numDict:
            if len(numDict[line]) == 0:
                continue
            numDict[line].sort()
            num41 = numDict[line][math.floor(len(numDict[line]) / 4)]
            num43 = numDict[line][math.floor(len(numDict[line]) / 4 * 3)]
            numDict[line] = [num41,num43]
        for line in Convert:
            num41, num43 = numDict[line[cateI]]
            if convertReflact[category][line[header.index(category)]] + ' < ' + str(num41) not in convertReflact[numeric].values():
                convertReflact[numeric][len(convertReflact[numeric].values()) + 1] \
                    = convertReflact[category][line[header.index(category)]] + ' < ' + str(num41) #str(line[header.index(category)])
                convertReflact[numeric][len(convertReflact[numeric].values()) + 1] \
                    = convertReflact[category][line[header.index(category)]] + ' > ' + str(num43)
                convertReflact[numeric][len(convertReflact[numeric].values()) + 1] \
                    = convertReflact[category][line[header.index(category)]] + ': ' + str(num41) + 'to' + str(num43)
            if line[numI] < num41:
                line[numI] = list(convertReflact[numeric].keys())[list(convertReflact[numeric].values()).index(
                    convertReflact[category][line[header.index(category)]] + ' < ' + str(num41))]
            elif line[numI] > num43:
                line[numI] = list(convertReflact[numeric].keys())[list(convertReflact[numeric].values()).index(
                    convertReflact[category][line[header.index(category)]] + ' > ' + str(num43))]
            else:
                line[numI] = list(convertReflact[numeric].keys())[list(convertReflact[numeric].values()).index(
                    convertReflact[category][line[header.index(category)]] + ': ' + str(num41) + 'to' + str(num43))]

# 0 以轨迹开始时间划分，-1 以轨迹结束时间划分
def sortByStartTime(elem):
    return elem[0][1]
def sortByEndTime(elem):
    return elem[-1][1]
def sortByStartEventTime(elem):
    return elem[1]
def sortByEndEventTime(elem):
    return elem[2]

def workLoad(Convert, header, State,StartSort,EndSort):
    # 案例负载
    StartSort.sort(key=sortByStartTime)
    EndSort.sort(key=sortByEndTime)
    workLoad = {}
    for i in range(len(StartSort)):
        j = i + 1
        workLoad[StartSort[i][0][0]] = []
        while j < len(StartSort) and StartSort[j][0][2] < StartSort[i][-1][2]:
            workLoad[StartSort[i][0][0]].append(StartSort[j][0][0])
            j += 1
    for i in range(len(EndSort),0):
        j = i - 1
        while j >= 0 and EndSort[j][-1][2] > EndSort[i][0][2]:
            if EndSort[j][0][0] not in workLoad[EndSort[i][0][0]]:
                workLoad[EndSort[i][0][0]].append(EndSort[j][0][0])
            j -= 1
    for line in Convert:
        if line[0] in workLoad.keys():
            line.append(len(workLoad[line[0]])+1)
        else:
            line.append(1)
    header.append('caseLoad')
    State.append(3)

    # 事件负载
    StartSort = copy.deepcopy(Convert)
    EndSort = copy.deepcopy(Convert)
    StartSort.sort(key=sortByStartEventTime)
    EndSort.sort(key=sortByEndEventTime)
    for i in range(len(StartSort)):
        j = i + 1
        workLoad = []
        while j < len(StartSort) and StartSort[j][1] < StartSort[i][2]:
            workLoad.append(StartSort[j])
            j += 1
        k = EndSort.index(StartSort[i]) - 1
        while k >= 0 and EndSort[k][2] > EndSort[i][1]:
            if EndSort[k] not in workLoad:
                workLoad.append(StartSort[k])
            k -= 1
        temp = Convert[Convert.index(StartSort[i])]
        temp.append(len(workLoad)+1)
        resouceLoad = 1
        for line in workLoad:
            if line[4] == StartSort[i][4]:
                resouceLoad += 1
        temp.append(resouceLoad)
    header.append('eventLoad')
    State.append(4)
    header.append('resourceLoad')
    State.append(4)

# 属性捆绑
def boundAttibute(name, att1, att2):
    numI1 = header.index(att1)
    numI2 = header.index(att2)
    header.append(name)
    State.append(2)
    convertReflact[name] = {}
    count = 1
    for line in Convert:
        bound = convertReflact[att1][line[numI1]]+': '+convertReflact[att2][line[numI2]]
        if bound in convertReflact[name].values():
            line.append(list(convertReflact[name].values()).index(bound)+1)
        else:
            convertReflact[name][count] = bound
            line.append(count)
            count += 1



EL = ['BPIC2017']#'process104fusion', 'Demo','hd','BPIC2015_3','sepsis','CoSeLoG',
Att = [#[3,4,5,6,7], [3,4,5,6,7,8,9,10,11,12,13],
       # [3,4,5,6,7,8,9,10,11,12],[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],#
       # [3,5,6,7,8,9,10,11,12,14,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
       # [3,4,5,6,7,8,9],
       [3,4,5,7,9,11,13,14]]
Delay = 0.8
for eventlog, attribute in zip(EL,Att):
    Convert, header, convertReflact = LogC(eventlog,attribute)#需转换属性值的下标    line[1] = line[2] - line[-10]
    eventlog = eventlog+'_O'
    if os.path.exists(eventlog) != True:
        os.mkdir(eventlog)
    # 特征类别编号 0 活动；1 分类静态特征；2 分类动态特征；3 数值静态特征；4 数值动态特征
    State = [0,4,4,0]
    for i in range(1, len(Convert[0])-8):
        if i + 3 in attribute:
            State.append(1)
        else:
            State.append(3)
    for i in range(5):
        State.append(4)
    orginal_trace = distinMD(copy.deepcopy(Convert), State)
    if eventlog == 'sepsis_O':
        boundAttibute('act-org', 'concept:name', 'org:group')
    else:
        boundAttibute('act-org', 'concept:name', 'org:resource')
    # 添加工作负载属性
    workLoad(Convert, header, State,copy.deepcopy(orginal_trace),copy.deepcopy(orginal_trace))
    # duration离散化处理前备份
    # for line in Convert:
    #     numI = header.index('duration')
    #     line.append(line[numI])
    # header.append('eventDuration')
    # State.append(4)
    # 数值属性离散化处理，可给定某分类属性，分别对各类进行离散化处理
    # numAttrDiscre('duration', 'concept:name')

    # numAttrDiscre('eventLoad', 'concept:name')
    # if eventlog == 'sepsis':
    #     numAttrDiscre('resourceLoad', 'org:group')
    # else:
    #     numAttrDiscre('resourceLoad', 'org:resource')

    # 按轨迹分开
    orginal_trace = list()
    trace_temp = list()
    flag = Convert[0][0]
    for line in Convert:
        if flag == line[0]:
            trace_temp.append(line)
        else:
            orginal_trace.append(trace_temp)
            trace_temp = list()
            trace_temp.append(line)
        flag = line[0]

    # 导出案例级别属性文件
    case = []
    convertReflact['varient'] = {}
    varients = []
    for line in orginal_trace:
        clist = []
        # varient = convertReflact['concept:name'][line[0][3]]
        for i in range(len(line[0])):
            if State[i] == 1 or State[i] == 3 or header[i] in ['month', 'day', 'week']:
                clist.append(line[0][i])
        # for line2 in line[1:]:
        #     varient = varient + ' -> ' + convertReflact['concept:name'][line2[3]]
        # if varient in convertReflact['varient'].values():
        #     clist.append(list(convertReflact['varient'].keys())[list(
        #         convertReflact['varient'].values()).index(varient)])
        # else:
        #     convertReflact['varient'][len(convertReflact['varient']) + 1] = varient
        #     clist.append(len(convertReflact['varient']))
        varients.append(len(line))
        clist.append(len(line))

        clist.append(datetime.fromtimestamp(time.mktime(line[-1][2])).timestamp() / (24*60*60)
                     - datetime.fromtimestamp(time.mktime(line[0][1])).timestamp() / (24*60*60))
        case.append(clist)

    caseDelay = [line[-1] for line in case]
    caseDelay.sort()
    caseDelay = caseDelay[int(len(caseDelay)*Delay)]
    caseDelay = 1#3
    len2 = varients[int(len(varients)/3)]
    len3 = varients[int(len(varients)/3)]*2
    convertReflact['varient'] = {1: '< '+str(len2), 2: '> ' + str(len3), 3: str(len2) + ' to ' + str(len3)}
    for line in case:
        # if line[-2] < len2:
        #     line[-2] = 1
        # elif line[-2] > len3:
        #     line[-2] = 2
        # else:
        #     line[-2] = 3
        line.append(1 if line[-1] > caseDelay else 0)

    Dict = {}
    name = [header[i] for i in range(len(header)) if State[i] == 1 or State[i] == 3 or header[i] in ['month','day','week']]
    name.append('varient')
    name.append('caseDuration')
    name.append('caseDelay')
    for i in range(len(name)):
        alist = []
        for line in case:
            alist.append(line[i])
        Dict[name[i]] = alist

    new = DataFrame(Dict)

    new.to_csv(eventlog+"/"+eventlog+"_Case_O.csv", index=False)


    # 导出所有事件执行时间属性文件
    event = []
    for line in orginal_trace:
        for line2 in line:
            clist = []
            for i in range(4,len(line2)):
                if State[i] == 1 or State[i] == 3 or header[i] in ['month', 'day', 'week','duration']:
                    clist.append(line2[i])
            clist.append(len(line))
            clist.append(datetime.fromtimestamp(time.mktime(line[-1][2])).timestamp() / (24*60*60)
                         - datetime.fromtimestamp(time.mktime(line[0][1])).timestamp() / (24*60*60))
            event.append(clist)
    for line in event:
        # if line[-2] < len2:
        #     line[-2] = 1
        # elif line[-2] > len3:
        #     line[-2] = 2
        # else:
        #     line[-2] = 3
        line.append(1 if line[-1] > caseDelay else 0)
    Dict = {}
    name = [header[i] for i in range(4,len(header)) if State[i] == 1 or State[i] == 3 or header[i] in ['month','day','week','duration']]
    name.append('varient')
    name.append('caseDuration')
    name.append('caseDelay')
    for i in range(len(name)):
        alist = []
        for line in event:
            alist.append(line[i])
        Dict[name[i]] = alist

    new = DataFrame(Dict)
    new.to_csv(eventlog+"/"+eventlog+"_duration_O.csv", index=False)


    # 导出事件级别属性文件
    for e in convertReflact['concept:name']:
        event = []
        event_org = [] # 包含事件的活动与资源前缀
        for line in orginal_trace:
            for line2 in line:
                clist = []
                if line2[3] == e:
                    for i in range(4,len(line2)):
                        if (State[i] == 2 or State[i] == 4) and header[i] not in ['duration','act-org']:
                            clist.append(line2[i])
                    clist.append(line2[header.index('duration')])#%3
                    event.append(clist)

                    for line3 in line:
                        olist = []
                        for i in range(4, len(line2)):
                            if (State[i] == 2 or State[i] == 4) and header[i] != 'duration':
                                if header[i] == 'act-org':
                                    olist.append(line3[i])
                                else:
                                    olist.append(line2[i])
                        olist.append(line2[header.index('duration')])# % 3
                        event_org.append(olist)
                        if line3 == line2:
                            break


        if event == []:
            continue
        eventDelay = [line[-2] for line in event]
        eventDelay.sort()
        eventDelay = eventDelay[int(len(eventDelay) * Delay)]
        for line in event:
            line.append(1 if line[-2] > eventDelay else 0)

        for line in event_org:
            line.append(1 if line[-2] > eventDelay else 0)

        Dict = {}
        name = [header[i] for i in range(4,len(header)) if (State[i] == 2 or State[i] == 4) and header[i] not in ['duration','act-org']]
        name.append('duration')
        name.append('eventDelay')
        for i in range(len(name)):
            alist = []
            for line in event:
                alist.append(line[i])
            Dict[name[i]] = alist

        new = DataFrame(Dict)
        if event != []:
            new.to_csv(eventlog+"/"+eventlog+"_"+convertReflact['concept:name'][e]+"_O.csv", index=False)

        Dict = {}
        name = [header[i] for i in range(4, len(header)) if (State[i] == 2 or State[i] == 4) and header[i] != 'duration']
        name.append('duration')
        name.append('eventDelay')
        for i in range(len(name)):
            alist = []
            for line in event_org:
                alist.append(line[i])
            Dict[name[i]] = alist

        new = DataFrame(Dict)
        if event != [] and event_org != []:
            new.to_csv(eventlog + "/" + eventlog + "_" + convertReflact['concept:name'][e] + "_org_O.csv", index=False)




    # 保存文件
    np.save(eventlog+"/"+eventlog+'_header.npy', convertReflact)
    # 读取文件
    new_dict = np.load(eventlog+"/"+eventlog+'_header.npy', allow_pickle='TRUE').item()