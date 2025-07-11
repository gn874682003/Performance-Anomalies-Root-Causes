import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

EL = ['hd','hd','BPIC2017','sepsis','BPIC2015_3','CoSeLoG']#'Demo','process104fusion',
sign = ['<', '>', 'v', '^', '*', 's', 'd', 'D', 'p', 'P', 'h', 'H', 'X',
        '<', '>', 'v', '^', '*', 's', 'd', 'D', 'p', 'P', 'h', 'H', 'X',
        '<', '>', 'v', '^', '*', 's', 'd', 'D', 'p', 'P', 'h', 'H', 'X',
        '<', '>', 'v', '^', '*', 's', 'd', 'D', 'p', 'P', 'h', 'H', 'X',
        '<', '>', 'v', '^', '*', 's', 'd', 'D', 'p', 'P', 'h', 'H', 'X',
        '<', '>', 'v', '^', '*', 's', 'd', 'D', 'p', 'P', 'h', 'H', 'X']
delay = 0.8

# 事件级别属性
# for el in EL:
#     resFlag = 1# regressor:1 classical:2 claReg:3
#     # 数值含义对照记录
#     convertReflact = np.load(el + '/' + el + '_header.npy', allow_pickle='TRUE').item()
#     # 根因结果预处理
#     new_dict = np.load(el+"/"+el+"_AresultR.npy",allow_pickle=True).item()#
#     # temp = new_dict['duration:Take in charge ticket > 5.116666666666666']
#     # new_dict.pop('duration:Take in charge ticket > 5.116666666666666')
#     # new_dict['duration:Take in charge ticket > 5.116'] = temp
#     for caseAttName in new_dict:
#         # if caseAttName != 'duration:Wait < 0.01597222222222222':
#         #     continue
#         if caseAttName == 'Case':
#             continue
#         classical = [line for line in new_dict[caseAttName] if len(line) == 4]
#         regressor = [line for line in new_dict[caseAttName] if len(line) == 5]
#         classical.sort(key=lambda x: x[0], reverse=True)
#         regressor.sort(key=lambda x: x[0], reverse=True)
#         claReg = []
#         # 取回归与分类的共同结果
#         for line in regressor:
#             for line2 in classical:
#                 if line[:2] == line2[:2]:
#                     line3 = line.copy()
#                     line3.extend([line2[-1]])
#                     claReg.append(line3)
#         # 选择根因结果类型
#         if resFlag == 1:
#             result = regressor
#         elif resFlag == 2:
#             result = classical
#         elif resFlag == 3:
#             result = claReg
#         if result == []:
#             continue
#         # 读取案例级别属性
#         eventlog = el + '_O'
#         eventName = caseAttName.split('duration:')[1]#
#         if '>' in eventName:
#             eventName = eventName.split(' > ')[0]
#         elif '<' in eventName:
#             eventName = eventName.split(' < ')[0]
#         elif ':' in eventName:
#             eventName = eventName.split(': ')[0]
#         dataDF = pd.read_csv(el + '/' + el + '_' + eventName + '.csv')
#         dataNumpy = dataDF.to_numpy()
#         dataDFO = pd.read_csv(eventlog + '/' + eventlog + '_'+eventName+'_O.csv')
#         dataNumpyO = dataDFO.to_numpy()
#         dataDFoO = pd.read_csv(eventlog + '/' + eventlog + '_' + eventName + '_org_O.csv')
#         dataNumpyoO = dataDFoO.to_numpy()
#         # 获取第一个根因
#         att = result[0][0]
#         i = list(dataDFO.keys()).index(att)
#         y = list(dataNumpyO[:, i])
#         x = list(dataNumpyO[:, -2])
#         plt.rcParams['font.family'] = ['sans-serif']
#         plt.rcParams['font.sans-serif'] = ['SimHei']
#         plt.rcParams['axes.unicode_minus'] = False
#         plt.scatter(x, y, 2, label='others')
#         x.sort()
#         eventDelay3 = x[int(len(x) / 4 * 3)]
#         eventDelay1 = x[int(len(x) / 4)]
#
#         si = 0
#         for line in result:
#             print(line)
#             if line[0] == 'act-org':
#                 if att != line[0]:
#                     plt.vlines(eventDelay1, min(y), max(y), colors='g', label='1/4 Duration')
#                     plt.vlines(eventDelay3, min(y), max(y), colors='r', label='3/4 Duration')
#                     # plt.legend(loc='center', borderaxespad = 8, bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False, shadow=False)
#                     plt.legend(loc='center', bbox_to_anchor=(0.5, -0.3), ncol=6, fontsize=6, frameon=False, shadow=False)#
#                     plt.show()
#                     i = list(dataDFoO.keys()).index(line[0])
#                     y = list(dataNumpyoO[:, i])
#                     x = list(dataNumpyoO[:, -2])
#                     plt.scatter(x, y, 2, label='others')
#                     att = line[0]
#                     si = 0
#                 y1 = []
#                 x1 = []
#                 for i in range(len(y)):
#                     if '>' in line[2] and y[i] > float(line[2].split('>')[1]):
#                         y1.append(y[i])
#                         x1.append(x[i])
#                     elif '<' in line[2] and y[i] < float(line[2].split('<')[1]):
#                         y1.append(y[i])
#                         x1.append(x[i])
#                     elif 'to' in line[2] and y[i] >= float(line[2].split('to')[0]) and y[i] <= float(
#                             line[2].split('to')[1]):
#                         y1.append(y[i])
#                         x1.append(x[i])
#                     elif '>' not in line[2] and '<' not in line[2] and 'to' not in line[2] and y[i] == line[1]:
#                         y1.append(y[i])
#                         x1.append(x[i])
#                 plt.scatter(x1, y1, marker=sign[si], label=line[2] + ': ' + str(round(line[-1], 2)))
#                 si += 1
#                 plt.xlabel('Event Duration')
#                 plt.ylabel(line[0])
#                 plt.title(caseAttName)
#             else:
#                 if att != line[0]:
#                     plt.vlines(eventDelay1, min(y), max(y), colors='g', label='1/4 Duration')
#                     plt.vlines(eventDelay3, min(y), max(y), colors='r', label='3/4 Duration')
#                     plt.legend(loc='center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9, frameon=False, shadow=False)#-0.45
#                     plt.show()
#                     i = list(dataDFO.keys()).index(line[0])
#                     y = list(dataNumpyO[:, i].astype(int))
#                     x = list(dataNumpyO[:, -2])
#                     plt.scatter(x, y, 2, label='others')
#                     att = line[0]
#                     si = 0
#                 y1 = []
#                 x1 = []
#                 x_o = []
#                 if line[0] in ['eventLoad','resourceLoad']:
#                     x_o = list(dataNumpy[:, list(dataDF.keys()).index(line[0])])
#                 for i in range(len(y)):
#                     if '>' in line[2] and y[i] > float(line[2].split('>')[1]):
#                         if x_o != [] and x_o[i] == line[1]:
#                             y1.append(y[i])
#                             x1.append(x[i])
#                         elif x_o == []:
#                             y1.append(y[i])
#                             x1.append(x[i])
#                     elif '<' in line[2] and y[i] < float(line[2].split('<')[1]):
#                         if x_o != [] and x_o[i] == line[1]:
#                             y1.append(y[i])
#                             x1.append(x[i])
#                         elif x_o == []:
#                             y1.append(y[i])
#                             x1.append(x[i])
#                     elif ':' in line[2] and y[i] >= float(line[2].split(': ')[1].split('to')[0]) and y[
#                                 i] <= float(line[2].split(':')[1].split('to')[1]):
#                             if x_o != [] and x_o[i] == line[1]:
#                                 y1.append(y[i])
#                                 x1.append(x[i])
#                             elif x_o == []:
#                                 y1.append(y[i])
#                                 x1.append(x[i])
#                     elif 'to' in line[2] and ':' not in line[2] and y[i] >= float(line[2].split('to')[0]) and y[i] <= float(line[2].split('to')[1]):
#                         if x_o != [] and x_o[i] == line[1]:
#                             y1.append(y[i])
#                             x1.append(x[i])
#                         elif x_o == []:
#                             y1.append(y[i])
#                             x1.append(x[i])
#                     elif '>' not in line[2] and '<' not in line[2] and 'to' not in line[2] and y[i] == line[1]:
#                         if x_o != [] and x_o[i] == line[1]:
#                             y1.append(y[i])
#                             x1.append(x[i])
#                         elif x_o == []:
#                             y1.append(y[i])
#                             x1.append(x[i])
#                 plt.scatter(x1, y1, marker=sign[si], label=line[2] + ': ' + str(round(line[-1], 2)))
#                 si += 1
#                 plt.xlabel('Event Duration')
#                 plt.ylabel(line[0])
#                 plt.title(caseAttName)
#         plt.vlines(eventDelay1, min(y), max(y), colors='g', label='1/4 Duration')
#         plt.vlines(eventDelay3, min(y), max(y), colors='r', label='3/4 Duration')
#         plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.3), ncol=3, fontsize=9, frameon=False, shadow=False)
#         plt.show()
#         print('end')

# print('end')

# 案例级别属性
for el in EL:
    convertReflact = np.load(el + '/' + el + '_header.npy', allow_pickle='TRUE').item()

    resFlag = 1 # regressor:1 classical:2 claReg:3
    # 根因结果预处理
    new_dict = np.load(el+"/"+el+"result_C_S_XGB.npy",allow_pickle=True).item()#Ar_X
    new_dict_reg = np.load(el + "/" + el + "_Aresult.npy", allow_pickle=True).item()  #R
    # new_dict['Case'].append(['duration',129.0,'enter date advice on treatment of environmental permit: 0.0to0.9180555555555555',0.01])#3.939
    # new_dict['Case'].append(['duration', 149.0, 'generate draft decision environmental permit > 0.0006944444444444445', 0.033])#3.542
    # new_dict['Case'].append(['duration', 132.0, 'enter date draft decisionenvironmental permit: 0.0to0.4986111111111111', 0.0])#5.144
    classical = [line for line in new_dict['Case'] if len(line) == 4]
    regressor = [line for line in new_dict_reg['Case'] if len(line) == 5]
    classical.sort(key=lambda x: x[2], reverse=True)
    regressor.sort(key=lambda x: x[2], reverse=True)
    classical.sort(key=lambda x: x[0], reverse=True)
    regressor.sort(key=lambda x: x[0], reverse=True)
    claReg = []
    # 取回归与分类的共同结果
    for line in regressor:
        for line2 in classical:
            if line[:2] == line2[:2]:
                line3 = line.copy()
                line3.extend([line2[-1]])
                claReg.append(line3)
    # 选择根因结果类型
    if resFlag == 1:
        result = regressor
    elif resFlag == 2:
        result = classical
    elif resFlag == 3:
        result = claReg
    # 读取案例级别属性
    eventlog = el + '_O'
    dataDF = pd.read_csv(eventlog + '/' + eventlog + '_Case_O.csv')
    dataNumpy = dataDF.to_numpy()
    dataDFD = pd.read_csv(el + '/' + el + '_duration.csv')
    dataNumpyD = dataDFD.to_numpy()
    dataDFDO = pd.read_csv(eventlog + '/' + eventlog + '_duration_O.csv')
    dataNumpyDO = dataDFDO.to_numpy()
    # 获取第一个根因
    att = result[0][0]
    if att == 'duration':
        i = list(dataDFDO.keys()).index(att)
        y = list(dataNumpyDO[:, i])
        x = list(dataNumpyDO[:, -2])
        att = result[0][2]
    else:
        i = list(dataDF.keys()).index(att)
        y = list(dataNumpy[:, i])
        x = list(dataNumpy[:, -2])
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.scatter(x, y, 2, label='others')
    x.sort()
    caseDelay = x[int(len(x)*delay)]
    caseDelay = 20#3
    si = 0
    for line in result:
        print(line)
        if line[0] == 'duration':
            if '>' in line[2]:
                attline = line[2].split(' > ')[0]
            elif '<' in line[2]:
                attline = line[2].split(' < ')[0]
            else:
                attline = line[2].split(': ')[0]
            if att not in line[2]:
                if att in dataDFD.keys():
                    plt.vlines(caseDelay, min(y), max(y), colors='r', label='Case Delay')
                else:
                    plt.vlines(caseDelay, max(0,min(list(dataNumpyDO[:, id]))), max(list(dataNumpyDO[:, id])), colors='r', label='Case Delay')
                plt.legend(loc='center', bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False, shadow=False)
                plt.show()
                id = list(dataDFDO.keys()).index(line[0])
                plt.scatter([dataNumpyDO[ido, -2] for ido in range(dataNumpyDO.shape[0]) if dataNumpyDO[ido, id] >= 0],
                            [dataNumpyDO[ido, id] for ido in range(dataNumpyDO.shape[0]) if dataNumpyDO[ido, id] >= 0], 2, label='others')
                att = attline
                si = 0
            else:
                id = list(dataDFDO.keys()).index(line[0])
            yd1 = []
            xd1 = []
            for yddi, ydi, xdi in zip(list(dataNumpyD[:, id]), list(dataNumpyDO[:, id]), list(dataNumpyDO[:, -2])):
                if yddi == line[1]:
                    yd1.append(ydi)
                    xd1.append(xdi)
            # yd1 = [25.92453703703704, 4.593449074074074]
            # xd1 = [25.940659722222332, 9.137372685185255]
            if ':' in line[2]:
                valine = line[2].split(': ')[1]
            else:
                valine = line[2]
            if '>' in valine:
                label = valine.split('>')
                plt.scatter(xd1, yd1, marker=sign[si], label='> ' + str(round(float(label[1]), 2))
                                                             + ': ' + str(round(line[-1], 2)))
            elif '<' in valine:
                label = valine.split('<')
                plt.scatter(xd1, yd1, marker=sign[si], label='< ' + str(round(float(label[1]), 2))
                                                             + ': ' + str(round(line[-1], 2)))
            elif 'to' in valine:
                label = valine.split('to')
                plt.scatter(xd1, yd1, marker=sign[si], label=str(round(float(label[0]),2))+ 'to' +
                str(round(float(label[1]),2))+ ': ' + str(round(line[-1], 2)))
            si += 1
            plt.xlabel('Case Duration')
            plt.ylabel(line[0]+' of '+attline)
            plt.title(el)
        else:
            if att != line[0]:
                if att in dataDFD.keys():
                    plt.vlines(caseDelay, min(y), max(y), colors='r', label='Case Delay')
                else:
                    plt.vlines(caseDelay, max(0,min(list(dataNumpyDO[:, id]))), max(list(dataNumpyDO[:, id])), colors='r',
                               label='Case Delay')
                plt.legend(loc='center', bbox_to_anchor=(0.5,-0.2),ncol=4,frameon=False,shadow=False)
                plt.show()
                i = list(dataDF.keys()).index(line[0])
                y = list(dataNumpy[:, i])
                x = list(dataNumpy[:, -2])
                plt.scatter(x, y, 2, label='others')
                att = line[0]
                si = 0
            y1 = []
            x1 = []
            for i in range(len(y)):
                if line[0] == '毛利率':
                    if y[i] == line[1]:
                        y1.append(y[i])
                        x1.append(x[i])
                    continue
                if '>' in line[2] and y[i] > float(line[2].split('>')[1]):
                    y1.append(y[i])
                    x1.append(x[i])
                elif '<' in line[2] and y[i] < float(line[2].split('<')[1]):
                    y1.append(y[i])
                    x1.append(x[i])
                elif 'to' in line[2] and y[i] >= float(line[2].split('to')[0]) and y[i] <= float(line[2].split('to')[1]):
                    y1.append(y[i])
                    x1.append(x[i])
                elif '-' in line[2] and y[i] >= float(line[2].split('-')[0]) and y[i] <= float(line[2].split('-')[1]):
                    y1.append(y[i])
                    x1.append(x[i])
                elif '-' not in line[2] and '>' not in line[2] and '<' not in line[2] and 'to' not in line[2] and y[i] == line[1]:
                    y1.append(y[i])
                    x1.append(x[i])
            plt.scatter(x1, y1, marker=sign[si],label=line[2]+': '+str(np.round(line[-1],2)))#[0]round(line[-1],2)
            print(str(np.round(line[-1],2)))
            si += 1
            plt.xlabel('Case Duration')
            plt.ylabel(line[0])
            plt.title(el)
    if att in dataDFD.keys():
        plt.vlines(caseDelay, min(y), max(y), colors='r', label='Case Delay')
    else:
        plt.vlines(caseDelay, max(0,min(list(dataNumpyDO[:, id]))), max(list(dataNumpyDO[:, id])), colors='r',
                   label='Case Delay')
    plt.legend(loc='center', bbox_to_anchor=(0.5,-0.2),ncol=3,frameon=False,shadow=False)
    plt.show()
    print('end')


print('end')

