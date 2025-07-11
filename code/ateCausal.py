
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import warnings
import os
# from causalml.inference.meta import XGBTLearner, MLPTLearner
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor, BaseDRRegressor
from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier, BaseRClassifier, BaseDRLearner
from causalml.inference.iv import BaseDRIVRegressor, BaseDRIVLearner
from causalml.inference.meta import LRSRegressor
from causalml.match import NearestNeighborMatch, MatchOptimizer, create_table_one
from causalml.propensity import ElasticNetPropensityModel
from causalml.dataset import *
from causalml.metrics import *

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# imports from package
import logging
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import statsmodels.api as sm
from copy import deepcopy
import time

# logger = logging.getLogger('causalml')
# logging.basicConfig(level=logging.INFO)

EL = ['hd','BPIC2017','Sepsis','CoSeLoG','BPIC2015_3']# 'Demo','process104fusion',
# EL = ['simLog1','simLog2','simLog3','simLog4','simLog5']#
# EL = ['BPIC2015_1','BPIC2015_2','BPIC2015_4','BPIC2015_5']
for eventlog in EL:
    print(eventlog)
    start = time.time()
    # 读取文件
    convertReflact = np.load(eventlog+'/'+eventlog+'_header.npy', allow_pickle='TRUE').item()
    # 案例级别属性评估ATE
    dataDF = pd.read_csv(eventlog+'/'+eventlog+'_Case.csv')
    dataNumpy = dataDF.to_numpy()
    dict = {}
    recordATEC = []
    y = dataNumpy[:,-2]
    yc = dataNumpy[:,-1]
    maxTime = y.max() - y.min()#20 process104fusion
    for i,name in zip(range(len(dataDF.keys())-2), dataDF.keys()):
        treatment = dataNumpy[:, i]
        X = dataNumpy[:, [j for j in range(len(dataDF.keys())-2) if j != i]]
        for k in convertReflact[name].keys():
            if k in list(dataDF[name]) and np.unique(treatment).size > 1:
                treatment2 = np.array(['treatment_a' if val == k else 'control' for val in treatment])
                # if np.sum(treatment2 == 'treatment_a') < 3:
                #     continue
                # print(np.sum(treatment2 == 'treatment_a'))
                learner_dr = BaseSRegressor(XGBRegressor(), control_name='control')  # BaseSRegressor
                # learner_dr = BaseTRegressor(control_learner=LGBMRegressor(), treatment_learner=LGBMRegressor(),
                #                             control_name='control')
                ate_s = learner_dr.estimate_ate(X=X, treatment=treatment2, y=y)
                gini = np.load('gini.npy', allow_pickle='TRUE').item()
                if ate_s[0] / maxTime > 0.05 and gini == 1:
                    recordATEC.append([name, k, convertReflact[name][k], ate_s[0], ate_s[0] / maxTime])
                # trey = list(set([yi for tre, yi in zip(treatment2, yc) if tre == 'treatment_a']))
                # cony = list(set([yi for tre, yi in zip(treatment2, yc) if tre == 'control']))
                # if len(trey) > 1 and len(cony) > 1:
                #     learner_c = BaseSClassifier(XGBClassifier(), control_name='control') # XGBRegressor(),
                #     yc = yc.astype(int)
                #     ate_c = learner_c.estimate_ate(X=X, treatment=treatment2, y=yc)
                #     if ate_c[0] > 0.05:
                #         recordATEC.append([name, k, convertReflact[name][k], ate_c[0]])
                # elif len(trey) == 1 and trey == 1 and len(cony) == 1 and cony == 0:
                #     recordATEC.append([name, k, convertReflact[name][k], 1])
                # ginic = np.load('gini.npy', allow_pickle='TRUE').item()


    # 事件执行时间评估ATE
    dataDF = pd.read_csv(eventlog + '/' + eventlog + '_duration.csv')
    dataNumpy = dataDF.to_numpy()
    y = dataNumpy[:, -2]
    yc = dataNumpy[:, -1]
    i = list(dataDF.keys()).index('duration')
    treatment = dataNumpy[:, i]
    X = dataNumpy[:, [j for j in range(len(dataDF.keys()) - 2) if j != i]]
    k = np.unique(treatment)
    for m in k:
        recordATEE = []
        treatment2 = np.array(['treatment_a' if val == m else 'control' for val in treatment])
        learner_dr = BaseSRegressor(XGBRegressor(), control_name='control')  # BaseSRegressor
        # learner_dr = BaseTRegressor(control_learner=LGBMRegressor(), treatment_learner=LGBMRegressor(),
        #                             control_name='control')
        ate_s = learner_dr.estimate_ate(X=X, treatment=treatment2, y=y)
        gini = np.load('gini.npy', allow_pickle='TRUE').item()
        if ate_s[0] / maxTime > 0.05 and gini == 1:
            recordATEC.append(['duration', m, convertReflact['duration'][m], ate_s[0], ate_s[0] / maxTime])
        # trey = list(set([yi for tre, yi in zip(treatment2, yc) if tre == 'treatment_a']))
        # cony = list(set([yi for tre, yi in zip(treatment2, yc) if tre == 'control']))
        # if len(trey) > 1 and len(cony) > 1:
        #     learner_c = BaseSClassifier(XGBClassifier(), control_name='control')  #, XGBRegressor()
        #     yc = yc.astype(int)
        #     ate_c = learner_c.estimate_ate(X=X, treatment=treatment2, y=yc)
        #     if ate_c[0] > 0.05:
        #         recordATEC.append(['duration', m, convertReflact['duration'][m], ate_c[0]])
        # elif len(trey) == 1 and trey[0] == 1:# and len(cony) == 1 and cony == 0:
        #     ate_c[0] = 1
        #     if ate_c[0] > 0.05:
        #         recordATEC.append(['duration', m, convertReflact['duration'][m], ate_c[0]])
        # else:
        #     ate_c[0] = 0
        startR = time.time()
        # 事件级别属性回归评估ATE
        if (ate_s[0] / maxTime > 0.05 and gini == 1):# or ate_c[0] > 0.05
            if " > " in convertReflact['duration'][m]:
                act = convertReflact['duration'][m].split(" > ")[0]
                flag = 1
            elif " < " in convertReflact['duration'][m]:
                act = convertReflact['duration'][m].split(" < ")[0]
                flag = 2
            else:
                act = convertReflact['duration'][m].split(": ")[0]
                flag = 0
            if os.path.exists(eventlog + '/' + eventlog + '_' + act + '.csv'):
                dataDFE = pd.read_csv(eventlog + '/' + eventlog + '_' + act + '.csv')
            else:
                continue
            dataNumpy3 = dataDFE.to_numpy()
            y3 = dataNumpy3[:, -3]
            y3c = np.array([1 if i == flag else 0 for i in dataNumpy3[:, -2]])
            for i3, name in zip(range(len(dataDFE.keys()) - 3), dataDFE.keys()):
                treatment3 = dataNumpy3[:, i3]
                X3 = dataNumpy3[:, [j for j in range(len(dataDFE.keys()) - 3) if j != i3]]
                for n in convertReflact[name].keys():
                    if n in list(dataDFE[name]) and np.unique(treatment3).size > 1:
                        treatment2 = np.array(['treatment_a' if val == n else 'control' for val in treatment3])
                        learner_dr = BaseSRegressor(XGBRegressor(), control_name='control')  # BaseDRRegressor
                        # learner_dr = BaseTRegressor(control_learner=LGBMRegressor(), treatment_learner=LGBMRegressor(),
                        #                             control_name='control')
                        ate_s = learner_dr.estimate_ate(X=X3, treatment=treatment2, y=y3)
                        gini = np.load('gini.npy', allow_pickle='TRUE').item()
                        if ate_s[0] / maxTime > 0.05 and gini == 1:
                            recordATEE.append([name, n, convertReflact[name][n], ate_s[0], ate_s[0] / maxTime])
                        # trey = list(set([yi for tre, yi in zip(treatment2, y3c) if tre == 'treatment_a']))
                        # cony = list(set([yi for tre, yi in zip(treatment2, y3c) if tre == 'control']))
                        # if len(trey) > 1 and len(cony) > 1:
                        #     learner_c = BaseSClassifier(XGBClassifier(), control_name='control')  # , XGBRegressor()
                        #     y3c = y3c.astype(int)
                        #     ate_c = learner_c.estimate_ate(X=X3, treatment=treatment2, y=y3c)
                        #     if ate_c[0] > 0.05:
                        #         recordATEE.append([name, n, convertReflact[name][n], ate_c[0]])
                        # elif len(trey) == 1 and trey == 1 and len(cony) == 1 and cony == 0:
                        #     recordATEE.append([name, n, convertReflact[name][n], 1])

            #事件的活动与资源前缀评估ATE
            if os.path.exists(eventlog + '/' + eventlog + '_' + act + '_org.csv'):
                dataDFE = pd.read_csv(eventlog + '/' + eventlog + '_' + act + '_org.csv')
            else:
                continue
            dataNumpy3 = dataDFE.to_numpy()
            y3 = dataNumpy3[:, -3]
            y3c = np.array([1 if i == flag else 0 for i in dataNumpy3[:, -2]])
            name = 'act-org'
            i3 = list(dataDFE.keys()).index(name)
            treatment3 = dataNumpy3[:, i3]
            X3 = dataNumpy3[:, [j for j in range(len(dataDFE.keys()) - 3) if j != i3]]
            for n in convertReflact[name].keys():
                if n in list(dataDFE[name]) and np.unique(treatment3).size > 1:
                    treatment2 = np.array(['treatment_a' if val == n else 'control' for val in treatment3])
                    learner_dr = BaseSRegressor(XGBRegressor(), control_name='control')  # BaseDRRegressor
                    # learner_dr = BaseTRegressor(control_learner=LGBMRegressor(), treatment_learner=LGBMRegressor(),
                    #                             control_name='control')
                    ate_s = learner_dr.estimate_ate(X=X3, treatment=treatment2, y=y3)
                    gini = np.load('gini.npy', allow_pickle='TRUE').item()
                    if ate_s[0] / maxTime > 0.05 and gini == 1:
                        recordATEE.append([name, n, convertReflact[name][n], ate_s[0], ate_s[0] / maxTime])
                    # trey = list(set([yi for tre, yi in zip(treatment2, y3c) if tre == 'treatment_a']))
                    # cony = list(set([yi for tre, yi in zip(treatment2, y3c) if tre == 'control']))
                    # if len(trey) > 1 and len(cony) > 1:
                    #     learner_c = BaseSClassifier(XGBClassifier(), control_name='control')  #XGBRegressor(),
                    #     y3c = y3c.astype(int)
                    #     ate_c = learner_c.estimate_ate(X=X3, treatment=treatment2, y=y3c)
                    #     if ate_c[0] > 0.05:
                    #         recordATEE.append([name, n, convertReflact[name][n], ate_c[0]])
                    # elif len(trey) == 1 and trey == 1 and len(cony) == 1 and cony == 0:
                    #     recordATEE.append([name, n, convertReflact[name][n], 1])

            recordATEE.sort(key=lambda x: x[3], reverse=True)
            dict['duration' + ':' + convertReflact['duration'][m]] = recordATEE
        endR = time.time()
        print(endR - startR, convertReflact['duration'][m])
    recordATEC.sort(key=lambda x: x[3], reverse=True)
    dict['Case'] = recordATEC

    # 事件级别属性评估ATE
    if (ate_s[0] / maxTime > 0.05 and gini == 1):
        if " > " in convertReflact['duration'][m]:
            act = convertReflact['duration'][m].split(" > ")[0]
        elif " < " in convertReflact['duration'][m]:
            act = convertReflact['duration'][m].split(" < ")[0]
        else:
            act = convertReflact['duration'][m].split(": ")[0]
        if os.path.exists(eventlog + '/' + eventlog + '_' + act + '.csv'):
            dataDF = pd.read_csv(eventlog + '/' + eventlog + '_' + act + '.csv')
        else:
            continue
        dataNumpy3 = dataDF.to_numpy()
        recordATEC.append(['duration', m, convertReflact['duration'][m], ate_s[0], ate_s[0] / maxTime])
        y3 = dataNumpy3[:, -2]
        y3c = dataNumpy3[:, -1]
        for i3, name in zip(range(len(dataDF.keys()) - 2), dataDF.keys()):
            treatment3 = dataNumpy3[:, i3]
            X3 = dataNumpy3[:, [j for j in range(len(dataDF.keys()) - 2) if j != i3]]
            for n in convertReflact[name].keys():
                if n in list(dataDF[name]) and np.unique(treatment3).size > 1:
                    treatment2 = np.array(['treatment_a' if val == n else 'control' for val in treatment3])
                    # if np.sum(treatment2 == 'treatment_a') < 3:
                    #     continue
                    # print(np.sum(treatment2 == 'treatment_a'))
                    learner_dr = BaseSRegressor(XGBRegressor(), control_name='control')  # BaseDRRegressor
                    # learner_dr = BaseTRegressor(control_learner=LGBMRegressor(), treatment_learner=LGBMRegressor(),
                    #                             control_name='control')
                    ate_s = learner_dr.estimate_ate(X=X3, treatment=treatment2, y=y3)
                    gini = np.load('gini.npy', allow_pickle='TRUE').item()
                    # trey = list(set([yi for tre, yi in zip(treatment2, y3c) if tre == 'treatment_a']))
                    # cony = list(set([yi for tre, yi in zip(treatment2, y3c) if tre == 'control']))
                    # if len(trey) > 1 and len(cony) > 1:
                    #     learner_c = BaseSClassifier(XGBClassifier(), control_name='control')  # , XGBRegressor()
                    #     y3c = y3c.astype(int)
                    #     ate_c = learner_c.estimate_ate(X=X3, treatment=treatment2, y=y3c)
                    #     if ate_c[0] > 0.05:
                    #         recordATEE.append([name, n, convertReflact[name][n], ate_c[0]])
                    # elif len(trey) == 1 and trey == 1 and len(cony) == 1 and cony == 0:
                    #     recordATEE.append([name, n, convertReflact[name][n], 1])
                    if ate_s[0] / maxTime > 0.05 and gini == 1:
                        recordATEE.append([name, n, convertReflact[name][n], ate_s[0], ate_s[0] / maxTime])
        recordATEE.sort(key=lambda x: x[3], reverse=True)
        dict[act + ':' + convertReflact['duration'][m]] = recordATEE
    for key in dict.keys():
        print(key)
        for root in dict[key]:
            print("  ",root[0],root[2],root[4])
    np.save(eventlog + "/" + eventlog + 'result_R_S_XGB.npy', dict)#_AresultR
    end = time.time()
    print(end - start)
