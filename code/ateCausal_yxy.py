
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

EL = ['hd','BPIC2017','Sepsis','CoSeLoG','BPIC2015_3']#

for eventlog in EL:
    print(eventlog)
    start = time.time()
    # 读取文件
    convertReflact = np.load(eventlog+'/'+eventlog+'_header.npy', allow_pickle='TRUE').item()
    # 案例级别属性评估ATE
    dataDF = pd.read_csv(eventlog + '/' + eventlog + '_Case.csv')
    dataNumpy = dataDF.to_numpy()
    dict = {}
    recordATEC = []
    y = dataNumpy[:, -2]
    yc = dataNumpy[:, -1]
    maxTime = y.max() - y.min()
    # 事件执行时间评估ATE
    dataDF = pd.read_csv(eventlog + '/' + eventlog + '_duration.csv')
    dataNumpy = dataDF.to_numpy()
    y = dataNumpy[:, -2]
    yc = dataNumpy[:, -1]
    i = list(dataDF.keys()).index('duration')
    treatment = dataNumpy[:, i]
    X = dataNumpy[:, [j for j in range(len(dataDF.keys()) - 2) if j != i]]
    k = np.unique(treatment)
    for act in convertReflact['concept:name'].values():
        recordATEE = []
        # 事件级别属性评估ATE
        if os.path.exists(eventlog + '/' + eventlog + '_' + act + '.csv'):
            dataDF = pd.read_csv(eventlog + '/' + eventlog + '_' + act + '.csv')
        else:
            continue
        dataNumpy3 = dataDF.to_numpy()
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
                    ate_s = learner_dr.estimate_ate(X=X3, treatment=treatment2, y=y3)
                    gini = np.load('gini.npy', allow_pickle='TRUE').item()
                    trey = list(set([yi for tre, yi in zip(treatment2, y3c) if tre == 'treatment_a']))
                    cony = list(set([yi for tre, yi in zip(treatment2, y3c) if tre == 'control']))
                    if len(trey) > 1 and len(cony) > 1:
                        learner_c = BaseSClassifier(XGBClassifier(), control_name='control')  # , XGBRegressor()
                        y3c = y3c.astype(int)
                        ate_c = learner_c.estimate_ate(X=X3, treatment=treatment2, y=y3c)
                        if ate_c[0] > 0.05:
                            recordATEE.append([name, n, convertReflact[name][n], ate_c[0]])
                    elif len(trey) == 1 and trey == 1 and len(cony) == 1 and cony == 0:
                        recordATEE.append([name, n, convertReflact[name][n], 1])
                    if ate_s[0] / maxTime > 0.05 and gini == 1:
                        recordATEE.append([name, n, convertReflact[name][n], ate_s[0], ate_s[0] / maxTime])
            recordATEE.sort(key=lambda x: x[3], reverse=True)
            dict[act] = recordATEE