import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from xgboost import XGBRegressor
import warnings

from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
from causalml.match import NearestNeighborMatch, MatchOptimizer, create_table_one
from causalml.propensity import ElasticNetPropensityModel
from causalml.dataset import *
from causalml.metrics import *

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

import causalml
print(causalml.__version__)

# # Generate synthetic data using mode 1
# y, X, treatment, tau, b, e = synthetic_data(mode=1, n=10000, p=8, sigma=1.0)
#
# # Ready-to-use S-Learner using LinearRegression
# learner_s = LRSRegressor()
# ate_s = learner_s.estimate_ate(X=X, treatment=treatment, y=y)
# print(ate_s)
# print('ATE estimate: {:.03f}'.format(ate_s[0][0]))
# print('ATE lower bound: {:.03f}'.format(ate_s[1][0]))
# print('ATE upper bound: {:.03f}'.format(ate_s[2][0]))
#
# # After calling estimate_ate, add pretrain=True flag to skip training
# # This flag is applicable for other meta learner
# ate_s = learner_s.estimate_ate(X=X, treatment=treatment, y=y, pretrain=True)
# print(ate_s)
# print('ATE estimate: {:.03f}'.format(ate_s[0][0]))
# print('ATE lower bound: {:.03f}'.format(ate_s[1][0]))
# print('ATE upper bound: {:.03f}'.format(ate_s[2][0]))
#
# # Ready-to-use T-Learner using XGB
# learner_t = XGBTRegressor()
# ate_t = learner_t.estimate_ate(X=X, treatment=treatment, y=y)
# print('Using the ready-to-use XGBTRegressor class')
# print(ate_t)
#
# # Calling the Base Learner class and feeding in XGB
# learner_t = BaseTRegressor(learner=XGBRegressor())
# ate_t = learner_t.estimate_ate(X=X, treatment=treatment, y=y)
# print('\nUsing the BaseTRegressor class and using XGB (same result):')
# print(ate_t)
#
# # Calling the Base Learner class and feeding in LinearRegression
# learner_t = BaseTRegressor(learner=LinearRegression())
# ate_t = learner_t.estimate_ate(X=X, treatment=treatment, y=y)
# print('\nUsing the BaseTRegressor class and using Linear Regression (different result):')
# print(ate_t)
#
# # X Learner with propensity score input
# # Calling the Base Learner class and feeding in XGB
# learner_x = BaseXRegressor(learner=XGBRegressor())
# ate_x = learner_x.estimate_ate(X=X, treatment=treatment, y=y, p=e)
# print('Using the BaseXRegressor class and using XGB:')
# print(ate_x)
#
# # Calling the Base Learner class and feeding in LinearRegression
# learner_x = BaseXRegressor(learner=LinearRegression())
# ate_x = learner_x.estimate_ate(X=X, treatment=treatment, y=y, p=e)
# print('\nUsing the BaseXRegressor class and using Linear Regression:')
# print(ate_x)
#
# # X Learner without propensity score input
# # Calling the Base Learner class and feeding in XGB
# learner_x = BaseXRegressor(XGBRegressor())
# ate_x = learner_x.estimate_ate(X=X, treatment=treatment, y=y)
# print('Using the BaseXRegressor class and using XGB without propensity score input:')
# print(ate_x)
#
# # Calling the Base Learner class and feeding in LinearRegression
# learner_x = BaseXRegressor(learner=LinearRegression())
# ate_x = learner_x.estimate_ate(X=X, treatment=treatment, y=y)
# print('\nUsing the BaseXRegressor class and using Linear Regression without propensity score input:')
# print(ate_x)
#
# # R Learner with propensity score input
# # Calling the Base Learner class and feeding in XGB
# learner_r = BaseRRegressor(learner=XGBRegressor())
# ate_r = learner_r.estimate_ate(X=X, treatment=treatment, y=y, p=e)
# print('Using the BaseRRegressor class and using XGB:')
# print(ate_r)
#
# # Calling the Base Learner class and feeding in LinearRegression
# learner_r = BaseRRegressor(learner=LinearRegression())
# ate_r = learner_r.estimate_ate(X=X, treatment=treatment, y=y, p=e)
# print('Using the BaseRRegressor class and using Linear Regression:')
# print(ate_r)
#
# # R Learner with propensity score input and random sample weight
# # Calling the Base Learner class and feeding in XGB
# learner_r = BaseRRegressor(learner=XGBRegressor())
# sample_weight = np.random.randint(1, 3, len(y))
# ate_r = learner_r.estimate_ate(X=X, treatment=treatment, y=y, p=e, sample_weight=sample_weight)
# print('Using the BaseRRegressor class and using XGB:')
# print(ate_r)
#
# # R Learner without propensity score input
# # Calling the Base Learner class and feeding in XGB
# learner_r = BaseRRegressor(learner=XGBRegressor())
# ate_r = learner_r.estimate_ate(X=X, treatment=treatment, y=y)
# print('Using the BaseRRegressor class and using XGB without propensity score input:')
# print(ate_r)
#
# # Calling the Base Learner class and feeding in LinearRegression
# learner_r = BaseRRegressor(learner=LinearRegression())
# ate_r = learner_r.estimate_ate(X=X, treatment=treatment, y=y)
# print('Using the BaseRRegressor class and using Linear Regression without propensity score input:')
# print(ate_r)
#
# # S Learner
# learner_s = LRSRegressor()
# cate_s = learner_s.fit_predict(X=X, treatment=treatment, y=y)
#
# # T Learner
# learner_t = BaseTRegressor(learner=XGBRegressor())
# cate_t = learner_t.fit_predict(X=X, treatment=treatment, y=y)
#
# # X Learner with propensity score input
# learner_x = BaseXRegressor(learner=XGBRegressor())
# cate_x = learner_x.fit_predict(X=X, treatment=treatment, y=y, p=e)
#
# # X Learner without propensity score input
# learner_x_no_p = BaseXRegressor(learner=XGBRegressor())
# cate_x_no_p = learner_x_no_p.fit_predict(X=X, treatment=treatment, y=y)
#
# # R Learner with propensity score input
# learner_r = BaseRRegressor(learner=XGBRegressor())
# cate_r = learner_r.fit_predict(X=X, treatment=treatment, y=y, p=e)
#
# # R Learner without propensity score input
# learner_r_no_p = BaseRRegressor(learner=XGBRegressor())
# cate_r_no_p = learner_r_no_p.fit_predict(X=X, treatment=treatment, y=y)
# alpha=0.2
# bins=30
# plt.figure(figsize=(12,8))
# plt.hist(cate_t, alpha=alpha, bins=bins, label='T Learner')
# plt.hist(cate_x, alpha=alpha, bins=bins, label='X Learner')
# plt.hist(cate_x_no_p, alpha=alpha, bins=bins, label='X Learner (no propensity score)')
# plt.hist(cate_r, alpha=alpha, bins=bins, label='R Learner')
# plt.hist(cate_r_no_p, alpha=alpha, bins=bins, label='R Learner (no propensity score)')
# plt.vlines(cate_s[0], 0, plt.axes().get_ylim()[1], label='S Learner',
#            linestyles='dotted', colors='green', linewidth=2)
# plt.title('Distribution of CATE Predictions by Meta Learner')
# plt.xlabel('Individual Treatment Effect (ITE/CATE)')
# plt.ylabel('# of Samples')
# plt.show()
# _=plt.legend()

train_summary, validation_summary = get_synthetic_summary_holdout(simulate_nuisance_and_easy_treatment,
                                                                  n=10000,
                                                                  valid_size=0.2,
                                                                  k=10)
print(train_summary)
print(validation_summary)
scatter_plot_summary_holdout(train_summary,
                             validation_summary,
                             k=10,
                             label=['Train', 'Validation'],
                             drop_learners=[],
                             drop_cols=[])
# Single simulation
train_preds, valid_preds = get_synthetic_preds_holdout(simulate_nuisance_and_easy_treatment,
                                                       n=50000,
                                                       valid_size=0.2)
#distribution plot for signle simulation of Training
distr_plot_single_sim(train_preds, kind='kde', linewidth=2, bw_method=0.5,
                      drop_learners=['S Learner (LR)',' S Learner (XGB)'])

#distribution plot for signle simulation of Validaiton
distr_plot_single_sim(valid_preds, kind='kde', linewidth=2, bw_method=0.5,
                      drop_learners=['S Learner (LR)', 'S Learner (XGB)'])

# Scatter Plots for a Single Simulation of Training Data
scatter_plot_single_sim(train_preds)

# Scatter Plots for a Single Simulation of Validaiton Data
scatter_plot_single_sim(valid_preds)

# Cumulative Gain AUUC values for a Single Simulation of Training Data
auucTrain = get_synthetic_auuc(train_preds)#, drop_learners=['S Learner (LR)']
print(auucTrain)
# Cumulative Gain AUUC values for a Single Simulation of Validaiton Data
auucValid = get_synthetic_auuc(valid_preds)#, drop_learners=['S Learner (LR)']
print(auucValid)
plt.show()
print('end')