import os
import glob
import sys
import pickle
import random 
import numpy as np
import src.data as d
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


test_bearing = sys.argv[1]
bucket_size = sys.argv[2]
exp_seed_str = sys.argv[3]
exp_seed = int(exp_seed_str)
scenario = sys.argv[4]

#Set the Python seed and the NumPy seed
os.environ['PYTHONHASHSEED']=str(exp_seed)
random.seed(exp_seed)
np.random.seed(exp_seed)

all_bearings = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17']


path = os.getcwd()
bearings_X = glob.glob(os.path.join("data", "LDM", "fft", "*")) 

X_200 = {} #when we have 200 features
y = {}
for b_adr in bearings_X:
    b_name = b_adr[-7:-4]
    X_200[b_name] = np.load(b_adr)
    y[b_name] = d.rul_calculator(X_200[b_name].shape[0])

if int(bucket_size) == 200:
    X = X_200
else:
    X = {}
    for b in X_200.keys():
        X[b] = np.array([np.max(t, axis=1) for t in np.split(X_200[b], int(bucket_size), axis=1)]).transpose()


bearings_op = glob.glob(os.path.join("data", "LDM", "op", "*")) 
op = {}
for b_adr in bearings_op:
    data = pd.read_csv(b_adr , delim_whitespace=False)
    op[b_adr[-7:-4]] = data.values[:,1:]

# If op and x sizes does not match 
for b in X.keys():
    if op[b].shape[0] != X[b].shape[0]:
        # print(b)
        # print("length before modifying", op[b].shape[0])
        op[b] = op[b][:X[b].shape[0],:]
        # print("length after modifying", op[b].shape[0])



with open(os.path.join("transition_times", "bucket_size_20", "seed_0", test_bearing + ".pkl"), 'rb') as f:
    transition_times = pickle.load(f)

train_bearings = np.setdiff1d(all_bearings, test_bearing).tolist()


if scenario == "op_ignore":
    X_scaled = d.my_scaler(X, train_bearings)

elif scenario == "op_norm":
    n_bins = 20
    X_scaled = d.op_scaler(X, op, train_bearings, n_bins=n_bins, random_state=exp_seed)
    scenario = os.path.join(scenario, "bin_number_"+str(n_bins))

elif scenario == "op_feature":
    X_op = {}
    for b in X.keys():
            X_op[b] = np.hstack((X[b], op[b]))
    X_scaled = d.my_scaler(X_op, train_bearings)




X_healthy, X_unhealthy, y_healthy, y_unhealthy = d.data_cutter(X_scaled, y, transition_times)
X_train, X_test, y_train, y_test = d.my_train_test_split(X_unhealthy, y_unhealthy, train_bearings, test_bearing)
est = HistGradientBoostingRegressor(random_state=2000+exp_seed)
est.fit(X_train, y_train)
y_hat_test = est.predict(X_test)
print(mean_absolute_error(y_hat_test, y_test))

os.makedirs(os.path.join("all_yhats", type(est).__name__, test_bearing, "bucket_size_"+bucket_size, scenario, "seed_"+exp_seed_str), exist_ok=True) 
np.save(os.path.join("all_yhats", type(est).__name__, test_bearing, "bucket_size_"+bucket_size, scenario, "seed_"+exp_seed_str, "y_hat"), y_hat_test)