import os
import glob
import sys
import pickle
import random 
import numpy as np
import src.data as d


test_bearing = sys.argv[1]
bucket_size = sys.argv[2]
exp_seed_str = sys.argv[3]
exp_seed = int(exp_seed_str)

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


train_bearings = np.setdiff1d(all_bearings, test_bearing).tolist()
X_scaled = d.my_scaler(X, train_bearings)
X_clusters = d.my_cluster(X_scaled, train_bearings, random_state=2023+exp_seed)
transition_times = {}
for b in X.keys():
    transition_times[b] = d.find_transition_time(X_clusters[b])

print(transition_times[test_bearing])

os.makedirs(os.path.join("transition_times", "bucket_size_"+bucket_size, "seed_"+exp_seed_str), exist_ok=True) 
with open(os.path.join("transition_times", "bucket_size_"+bucket_size, "seed_"+exp_seed_str, test_bearing + ".pkl"), 'wb') as f:
    pickle.dump(transition_times, f)

