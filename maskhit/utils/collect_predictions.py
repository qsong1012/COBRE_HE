import sys
import pandas as pd
import glob
import re

study = sys.argv[1]
timestr = sys.argv[2]
files = glob.glob(f'logs/{study}/{timestr}*.csv')
files.sort()
files = [x for x in files if re.match('.*-test.*-.*', x)]

res = []
for fold, fname in enumerate(files):
    print(fname)
    res_i = pd.read_csv(fname)
    res_i['fold'] = fold
    res.append(res_i)

df = pd.concat(res)
dfs = df.groupby(['fold']).mean()
dfss1 = pd.DataFrame([dfs.mean().to_dict()])
dfss2 = pd.DataFrame([dfs.std().to_dict()])

dfss1.index = ['avg']
dfss2.index = ['std']

print(pd.concat([dfs, dfss1, dfss2]))
