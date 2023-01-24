import sys
import os
import glob

cancer = sys.argv[1]
timestr = sys.argv[2]
timestr_new = sys.argv[3]

os.chdir(os.path.dirname(os.path.abspath(__file__)))
files = glob.glob(f'logs/{cancer}/{timestr}-*.log')
files = [x for x in files if not 'test' in x.split('-')]
files.sort()

print("Log files found:")
for log_file in files:
    print(log_file)

for i, log_file in enumerate(files):
    with open(log_file, 'r') as f:
        org_cmd = f.readline().rstrip()

    org_cmd = org_cmd.replace("Argument all_arguments:", '').replace("'", "")
    ckp = os.path.basename(log_file.replace('_meta.log', ''))

    new_cmd = ' '.join([
        'python train.py', org_cmd,
        f' --resume={ckp} --mode=test --test-type=test --resume-epoch=BEST --timestr={timestr_new}'
    ] + sys.argv[4:])
    print(new_cmd)
    os.system(new_cmd)
