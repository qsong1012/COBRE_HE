import sys
import os

study = sys.argv[1]
timestr = sys.argv[2]

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# model training
def batch_train():
    for i in range(5):
        args = []
        for arg in sys.argv[3:]:
            if 'resume=' in arg:
                arg = arg.replace('-0', f'-{i}')
            args.append(arg)
        new_cmd = ' '.join(['python train.py'] + args +
                           [f'--fold={i} --study={study} --timestr={timestr}'])
        print(new_cmd)
        return_code = os.system(new_cmd)
        if return_code:
            return return_code
    return 0


if __name__ == '__main__':

    return_code = batch_train()
    if return_code:
        pass
    else:
        # model testing
        new_cmd = f'python quick_test.py {study} {timestr} {timestr}-test'
        new_cmd += ' --prob-mask=0 --prop-mask=0,1,0 --mlm-loss=null'
        new_cmd = new_cmd.replace(' --override-logs', '')
        print(new_cmd)
        os.system(new_cmd)
