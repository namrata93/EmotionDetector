import os
import sys
def get_label(name):
    dict1 = {'AN': 1, 'DI': 2, 'FE': 3, 'HA': 4, 'SA': 5, 'SU': 6, 'NE': 7}
    for k in dict1.keys():
        if k in name:
            return str(dict1[k])
    return str(-1)

with open('photo.txt','r') as f:
    names = [line for line in f.readlines()]
    for name in names:
        os.system('/Users/namrataprabhu/face-analysis-sdk/build/bin/face-fit {0} {0}.txt'.format(name))
        try:
            with open('{0}.txt'.format(name),'r') as lm:
                lm_lines = lm.readlines()
                #print(''.join(lm_lines))
                landmarks = [line[:-1].strip().split() for line in lm_lines[2:-1]]
                x_c = [line[0] for line in landmarks]
                y_c = [line[1] for line in landmarks]
                label = get_label(name)
                #if not label == '7':
                print(','.join(x_c+y_c))
        except:
            pass
            #sys.stderr.write(name+'\n')
