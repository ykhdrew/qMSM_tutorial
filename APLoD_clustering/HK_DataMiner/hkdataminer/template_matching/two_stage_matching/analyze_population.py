###Edit by Hanlin Gu on 1/9/2020
###obtain population of different conformations, 'result$.txt' is the revised version of '2nd_stage_brute_force_classification_result.dat', removed:@
###2nd_stage_brute_force_classification_result.dat has 8 columns, the 1st column is the number, 2nd-7th columns are distance of three conformations in two range(so is 6)
###   8th column is the minmum distance of 2nd-7th distance, 9th column is the conformation number which will assign. 
###Actually here we have three conformations in two range, so is 6 classes, even number classes are right which represent blue range and odd number classes are wrong which
###   represent the green range
import numpy as np
import matplotlib.pyplot as plt


def cal(path):
    data = np.loadtxt(path)
    print(data.shape)
    open = 0
    intermmediate = 0
    close = 0
    distance = []
    distance1 = []
    distance2 = []
    for i in range(data.shape[0]):
        if data[i, 8] == 0:
            open = open + 1
            distance.append(data[i, 7])
        if data[i, 8] == 2:
            distance1.append(data[i, 7])
            intermmediate = intermmediate + 1
        if data[i, 8] == 4:
            distance2.append(data[i, 7])

            close = close + 1

    plt.scatter(np.arange(len(distance)), distance)
    plt.savefig('large_range_distribution.png')
    plt.show()
    print(close)
    sum = open + close + intermmediate
    exp = [float(open / sum), float(intermmediate / sum), float(close / sum)]
    return exp


if __name__ == '__main__':
    path = ['result1.txt', 'result2.txt',
            'result3.txt']
    array = []
    real = [0.4, 0.3, 0.3]

    for i in range(3):
        print(path[i])
        exp = cal(path[i])
        print(exp)
        array.append(exp)
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
        plt.title('propotion comparison')
        plt.plot(np.arange(3), exp, color='y', label='experiment3')
    plt.plot(np.arange(3), real, color='r', label='real')
    plt.legend(loc='upper right')
    
    
    plt.savefig('proportion_comparison')
    mean = np.mean(array, 0)
    rows = ['%d' % x for x in range(3)]
    std = np.std(array, 0)

    plt.cla()
    columns = ('mean', 'std')
    cell_text = np.transpose(np.array([mean, std]))
    table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns, loc='center')
    table.scale(1, 4)
    table.set_fontsize(14)
    plt.axis('off')
    plt.title('three score')
    plt.savefig( 'comparison table')
