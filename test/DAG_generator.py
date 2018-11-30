from random import shuffle as sl
from random import randint as rd

def w2f(f, num, fg):
    f.write(str(num))
    if fg == True:
        f.write('\n')
    else:
        f.write(' ')


def DataMake(c):
    f = open('data' + str(c) + '.in', 'w')
    n = 5
    node = range(1, n + 1)
    sl(node)
    sl(node)
    m = rd(1, (n-1) * n/2)
    w2f(f, n, 0)
    w2f(f, m, 1)
    lines = ""
    ln = 0
    for i in range(0, m):
        p1 = rd(1, n - 1)
        p2 = rd(p1 + 1, n)
        x = node[p1 - 1]
        y = node[p2 - 1]
        if lines.find('{0},{1}'.format(x,y)) == -1:
            w2f(f, x, 0)
            w2f(f, y, 1)
            ln = ln + 1
            lines = lines + ';' + '{0},{1}'.format(x,y)
    print n, ' node', ln, ' edges'
    print lines
    f.close()


DataMake(1)
print 'Done'

