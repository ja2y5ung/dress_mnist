import numpy as np
import matplotlib.pyplot as plt

trainNum = 6000
testNum = 1000

def init_data():
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')    
    return x_train, y_train, x_test, y_test

def data_ready(x_train, y_train, x_test, y_test):
    x = np.zeros((trainNum,28,28))
    y = np.zeros(trainNum)
    xx = np.zeros((testNum,28,28))
    yy = np.zeros(testNum)
    for ii in range(10):
        x[ii*(trainNum//10):(ii+1)*(trainNum//10),::] = x_train[y_train==ii,::][0:(trainNum//10)]
        y[ii*(trainNum//10):(ii+1)*(trainNum//10)] = y_train[y_train==ii][0:(trainNum//10)]
        xx[ii*(testNum//10):(ii+1)*(testNum//10),::] = x_test[y_test==ii,::][0:(testNum//10)]
        yy[ii*(testNum//10):(ii+1)*(testNum//10)] = y_test[y_test==ii][0:(testNum//10)]
    x_train2 = x
    y_train2 = y
    x_test2 = xx
    y_test2 = yy

    return x_train2, y_train2, x_test2, y_test2

# 준비된 사진 보는 함수
def print_data(data, row, col, data_num):
    num = row * col
    fig = plt.figure()
    for i in range(1,num+1):
        ax = fig.add_subplot(row,col,i)
        plt.imshow(data[i+data_num])
    plt.show()

def createTmpl(trainSet):
    tmpl = np.zeros((28,28*10))
    for i in range(10):
        imsi = trainSet[(trainNum//10)*i : (trainNum//10)*i+(trainNum//10)]
        tmpl[:,i*28:(i+1)*28] = np.mean(imsi, axis = 0)
    return tmpl

def tmplMatch(tmpl, testSet):
    result = np.zeros((testNum,10))

    for i in range(10):
        for j in range(testNum//10):
            imsiTest = np.tile(testSet[i][j], (1,10))
            error = np.abs(tmpl - imsiTest)
            errorSum = [error[:,0:28].sum(), error[:,28:56].sum(),\
                        error[:,56:84].sum(), error[:,84:112].sum(),\
                        error[:,112:140].sum(), error[:,140:168].sum(),\
                        error[:,168:196].sum(), error[:,196:224].sum(),\
                        error[:,224:252].sum(), error[:,252:280].sum(),]

            result[j,i] = np.argmin(errorSum)
    return result

################################## main ################################

x_train, y_train, x_test, y_test = init_data()
x_train2, y_train2, x_test2, y_test2 = data_ready(x_train, y_train, x_test, y_test)

tmpl = createTmpl(x_train2)
##result = tmplMatch(tmpl,x_test2)

plt.imshow(tmpl); plt.show()







