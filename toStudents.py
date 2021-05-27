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

def data_ready_knn(trainSet, testSet):
    trs = trainNum // 10
    tes = testNum // 10
    
    trainSetf = np.zeros((trainNum, 28*28))
    testSetf = np.zeros((testNum, 28*28))
    for i in range(10):
        for j in range(trs):
            trainSetf[(i * trs) + j] = trainSet[j+(i*trs)].flatten()

    for i in range(10):
        for j in range(tes):
            testSetf[(i * tes) + j] = testSet[j+(i*tes)].flatten()
    return trainSetf, testSetf

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
    result = np.zeros((testNum//10,10)) #100 x 10
    for i in range(10): # 10
        for j in range(testNum//10): # 1000
            imsiTest = np.tile(testSet[j+i*(testNum//10)], (1,10))

            error = np.abs(tmpl - imsiTest) #6000x28x28
            errorSum = [error[:,0:28].sum(), error[:,28:56].sum(),\
                        error[:,56:84].sum(), error[:,84:112].sum(),\
                        error[:,112:140].sum(), error[:,140:168].sum(),\
                        error[:,168:196].sum(), error[:,196:224].sum(),\
                        error[:,224:252].sum(), error[:,252:280].sum(),]
            result[j,i] = np.argmin(errorSum)
    return result

def knn(trainSet, testSet, k): 
    trS1,trS2 = trainSet.shape 
    teS1,teS2 = testSet.shape

    trS3 = int(trS1/10) 
    teS3 = int(teS1/10)

    label = np.tile(np.arange(0,10), (teS3,1)) 
    result = np.zeros((teS3,10))

    for i in range(teS1): 
        imsi = np.sum((trainSet - testSet[i,:])**2,axis=1) 
        no = np.argsort(imsi)[0:k] 
        hist, bins = np.histogram(no//trS3, np.arange(-0.5,10.5,1))
        result[i%teS3, i//teS3] = np.argmax(hist) 
    return result

def feat1(trainSet, testSet):
    trS1 = 10; trS2 = trainNum // 10 #600
    teS1 = 10; teS2 = testNum // 10 #100

    trainSetf = np.zeros((trS1 * trS2, 5))
    testSetf = np.zeros((teS1 * teS2, 5))

    for i in range(trS1):
        for j in range(trS2):
            imsi = trainSet[j+(i*trS2)]
            imsi = np.where(imsi != 0)
            imsi2 = np.mean(imsi,1)
            imsi3 = np.cov(imsi)
            trainSetf[j + (i*trS2)] = np.array([imsi2[0], imsi2[1], imsi3[0,0],imsi3[0,1],imsi3[1,1]])

    for i in range(teS1):
        for j in range(teS2):
            imsi = testSet[j+(i*teS2)]
            imsi = np.where(imsi != 0)
            imsi2 = np.mean(imsi,1)
            imsi3 = np.cov(imsi)
            testSetf[j + (i*teS2)] = np.array([imsi2[0],imsi2[1],imsi3[0,0],imsi3[0,1],imsi3[1,1]])
    return trainSetf, testSetf

def feat2(trainSet, testSet, dX):
    size = trainSet[0].shape[0]; s = size - dX+1
    
    trS1 = 10; trS2 = trainNum // 10
    teS1 = 10; teS2 = testNum  // 10
    
    trainImsi = np.zeros((trS1 * trS2, s, s)); testImsi = np.zeros((teS1 * teS2, s,s))
    trainSetf = np.zeros((trS1 * trS2, s*s)); testSetf = np.zeros((teS1 * teS2, s*s))

    for i in range(10):
        for j in range(trainNum): # 600
            imsi = trainSet[j+(i*trS2)]
            for ii in range(s):
                for jj in range(s):
                    trainImsi[i*(trS2+j), ii, jj] = imsi[ii:dX+ii , jj:dX+jj].sum()
            trainSetf[j+(i*trS2),:] = trainImsi[j+(i*trS2), :].flatten()

    for i in range(10):
        for j in rangee(testNum):
            imsi = testSet[j+(i*teS2)]
            for ii in range(s):
                for jj in range(s):
                    testImsi[i*teS2+j, ii, jj] = imsi[ii:dX+ii,jj:dX+jj].sum()
            testSetf[i*teS2+j,:] = testImsi[i*teS2+j, ::].flatten()

    return trainSetf, testSetf
################################## main ################################

x_train, y_train, x_test, y_test = init_data()
x_train2, y_train2, x_test2, y_test2 = data_ready(x_train, y_train, x_test, y_test)
trainSetf, testSetf = feat2(x_train2, x_test2,10)
result = knn(trainSetf, testSetf, 10)




