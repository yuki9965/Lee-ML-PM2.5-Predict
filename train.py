
import pandas as pd
import numpy as np
"""
数据处理
"""
data = pd.read_csv("train.csv")
pm2_5 = data[data['observation'] == 'PM2.5'].ix[:,3:]

# print(pm2_5)
tempxlist = []
tempylist = []

for i in range(15):
    tempx = pm2_5.iloc[:, i:i+9]
    tempx.columns = np.array(range(9))
    tempy = pm2_5.iloc[:, i+9]
    tempy.cloumns = np.array(1)
    tempxlist.append(tempx)
    tempylist.append(tempy)

xdata = pd.concat(tempxlist)
x = np.array(xdata, float)
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1)
print("x's shpae:{}".format(x.shape))

ydata = pd.concat(tempylist)
y = np.array(ydata, float)
print("y's shape:{}".format(y.shape))

# 初始化一个参数矩阵
w=np.zeros((len(x[0])))

#初始化一个learning rate
lr=10
iteration=10000   #迭代10000次
s_grad=np.zeros(len(x[0]))
for i in range(iteration):
    tem=np.dot(x,w)     #&y^*&(预测值)
    loss=y-tem     
    grad=np.dot(x.transpose(),loss)*(-2)
    s_grad+=grad**2
    ada=np.sqrt(s_grad)
    w=w-lr*grad/ada
    if i % 200 == 0:
        print("step:{step},loss = {loss}".format(step = i, loss=loss))

testdata=pd.read_csv('test(1).csv')
pm2_5_test=testdata[testdata['AMB_TEMP']=='PM2.5'].ix[:,2:]
x_test=np.array(pm2_5_test,float)
x_test_b=np.concatenate((np.ones((x_test.shape[0],1)),x_test),axis=1)
y_star=np.dot(x_test_b,w)
y_pre=pd.read_csv('sampleSubmission.csv')
y_pre.value=y_star
y_pre.to_csv('sampleSubmission.csv',index = False)


real=pd.read_csv('answer.csv')
erro=abs(y_pre.value-real.value).sum()/len(real.value)
print("final error = {}".format(erro))
