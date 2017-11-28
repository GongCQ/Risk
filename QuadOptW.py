import numpy as np
import scipy.optimize as opt
import random

def GetStocQuadMat(size, low, up):
    mat = np.zeros([size, size], dtype=float)
    for i in range(size):
        for j in range(size):
            mat[i][j] = random.uniform(low, up)
    return mat

class QuadOptW:
    def __init__(self, quadMat, maxW = 1.0, minW = 0.0, sumW = 1.0):
        if quadMat.shape[0] != quadMat.shape[1]:
            raise Exception('error: quadMat must be square matrix!')
        self.size = quadMat.shape[0]
        self.quadMat = quadMat
        self.maxW = maxW
        self.minW = minW
        self.sumW = sumW
        if not (self.minW <= 1 / self.size <= self.maxW):
            raise Exception('error: the mean of weight must between minW and maxW!')

    def ObjFun(self, w, sign=1):
        return sign * np.dot(np.dot(w, self.quadMat), w.T)

    def DerFun(self, w, sign=1):
        der = np.zeros([self.size])
        for i in range(self.size):
            for j in range(self.size):
                der[i] += self.quadMat[i][j] * w[j]
                der[i] += self.quadMat[j][i] * w[j]
        return sign * der

    def Resolve(self):
        consList = [{'type': 'eq',
                     'fun' : lambda w: np.array([np.sum(w) - self.sumW]),
                     'jac' : lambda w: np.ones([self.size])}]
        for i in range(self.size):
            jacMax = [0] * self.size
            jacMax[i] = -1
            consMaxStr = "{'type': 'ineq', " \
                         " 'fun' : lambda w: np.array([" + str(self.maxW) + " - w[" + str(i) + "]]), " \
                         " 'jac' : lambda w: np.array(" + str(jacMax) + ")} "
            consMax = eval(consMaxStr)
            consList.append(consMax)

            jacMin = [0] * self.size
            jacMin[i] = 1
            consMinStr = "{'type': 'ineq', " \
                         " 'fun' : lambda w: np.array([w[" + str(i) + "] - " + str(self.minW) + "]), "\
                         " 'jac' : lambda w: np.array(" + str(jacMin) + ")} "
            consMin = eval(consMinStr)
            consList.append(consMin)
        constrain = tuple(consList)

        wInit = np.ones([self.size]) / self.size
        resMin = opt.minimize(self.ObjFun, x0 = wInit, args=(1.0,), jac=self.DerFun,
                              constraints = constrain, method = 'SLSQP', options = {'disp': False})
        resMax = opt.minimize(self.ObjFun, x0 = wInit, args=(-1.0,), jac=self.DerFun,
                              constraints = constrain, method = 'SLSQP', options = {'disp': False})
        infoDict = {'min'    : resMin.fun,
                    'minW'   : resMin.x,
                    'minSuc': resMin.success,
                    'minMes' : resMin.message,
                    'minIte' : resMin.nit,
                    'max'    : -resMax.fun,
                    'maxW'   : resMax.x,
                    'maxSuc': resMax.success,
                    'maxMes' : resMax.message,
                    'maxIte' : resMax.nit}
        return infoDict

# for i in range(100):
#     size = 100
#     quadMat = GetStocQuadMat(size, -0.01, 0.01)
#     qo = QuadOptW(quadMat, 3 / size, 0.0)
#     result = qo.Resolve()
#     print(str(result['min']) + ' ' + str(result['max']))
