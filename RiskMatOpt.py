import numpy as np
import scipy.optimize as opt
import warnings


class QuadOptW:
    def __init__(self, quadMat, maxW = 1.0, minW = 0.0, sumW = 1.0, scale = True):
        if quadMat.shape[0] != quadMat.shape[1]:
            raise Exception('error: quadMat must be square matrix!')
        self.size = quadMat.shape[0]

        self.scaleCoef = (1 / np.linalg.norm(quadMat, ord='fro')) if scale else 1
        if not np.isfinite(self.scaleCoef):
            warnings.warn('scale coeficient is not finite, use 1 instead')
            self.scaleCoef = 1
        self.quadMat = quadMat * self.scaleCoef*10
        self.maxW = maxW
        self.minW = minW
        self.sumW = sumW

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
        if self.size == 0:
            infoDict = {'min': np.nan,
                        'minW': np.array([]),
                        'minSuc': 'succ',
                        'minMes': 'empty matrix',
                        'minIte': 0,
                        'max': np.nan,
                        'maxW': np.array([]),
                        'maxSuc': 'succ',
                        'maxMes': 'empty matrix',
                        'maxIte': 0}
            return infoDict

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
                    'minSuc' : resMin.success,
                    'minMes' : resMin.message,
                    'minIte' : resMin.nit,
                    'max'    : -resMax.fun,
                    'maxW'   : resMax.x,
                    'maxSuc' : resMax.success,
                    'maxMes' : resMax.message,
                    'maxIte' : resMax.nit}
        if np.isnan(resMin.fun) or np.isnan(resMax.fun):
            warnings.warn('invalid optimal value, optimizition has failed.')
        return infoDict

class RiskMat:
    rtnFrame = None
    rtnMat = None
    ccmMat = None
    covMat = None
    ccmError = None
    covError = None
    shrinkMat = None
    minSize = 0
    validIndex = None

    def __init__(self, rtnFrame, minSize, validLast = False):
        self.rtnFrame = rtnFrame
        self.minSize = minSize

        # filter invalid data
        self.rtnFrameVld = None
        self.validIndex = []
        for i in range(len(rtnFrame.columns)):
            if sum(np.isfinite(rtnFrame[i].get_values())) > minSize \
               and (not validLast or np.isfinite(rtnFrame.iloc[-1, i])):
                self.validIndex.append(i)
        self.rtnFrameVld = self.rtnFrame[self.validIndex]
        frame = self.rtnFrameVld
        dataLen = frame.shape[0]
        sampleLen = frame.shape[1]
        self.rtnMat = frame.get_values()

        # covariance matrix(covMat) and constant correlation coefficient matrix(ccmMat)
        self.stdVec = frame.std().get_values()
        self.covMat = frame.cov(min_periods=minSize).get_values()
        corMean = 0
        count = 0
        for i in range(sampleLen):
            for j in range(i + 1, sampleLen):
                if self.stdVec[i] > 0 and self.stdVec[j] > 0:
                    corMean += self.covMat[i][j] / (self.stdVec[i] * self.stdVec[j])
                    count += 1
        corMean /= (count + 1) / 2
        self.ccmMat = np.nan * np.zeros([sampleLen, sampleLen], dtype=float)
        for i in range(sampleLen):
            for j in range(i + 1, sampleLen):
                self.ccmMat[i][j] = corMean * self.stdVec[i] * self.stdVec[j]
                self.ccmMat[j][i] = self.ccmMat[i][j]
        for i in range(sampleLen):
            self.ccmMat[i][i] = self.stdVec[i] ** 2

        # estimation error and shrink matrix
        self.covError = 0
        for t in range(dataLen):
            rtnSec = np.nan_to_num(self.rtnMat[t: t + 1])
            covErrorMat = np.dot(rtnSec.T, rtnSec) - np.nan_to_num(self.covMat)
            self.covError += np.linalg.norm(covErrorMat, ord='fro')
        self.covError /= dataLen * (dataLen - 1)
        self.ccmError = np.linalg.norm(np.nan_to_num(self.covMat - self.ccmMat), ord='fro') - self.covError
        self.shrinkMat = (self.covError / (self.covError + self.ccmError)) * self.ccmMat + \
                         (self.ccmError / (self.covError + self.ccmError)) * self.covMat

        self.riskMatDict = {'cov': self.covMat,
                            'ccm': self.ccmMat,
                            'shrink': self.shrinkMat}


    def Optimize(self, matType='cov', minW=0.0, maxW=1.0, crossBorderOrder=5, pca=None):
        '''
        optimize weight vector
        :param matType: risk matrix type, 'cov', 'ccm' or 'shrink'
        :param minW: min weight
        :param maxW: max weight
        :param crossBorderOrder: cross border error limit
        :param pca: if pca is not None, eigen decomposition and Principal component analysis will be
                    applied to risk matrix, information proportion threshold in  will be set by this
                    parameter, namely, pca. And if pca is None, risk matrix will not be adjusted.
        :return: min weight array, max weight array, min value, max value
        '''
        if matType not in self.riskMatDict.keys():
            raise Exception('unknown risk matrix type!')
        riskMat = np.nan_to_num(self.riskMatDict[matType]) # nan_to_num is important and necessary!
        if pca is not None:
            riskMat = self.StatsRiskMat(pca, matType, minW, maxW, crossBorderOrder)
        crossBorder = (minW + maxW) / 2 * 0.1 ** crossBorderOrder

        sampleLen = len(self.rtnFrame.columns)
        optmizer = QuadOptW(riskMat, maxW=maxW, minW=minW, sumW = 1.0, scale = True)
        optResInfo = optmizer.Resolve()
        optMinWArr = optResInfo['minW']
        optMaxWArr = optResInfo['maxW']
        minWArr = np.nan * np.zeros([sampleLen])
        maxWArr = np.nan * np.zeros([sampleLen])
        for w in range(len(optMinWArr)):
            index = self.validIndex[w]
            if optMinWArr[w] < minW - crossBorder:
                warnings.warn('invalid minW ' + str(optMinWArr[w]) + ', index is ' + str(index) + '/' + str(sampleLen))
                minWArr[index] = np.nan
            else:
                minWArr[index] = optMinWArr[w]
        for w in range(len(optMaxWArr)):
            index = self.validIndex[w]
            if optMaxWArr[w] > maxW + crossBorder:
                warnings.warn('invalid maxW ' + str(optMaxWArr[w]) + ', index is ' + str(index) + '/' + str(sampleLen))
                maxWArr[index] = np.nan
            else:
                maxWArr[index] = optMaxWArr[w]
        return minWArr, maxWArr, optResInfo['min'], optResInfo['max']

    def StatsRiskMat(self, pca=0.85, matType='cov', minW=0.0, maxW=1.0, crossBorderOrder=5):
        if matType not in self.riskMatDict.keys():
            raise Exception('unknown risk matrix type!')
        riskMat = np.nan_to_num(self.riskMatDict[matType]) # nan_to_num is important and necessary!

        eigVal, eigVec = np.linalg.eig(riskMat)
        typeStr = str(eigVec.dtype)
        if typeStr[0 : min(len(typeStr), 5)] != 'float':  # complex eigenvector, ignore imaginary part.
            warnings.warn('np.linalg.eig() return some complex eigenvectors, imaginary part will be ignored.')
            eigVec = eigVec.real
            eigVal = eigVal.real
        evSort = np.argsort(-eigVal)  # sort eigenvalue descendingly.
        eigVal = eigVal[evSort]
        eigVec = eigVec[:, evSort]
        for i in range(len(eigVal)):
            if eigVal[i] < 0:  # negative eigenvalue, replace it by zero.
                warnings.warn('np.linalg.eig() return a negative eigenvalue ' + str(eigVal[i]) +
                              ', and it will be replaced by zero.')
                eigVal[i] = 0

        eigValSum = np.sum(eigVal)
        eigValPcaSum = 0
        pcaEigVal = np.zeros([len(eigVal)], dtype=float)
        for i in range(len(eigVal)):
            eigValPcaSum += eigVal[i]
            pcaEigVal[i] = eigVal[i]
            if eigValPcaSum >= pca * eigValSum:
                break

        statRiskMat = np.dot(np.dot(eigVec, np.diag(pcaEigVal)), eigVec.T)
        return statRiskMat

# for test =======================
# import random
# import pandas
# shape = [100, 50]
# dataArr = np.nan * np.zeros(shape, dtype=float)
# for i in range(shape[0]):
#     for j in range(shape[1]):
#         dataArr[i][j] = random.uniform(-1, 1)
# dataFrm = pandas.DataFrame(dataArr)
# riskMat = RiskMat(dataFrm, shape[0] * 0.5, validLast=True)
# riskMat.Optimize(matType='cov', minW=0, maxW=0.1, crossBorderOrder=5, pca=0.85)

