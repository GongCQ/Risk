import DataAdapter
import gongcq.Tools as Tools
import numpy as np
import pandas as pd
import QuadOptW
import os
import datetime as dt

class Config:
    def __init__(self, codeList, maxPosPerStock, minPosPerStock, minHoldNum, riskLevel):
        self.codeList = codeList
        self.maxPos = maxPosPerStock
        self.minHoldNum = minHoldNum
        self.riskLevel = riskLevel

class ShrinkMat:
    def __init__(self, rtnFrame, minSize):
        self.rtnFrame = rtnFrame
        self.rtnMat = None
        self.ccmMat = None
        self.covMat = None
        self.ccmError = None
        self.covError = None
        self.shrinkMat = None

        self.rtnFrameVld = None
        self.validIndex = []
        for i in range(len(rtnFrame.columns)):
            if sum(np.isfinite(rtnFrame[i].get_values())) >= minSize:
                self.validIndex.append(i)
        self.rtnFrameVld = self.rtnFrame[self.validIndex]

        self.Update()

    def Update(self):
        frame = self.rtnFrameVld
        dataLen = frame.shape[0]
        sampleLen = frame.shape[1]
        self.rtnMat = frame.get_values()
        self.stdVec = np.nan_to_num(frame.std().get_values())
        self.covMat = np.nan_to_num(frame.cov(min_periods=int(dataLen / 2)).get_values())

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

        self.covError = 0
        for t in range(dataLen):
            rtnSec = np.nan_to_num(self.rtnMat[t : t + 1])
            covErrorMat = np.dot(rtnSec.T, rtnSec) - self.covMat
            self.covError += np.linalg.norm(covErrorMat, ord='fro')
        self.covError /= dataLen * (dataLen - 1)
        self.ccmError = np.linalg.norm(self.covMat - self.ccmMat, ord='fro') - self.covError

        self.shrinkMat = (self.covError / (self.covError + self.ccmError)) * self.ccmMat + \
                         (self.ccmError / (self.covError + self.ccmError)) * self.covMat


class Strategy:
    def __init__(self, codeList, csMap, capacity, mkt, poolCodeList):
        self.csMap = csMap
        self.capacity = capacity
        self.da = DataAdapter.DataAdapter(codeList, csMap, capacity)
        self.poolCodeList = poolCodeList
        self.mkt = mkt
        self.actMatHigh = [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
        self.actMatLow = [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
        self.actBm = mkt.CreateAccount('[bm]', 100000000, csMap)
        for i in range(4):
            for j in range(3):
                self.actMatHigh[i][j] = mkt.CreateAccount('[h,' + str(i) + ',' + str(j) + ']', 100000000, csMap)
                self.actMatLow[i][j] = mkt.CreateAccount('[l,' + str(i) + ',' + str(j) + ']', 100000000, csMap)
        self.lastTrade = 0

    def NewDayHandler(self, mkt):

        self.lastTrade += 1
        if self.lastTrade >= 22:
            self.lastTrade = 0
            print('trade')
        else:
            return

        poolSize = len(self.poolCodeList)

        # 0. evaluate covariance matrix, constant correlation-coefficient matrix, and shrink matrix
        rtnFrm0 = pd.DataFrame(np.array(self.capacity * [[]], dtype = float))
        rtnFrm1 = pd.DataFrame(np.array(self.capacity * [[]], dtype = float))
        rtnFrm2 = pd.DataFrame(np.array(self.capacity * [[]], dtype = float))
        rtnFrm3 = pd.DataFrame(np.array(self.capacity * [[]], dtype = float))
        for c in range(poolSize):
            code = self.poolCodeList[c]
            seq = mkt.dsList[0].GetSeq(code)
            stock = self.da.stockList[seq]
            rtnFrm0[c] = stock.rtnResArr0
            rtnFrm1[c] = stock.rtnResArr1
            rtnFrm2[c] = stock.rtnResArr2
            rtnFrm3[c] = stock.rtnResArr3

        shrinkMatList = [None, None, None, None]
        shrinkMatList[0] = ShrinkMat(rtnFrm0, int(self.capacity / 2))
        shrinkMatList[1] = ShrinkMat(rtnFrm1, int(self.capacity / 2))
        shrinkMatList[2] = ShrinkMat(rtnFrm2, int(self.capacity / 2))
        shrinkMatList[3] = ShrinkMat(rtnFrm3, int(self.capacity / 2))

        # 1. risk optimize and trade
        if not mkt.crtDate >= dt.datetime(2016, 6, 1):
            return
        riskMatMat = [[shrinkMatList[0].covMat, shrinkMatList[0].ccmMat, shrinkMatList[0].shrinkMat],
                      [shrinkMatList[1].covMat, shrinkMatList[1].ccmMat, shrinkMatList[1].shrinkMat],
                      [shrinkMatList[2].covMat, shrinkMatList[2].ccmMat, shrinkMatList[2].shrinkMat],
                      [shrinkMatList[3].covMat, shrinkMatList[3].ccmMat, shrinkMatList[3].shrinkMat]]
        self.Trade(self.actBm, np.ones([poolSize], dtype=float) / poolSize, list(range(poolSize)))
        for i in range(4):
            for j in range(3):
                if len(riskMatMat[i][j]) == 0:
                    continue
                optimizer = QuadOptW.QuadOptW(100000 * riskMatMat[i][j], maxW = 0.1, minW = 0)
                optResultDict = optimizer.Resolve()
                print(str(i) + ',' + str(j) + ' ' + str(optResultDict['min']) + ', ' + str(optResultDict['max']))

                wArrHigh = optResultDict['maxW']
                self.Trade(self.actMatHigh[i][j], wArrHigh, shrinkMatList[i].validIndex)
                wArrLow = optResultDict['minW']
                self.Trade(self.actMatLow[i][j], wArrLow, shrinkMatList[i].validIndex)

        debug = 0

    def Trade(self, account, weight, wVld):
        weightAll = [0] * len(self.poolCodeList)
        for i in range(len(wVld)):
            weightAll[wVld[i]] = weight[i]
        account.ClearAll(comment='clear all')
        for w in range(len(weightAll)):
            if np.isfinite(weightAll[w]):
                code = self.poolCodeList[w]
                if weightAll[w] < 1 / len(self.poolCodeList) / 10:
                    continue
                amount = weightAll[w] * account.netVal
                transReturn = account.AddDlg(code, None, None, amount, '买',
                                             comment='weight is ' + str(weightAll[w]), transPoint='c')
                if not transReturn:
                    print('fail to buy, symbol is ' + str(self.csMap.GetSymbol(code)) +
                          ', account is ' + str(account.accountID) +
                          ', weight is ' + str(weightAll[w]) + ', amount is ' + str(amount))

    def NetValStr(self):
        sbm = '=== benchmark: ' + os.linesep
        sbm += str(self.actBm.netVal) + ', ' + os.linesep
        sh = '=== high risk:' + os.linesep
        sl = '=== low  risk:' + os.linesep
        for i in range(4):
            for j in range(3):
                sh += str(self.actMatHigh[i][j].netVal) + ', '
                sl += str(self.actMatLow[i][j].netVal) + ', '
            sh += os.linesep
            sl += os.linesep
        return sbm + sh + sl

    def AfterAll(self, mkt):
        for i in range(4):
            for j in range(3):
                self.actMatHigh[i][j].EvalAccount(os.path.join('.', 'eval'), str(self.actMatHigh[i][j].accountID))
                self.actMatLow[i][j].EvalAccount(os.path.join('.', 'eval'), str(self.actMatLow[i][j].accountID))
        self.actBm.EvalAccount(os.path.join('.', 'eval'), str(self.actBm.accountID))
