import DataAdapter
import gongcq.Tools as Tools
import numpy as np
import pandas as pd
import os
import datetime as dt
import RiskMatOpt


class Strategy:
    def __init__(self, codeList, csMap, capacity, mkt, poolCodeList):
        self.csMap = csMap
        self.capacity = capacity
        self.da = DataAdapter.DataAdapter(codeList, csMap, capacity)
        self.poolCodeList = poolCodeList
        self.mkt = mkt
        self.actMatHigh = [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
        self.actMatLow = [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
        self.actMatHighPca = [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
        self.actMatLowPca = [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
        self.actBm = mkt.CreateAccount('[bm]', 100000000, csMap)
        for i in range(4):
            for j in range(3):
                self.actMatHigh[i][j] = mkt.CreateAccount('[h_,' + str(i) + ',' + str(j) + ']', 100000000, csMap)
                self.actMatLow[i][j] = mkt.CreateAccount('[l_,' + str(i) + ',' + str(j) + ']', 100000000, csMap)
                self.actMatHighPca[i][j] = mkt.CreateAccount('[hp,' + str(i) + ',' + str(j) + ']', 100000000, csMap)
                self.actMatLowPca[i][j] = mkt.CreateAccount('[lp,' + str(i) + ',' + str(j) + ']', 100000000, csMap)
        self.lastTrade = 0

    def NewDayHandler(self, mkt):

        self.lastTrade += 1
        if self.lastTrade >= 22:
            self.lastTrade = 0
            print('trade')
        else:
            return

        poolSize = len(self.poolCodeList)

        # 0. evaluate covariance matrix, constant correlation coefficient matrix, and shrink matrix
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
        riskMatList = []
        riskMatList.append(RiskMatOpt.RiskMat(rtnFrm0, int(self.capacity * 0.5), validLast=True))
        riskMatList.append(RiskMatOpt.RiskMat(rtnFrm1, int(self.capacity * 0.5), validLast=True))
        riskMatList.append(RiskMatOpt.RiskMat(rtnFrm2, int(self.capacity * 0.5), validLast=True))
        riskMatList.append(RiskMatOpt.RiskMat(rtnFrm3, int(self.capacity * 0.5), validLast=True))

        # 1. risk optimize and trade
        if not mkt.crtDate >= dt.datetime(2016, 2, 1):
            return
        self.Trade(self.actBm, np.ones([poolSize], dtype=float) / poolSize)
        rmList = ['cov', 'ccm', 'shrink']
        for i in range(len(riskMatList)):
            riskMat = riskMatList[i]
            for j in range(3):
                minW, maxW, minV, maxV = riskMat.Optimize(rmList[j], minW=0, maxW=0.1, crossBorderOrder=5, pca=None)
                print(str(i) + ',' + str(j) + ' ' + str(minV) + ', ' + str(maxV))
                self.Trade(self.actMatHigh[i][j], maxW)
                self.Trade(self.actMatLow[i][j], minW)
                minW, maxW, minV, maxV = riskMat.Optimize(rmList[j], minW=0, maxW=0.1, crossBorderOrder=5, pca=0.85)
                print('pca ' + str(i) + ',' + str(j) + ' ' + str(minV) + ', ' + str(maxV))
                self.Trade(self.actMatHighPca[i][j], maxW)
                self.Trade(self.actMatLowPca[i][j], minW)

        debug = 0

    def Trade(self, account, weight):
        account.ClearAll(comment='clear all')
        for w in range(len(weight)):
            code = self.poolCodeList[w]
            if np.isfinite(weight[w]):
                if weight[w] < 1 / len(self.poolCodeList) / 10:
                    continue
                amount = weight[w] * account.netVal
                transReturn = account.AddDlg(code, None, None, amount, '买',
                                             comment='weight is ' + str(weight[w]), transPoint='c')
                if transReturn != 'SUCCESS':
                    print('fail to buy, symbol is ' + str(self.csMap.GetSymbol(code)) +
                          ', account is ' + str(account.accountID) +
                          ', weight is ' + str(weight[w]) + ', amount is ' + str(amount) + ', transReturn is ' + transReturn)
            # else:
            #     print('weight is not a number, symbol is ' + str(self.csMap.GetSymbol(code)) +
            #           ', account is ' + str(account.accountID))

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

        sh += '=== high risk pca' + os.linesep
        sl += '=== low risk pca' + os.linesep
        for i in range(4):
            for j in range(3):
                sh += str(self.actMatHighPca[i][j].netVal) + ', '
                sl += str(self.actMatLowPca[i][j].netVal) + ', '
            sh += os.linesep
            sl += os.linesep

        return sbm + sh + sl

    def AfterAll(self, mkt):
        for i in range(4):
            for j in range(3):
                self.actMatHigh[i][j].EvalAccount(os.path.join('.', 'eval'), str(self.actMatHigh[i][j].accountID))
                self.actMatLow[i][j].EvalAccount(os.path.join('.', 'eval'), str(self.actMatLow[i][j].accountID))
                self.actMatHighPca[i][j].EvalAccount(os.path.join('.', 'eval'), str(self.actMatHighPca[i][j].accountID))
                self.actMatLowPca[i][j].EvalAccount(os.path.join('.', 'eval'), str(self.actMatLowPca[i][j].accountID))
        self.actBm.EvalAccount(os.path.join('.', 'eval'), str(self.actBm.accountID))
