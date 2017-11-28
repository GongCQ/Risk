import datetime as dt
import gongcq.CircSeries as cs
import numpy as np
import gongcq.Tools as Tools
import redis
import gongcq.Public as Public
import os

class ScaleIndex:
    def __init__(self, code, name, capacity):
        self.code = code
        self.name = name
        self.capacity = capacity
        self.dateSrs = cs.CircSeriesObj(capacity)
        self.openSrs = cs.CircSeriesNum(capacity)
        self.closeSrs = cs.CircSeriesNum(capacity)
        self.rtnSrs = cs.CircSeriesNum(capacity)
        self.mktRtnSrs = cs.CircSeriesNum(capacity)
        self.valRtnSrs = cs.CircSeriesNum(capacity)
        self.btmRtnSrs = cs.CircSeriesNum(capacity)

    def AppendData(self, date, openPrice, closePrice, rtn, mktRtn, aveRtn, btmRtn):
        self.dateSrs.Append(date)
        self.openSrs.Append(openPrice)
        self.closeSrs.Append(closePrice)
        self.rtnSrs.Append(rtn)
        self.mktRtnSrs.Append(mktRtn)
        self.valRtnSrs.Append(rtn - aveRtn)
        self.btmRtnSrs.Append(btmRtn)

class Stock:
    def __init__(self, code, symbol, name, mktCode, capacity):
        self.code = code
        self.symbol = symbol
        self.name = name
        self.mktCode = mktCode
        self.capacity = capacity
        self.firstDay = dt.datetime.max
        self.dateSrs = cs.CircSeriesObj(capacity)
        self.openSrs = cs.CircSeriesNum(capacity)
        self.closeSrs = cs.CircSeriesNum(capacity)
        self.rtnSrs = cs.CircSeriesNum(capacity)
        self.tovSrs = cs.CircSeriesNum(capacity)
        self.tvSrs = cs.CircSeriesNum(capacity)
        self.cvSrs = cs.CircSeriesNum(capacity)
        self.nvSrs = cs.CircSeriesNum(capacity)
        self.btmSrs = cs.CircSeriesNum(capacity)
        self.scale = -1
        self.chaVal = np.nan
        self.chaValQ = np.nan
        self.chaValP = np.nan
        self.adjTov = np.nan
        self.adjTovQ = np.nan
        self.adjTovP = np.nan
        self.beta = np.nan
        self.betaQ = np.nan
        self.betaP = np.nan
        self.rtnResArr0 = np.nan * np.zeros([capacity])
        self.rtnResArr1 = np.nan * np.zeros([capacity])
        self.rtnResArr2 = np.nan * np.zeros([capacity])
        self.rtnResArr3 = np.nan * np.zeros([capacity])

        self.chaValQG = np.nan
        self.chaValPG = np.nan
        self.betaQG = np.nan
        self.betaPG = np.nan

        self.bhv = np.nan
        self.bhvQ = np.nan

    def AppendData(self, date, openPrice, closePrice, rtn, tov, tv, cv, nv, scale):
        '''
        添加数据
        :param date: 日期
        :param openPrice: 开盘价
        :param closePrice: 收盘价
        :param tov: 换手率
        :param tv: 总市值
        :param cv: 流通市值
        :param nv: 净资产
        :param scale: 规模（0,1,2,3）
        :return:
        '''
        nv = nv if np.isfinite(nv) else self.nvSrs.GetLast()
        self.dateSrs.Append(date)
        self.openSrs.Append(openPrice)
        self.closeSrs.Append(closePrice)
        self.rtnSrs.Append(rtn)
        self.tovSrs.Append(tov)
        self.tvSrs.Append(tv)
        self.cvSrs.Append(cv)
        self.nvSrs.Append(nv)
        self.btmSrs.Append(nv / tv)
        self.scale = scale
        self.firstDay = date if np.isfinite(closePrice) and date < self.firstDay else self.firstDay

class StockGroup:
    def __init__(self, index, scale):
        self.index = index
        self.scale = scale
        self.stockList = []
        self.chaValSortIndex = []
        self.betaSortIndex = []

    def AppendStock(self, stock):
        self.stockList.append(stock)

    def ClearStock(self):
        self.stockList.clear()


class DataAdapter:
    def __init__(self, codeList, csMap, capacity):
        self.capacity = capacity
        self.csMap = csMap
        self.stockList = [None] * len(codeList)
        for i in range(len(codeList)):
            symbol = csMap.GetSymbol(codeList[i])
            name = csMap.GetNameByCode(codeList[i])
            mktCode = csMap.GetMktByCode(codeList[i])
            self.stockList[i] = Stock(codeList[i], symbol, name, mktCode, capacity)
        self.stockGroup = [None] * 4
        self.stockGroup[0] = StockGroup(ScaleIndex(2060002285, '中证100', capacity), 0)
        self.stockGroup[1] = StockGroup(ScaleIndex(2060002287, '中证200', capacity), 1)
        self.stockGroup[2] = StockGroup(ScaleIndex(2060002289, '中证500', capacity), 2)
        self.stockGroup[3] = StockGroup(ScaleIndex(2060005124, '中证1000', capacity), 3)
        self.chaValSortIndex = np.array(list(range(len(self.stockList))))
        self.betaSortIndex = np.array(list(range(len(self.stockList))))
        self.adjTovSortIndex = np.array(list(range(len(self.stockList))))

    def NewDayHandler(self, mkt):
        # 读取参数
        try:
            scaleStr = Tools.GetPara(0, 'scaleList')
            scaleList = scaleStr.split(',')
            for c in range(len(scaleList)):
                scaleList[c] = int(scaleList[c])
        except Exception as e:
            scaleList = [0, 1, 2, 3]
        try:
            afterIpo = int(Tools.GetPara(1, 'afterIpo'))
        except Exception as e:
            afterIpo = 90

        # 0.股票数据
        for stock in self.stockList:
            # 0.0.日行情
            priceRecord = mkt.dsList[0].GetRecord(stock.code)
            if priceRecord is not None:
                openPrice = priceRecord[6] if priceRecord[6] is not None else np.nan
                closePrice = priceRecord[3] if priceRecord[3] is not None else np.nan
                rtn = priceRecord[4] if priceRecord[4] is not None else np.nan
                tov = priceRecord[11] if priceRecord[11] is not None else np.nan
                tv = priceRecord[7] if priceRecord[7] is not None else np.nan
                cv = priceRecord[8] if priceRecord[8] is not None else np.nan
            else:
                openPrice = np.nan
                closePrice = np.nan
                rtn = np.nan
                tov = np.nan
                tv = np.nan
                cv = np.nan
            # 0.1.净资产
            nvRecord = mkt.dsList[2].GetRecord(stock.code)
            if nvRecord is not None:
                nv = nvRecord[3] if nvRecord[3] is not None else np.nan
            else:
                nv = np.nan
            # 0.2.规模
            scaleRecord = mkt.dsList[3].GetRecord(stock.code)
            if scaleRecord is not None:
                scale = scaleRecord[2] if scaleRecord[2] is not None else -1
            else:
                scale = -1
            # 0.3.添加数据
            stock.AppendData(mkt.crtDate, openPrice, closePrice, rtn, tov, tv, cv, nv, scale)
            
        # 1.股票重新分组
        for group in self.stockGroup:
            group.ClearStock()
        for stock in self.stockList:
            if stock.scale >= 0:
                self.stockGroup[stock.scale].AppendStock(stock)
                
        # 2.指数数据
        # 2.0.统计所有规模指数的加权平均收益率，作为市场收益率，权重为规模指数所包含的股票的数量
        priceRecordSet = mkt.dsList[1].data
        try:
            mktRtn = (priceRecordSet[0][4] * 1 + priceRecordSet[1][4] * 2 +
                      priceRecordSet[2][4] * 5 + priceRecordSet[3][4] * 10 ) / (1 + 2 + 5 + 10)
            aveRtn = (priceRecordSet[0][4] * 1 + priceRecordSet[1][4] * 1 +
                      priceRecordSet[2][4] * 1 + priceRecordSet[3][4] * 1 ) / 4
        except Exception as e:
            mktRtn = 0
            aveRtn = 0
        # 2.1.整理指数数据，其中各规模指数所对应的市值收益率为该规模指数收益率减市场收益率
        groupData = []
        for group in self.stockGroup:
            index = group.index
            # 整理指数日行情数据
            priceRecord = mkt.dsList[1].GetRecord(index.code)
            if priceRecord is not None:
                openPrice = priceRecord[3] if priceRecord[3] is not None else np.nan
                closePrice = priceRecord[2] if priceRecord[2] is not None else np.nan
                rtn = priceRecord[4] if priceRecord[4] is not None else np.nan
            else:
                openPrice = np.nan
                closePrice = np.nan
                rtn = np.nan
            # 准备要添加的数据
            groupData.append([mkt.crtDate, openPrice, closePrice, rtn, mktRtn, aveRtn, np.nan])
            # index.AppendData(mkt.crtDate, openPrice, closePrice, rtn, mktRtn, aveRtn, btmRtn)

        # 计算估值收益率
        btmArr = np.nan * np.zeros([len(self.stockList)])
        for s in range(len(self.stockList)):
            stock = self.stockList[s]
            if (mkt.crtDate - stock.firstDay).days < afterIpo or stock.scale < 0:
                continue
            btmArr[s] = stock.btmSrs.GetLast()
        vldNum = sum(np.isfinite(btmArr))
        if vldNum <= 1800 / 2:
            btmRtn = np.nan
        else:
            btmSortIndex = np.argsort(btmArr)
            btmTop = 0
            btmBot = 0
            for b in range(int(vldNum / 3)):
                rtnTop = self.stockList[btmSortIndex[b]].rtnSrs.GetLast()
                rtnBot = self.stockList[btmSortIndex[vldNum - 1 - b]].rtnSrs.GetLast()
                btmTop += rtnTop if np.isfinite(rtnTop) else 0
                btmBot += rtnBot if np.isfinite(rtnBot) else 0
            btmRtn = (btmTop - btmBot) / (int(vldNum / 3) + 1)
        for g in range(len(self.stockGroup)):
            groupData[g][6] = btmRtn
            self.stockGroup[g].index.AppendData(groupData[g][0], groupData[g][1], groupData[g][2],
                                          groupData[g][3], groupData[g][4], groupData[g][5], groupData[g][6])

        # 3.特异度
        # 3.0.计算各股票的特异度
        vldCountTotal = 1
        for group in self.stockGroup:
            # 三因子回归
            mktRtnArr = group.index.mktRtnSrs.series
            valRtnArr = group.index.valRtnSrs.series
            btmRtnArr = group.index.btmRtnSrs.series
            constArr = np.ones(len(mktRtnArr))
            vldMkt = np.isfinite(mktRtnArr)
            vldVal = np.isfinite(valRtnArr)
            vldBtm = np.isfinite(btmRtnArr)
            chaValArr = np.nan * np.zeros(len(group.stockList))
            betaArr = np.nan * np.zeros(len(group.stockList))
            vldCount = 1
            for s in range(len(group.stockList)):
                stock = group.stockList[s]
                stkRtnArr = stock.rtnSrs.series
                vldStk = np.isfinite(stkRtnArr)
                vld = np.array(vldStk) * np.array(vldMkt) * np.array(vldVal) * np.array(vldBtm)
                if sum(vld) < self.capacity * 0.75 or (mkt.crtDate - stock.firstDay).days < afterIpo:
                    stock.chaVal = np.nan
                    stock.beta = np.nan
                    stock.rtnResArr3 = np.nan * np.zeros([self.capacity])
                    stock.rtnResArr2 = np.nan * np.zeros([self.capacity])
                    stock.rtnResArr1 = np.nan * np.zeros([self.capacity])
                    stock.rtnResArr0 = stkRtnArr
                else:
                    # ........................................
                    # Eugene F. Fama 三因子回归
                    X3 = np.vstack([mktRtnArr[vld], valRtnArr[vld], btmRtnArr[vld], constArr[vld]]).T
                    y3 = stkRtnArr[vld]
                    c3, res3, rank3, sv3 = np.linalg.lstsq(X3, y3)
                    stock.chaVal = res3[0] / np.sum(y3 * y3)
                    stock.beta = (abs(c3[0]) + abs(c3[1]) + abs(c3[2])) / 3
                    chaValArr[s] = stock.chaVal
                    betaArr[s] = stock.beta
                    vldCount += 1
                    stock.rtnResArr3 = c3[0] * mktRtnArr + c3[1] * valRtnArr + c3[2] * btmRtnArr + c3[3] * constArr
                    # ........................................
                    # 市场 + 市值 因子回归
                    X2 = np.vstack([mktRtnArr[vld], valRtnArr[vld], constArr[vld]]).T
                    y2 = stkRtnArr[vld]
                    c2, res2, rank2, sv2 = np.linalg.lstsq(X2, y2)
                    stock.rtnResArr2 = c2[0] * mktRtnArr + c2[1] * valRtnArr + c2[2] * constArr
                    # ........................................
                    # 市场收益率单因子回归
                    X1 = np.vstack([mktRtnArr[vld], constArr[vld]]).T
                    y1 = stkRtnArr[vld]
                    c1, res1, rank1, sv1 = np.linalg.lstsq(X1, y1)
                    stock.rtnResArr1 = c1[0] * mktRtnArr + c1[1] * constArr
                    # ........................................
                    # 股票自身收益率
                    stock.rtnResArr0 = stkRtnArr
                    # ........................................

            vldCountTotal += vldCount
            # 特异度排名及分位
            group.chaValSortIndex = np.argsort(chaValArr)
            group.betaSortIndex = np.argsort(betaArr)
            for s in range(len(group.stockList)):
                seq = group.chaValSortIndex[s]
                stock = group.stockList[seq]
                if np.isnan(chaValArr[seq]):
                    stock.chaValQG = np.nan
                    stock.chaValPG = np.nan
                else:
                    stock.chaValQG = s
                    stock.chaValPG = s / vldCount
                seq1 = group.betaSortIndex[s]
                stock1 = group.stockList[seq1]
                if np.isnan(betaArr[seq1]):
                    stock1.betaQG = np.nan
                    stock1.betaPG = np.nan
                else:
                    stock1.betaQG = s
                    stock1.betaPG = s / vldCount
        # 3.1.特异度总排名
        chaValArrTotal = np.nan * np.zeros(len(self.stockList))
        betaArrTotal = np.nan * np.zeros(len(self.stockList))
        for s in range(len(self.stockList)):
            chaValArrTotal[s] = self.stockList[s].chaVal
            betaArrTotal[s] = self.stockList[s].beta
        self.chaValSortIndex = np.argsort(chaValArrTotal)
        self.betaSortIndex = np.argsort(betaArrTotal)
        for s in range(len(self.stockList)):
            seq = self.chaValSortIndex[s]
            stock = self.stockList[seq]
            if np.isnan(stock.chaVal):
                stock.chaValQ = np.nan
                stock.chaValP = np.nan
            else:
                stock.chaValQ = s
                stock.chaValP = s / vldCountTotal
            seq1 = self.betaSortIndex[s]
            stock1 = self.stockList[seq1]
            if np.isnan(stock.chaVal):
                stock1.betaQ = np.nan
                stock1.betaP = np.nan
            else:
                stock1.betaQ = s
                stock1.betaP = s / vldCountTotal
        # 4.市值调整换手率
        # 4.0.统计过去流通市值和换手率的均值
        cvArr = np.nan * np.zeros(len(self.stockList))
        tovArr = np.nan * np.zeros(len(self.stockList))
        constArr = np.ones([len(self.stockList)])
        vldBasic = 0
        for s in range(len(self.stockList)):
            stock = self.stockList[s]
            if (mkt.crtDate - stock.firstDay).days < afterIpo or stock.scale not in scaleList:
                continue
            vldBasic += 1
            cvArr[s] = np.nanmean(stock.cvSrs.series)
            tovArr[s] = np.nanmean(stock.tovSrs.series)
        # 4.1.去掉极值
        vldTov = sum(np.isfinite(tovArr))
        tovSortIndex = np.argsort(tovArr)
        extreme = int(0.02 * vldTov)
        for s in range(extreme):
            seqTop = tovSortIndex[s]
            seqBot = tovSortIndex[vldTov - 1 - s]
            tovArr[seqTop] = np.nan
            tovArr[seqBot] = np.nan
        # 4.2.回归
        vld = np.isfinite(tovArr) * np.isfinite(cvArr)
        if sum(vld) > vldBasic / 2:
            c, r, rank, sv = np.linalg.lstsq(np.vstack([cvArr[vld], constArr[vld]]).T, tovArr[vld])
            resArr = np.nan * np.zeros(len(self.stockList))
            vldCount = 1
            for s in range(len(self.stockList)):
                resArr[s] = tovArr[s] - (c[0] * cvArr[s] + c[1])
                vldCount += 1 if np.isfinite(resArr[s]) else 0
            resSortIndex = np.argsort(resArr)
            for s in range(len(self.stockList)):
                seq = resSortIndex[s]
                stock = self.stockList[seq]
                if np.isfinite(resArr[seq]):
                    stock.adjTov = resArr[seq]
                    stock.adjTovQ = s
                    stock.adjTovP = s / vldCount
                else:
                    stock.adjTov = np.nan
                    stock.adjTovQ = np.nan
                    stock.adjTovP = np.nan
            self.adjTovSortIndex = resSortIndex

        # 5.综合排名
        bhvArr = np.nan * np.zeros(len(self.stockList))
        for s in range(len(self.stockList)):
            stock = self.stockList[s]
            # stock.bhv = (stock.chaValP + stock.betaP + stock.adjTovP) / 3
            stock.bhv = (stock.chaValP + stock.adjTovP) / 2
            bhvArr[s] = stock.bhv
        bhvSortIndex = np.argsort(bhvArr)
        for s in range(len(self.stockList)):
            seq = bhvSortIndex[s]
            stock = self.stockList[seq]
            if np.isnan(stock.bhv):
                stock.bhvQ = np.nan
            else:
                stock.bhvQ = s

        # 存入db
        # try:
        #     paraDict = Public.GetPara(os.path.join('.', 'config', 'db.txt'))
        #     rd = redis.StrictRedis(host = paraDict['host'], port = int(paraDict['port']), password = paraDict['pw'], db = paraDict['db'])
        # except Exception as e:
        #     Public.WriteLog('Fail to open Redis DB: ' + str(e))
        #     return
        #
        # dtStr = mkt.crtDate.strftime('%Y%m%d')
        # pipe = rd.pipeline(transaction=True)
        # for stock in self.stockList:
        #     dataDict = {'name': stock.name, 'date': mkt.crtDate.strftime('%Y-%m-%d'),
        #                 'chaVal': stock.chaVal, 'chaValQ': stock.chaValQ, 'chaValP': stock.chaValP,
        #                 'beta': stock.beta, 'betaQ': stock.betaQ, 'betaP': stock.betaP,
        #                 'adjTov': stock.adjTov, 'adjTovQ': stock.adjTovQ, 'adjTovP': stock.adjTovP,
        #                 'bhv': stock.bhv, 'bhvQ': stock.bhvQ, 'scale': stock.scale, 'rtn': stock.rtnSrs.GetLast(),
        #                 'updateTime': str(dt.datetime.now())}
        #     rd.hmset(stock.symbol, dataDict)
        #     rd.hmset(stock.symbol + '.' + dtStr, dataDict)
        # pipe.execute()
        debug = 0
            
