import gongcq.Tools as Tools
import gongcq.Market as Market
import gongcq.CodeSymbol as CodeSymbol
import datetime as dt
import pymysql as ms
import Strategy
import random

connStr = 'reader/reader@172.16.8.20:1521/dbcenter'
# connStr = 'readonly/anegIjege@121.43.68.222:15210/upwhdb'
# msConfig = {'host':'172.16.8.192', 'port':3306, 'user':'root', 'password':'123456', 'db':'gongchengqi', 'charset':'utf8mb4', 'cursorclass':ms.cursors.Cursor}
sqlBhv = "SELECT tradedate, code, chaVal, adjTov, spdBia, chaValP, adjTovP, spdBiaP FROM bhvIdx WHERE tradedate = '{TRADE_DATE}'"
sqlFin = "SELECT NVL(F.INFO_PUB_DATE, F.END_DATE + 30) INFO_PUB_DATE, P.SEC_UNI_CODE, F.END_DATE, F.BS_30000  " \
         "FROM UPCENTER.FIN_BALA_SHORT F JOIN UPCENTER.PUB_SEC_CODE P   " \
         "               ON P.ORG_UNI_CODE = F.COM_UNI_CODE AND P.ISVALID = 1 AND F.ISVALID = 1   " \
         "WHERE NOT EXISTS(SELECT 1   " \
         "                 FROM UPCENTER.FIN_BALA_SHORT F1   " \
         "                 WHERE F1.ISVALID = 1  " \
         "                       AND F1.INFO_PUB_DATE <= TO_DATE('{TRADE_DATE}', 'YYYY-MM-DD') AND F1.END_DATE > F.END_DATE   " \
         "                       AND F1.COM_UNI_CODE = F.COM_UNI_CODE )  " \
         "      AND P.SEC_SMALL_TYPE_PAR = 101 AND F.INFO_PUB_DATE <= TO_DATE('{TRADE_DATE}', 'YYYY-MM-DD')"
csMap = CodeSymbol.CodeSymbol(connStr)
codeList, symbolList, nameList, mktList = CodeSymbol.GetAllCode(connStr)
scaleStockCodeList = Tools.GetScaleStockCode(connStr, tradeDate=dt.datetime(2016, 2, 1))

mkt = Market.Market(connStr, Tools.sqlCld, dt.datetime(2016, 1, 1))
mkt.CreateDataSource(connStr, Tools.sqlPrc  , codeList        , csMap = None, fieldNum = Tools.numPrc  , label = 'price')
mkt.CreateDataSource(connStr, Tools.sqlSi   , Tools.siCodeList, csMap = None, fieldNum = Tools.numSi   , label = 'scale index')
mkt.CreateDataSource(connStr, sqlFin        , codeList        , csMap = None, fieldNum = 4             , label = 'net value')
mkt.CreateDataSource(connStr, Tools.sqlScale, codeList        , csMap = None, fieldNum = Tools.numScale, label = 'scale')
mkt.CreateDataSource(connStr, Tools.sqlBm   , [2060002293]    , csMap = None, fieldNum = Tools.numBm   , label = 'INDEX')


random.shuffle(scaleStockCodeList[0])
random.shuffle(scaleStockCodeList[1])
random.shuffle(scaleStockCodeList[2])
random.shuffle(scaleStockCodeList[3])
allCode = []
allCode.extend(scaleStockCodeList[0])
allCode.extend(scaleStockCodeList[1])
allCode.extend(scaleStockCodeList[2])
random.shuffle(allCode)

# ssc = allCode
ssc = allCode[0 : 100]
# ssc = scaleStockCodeList[0]
# ssc = scaleStockCodeList[1][0 : 100]
# ssc = scaleStockCodeList[2][0 : 100]

stg = Strategy.Strategy(codeList, csMap, 60, mkt, ssc)
mkt.AddAfterCloseReceiver(stg.NewDayHandler)
mkt.AddAfterCloseReceiver(stg.da.NewDayHandler)
