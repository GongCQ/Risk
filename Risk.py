import numpy as np
import datetime
import time
import os
import shutil
import ZODB.FileStorage as fs
import ZODB
import transaction as ts
import warnings
import sys
import pymysql as ms
sys.setrecursionlimit(10000)  # 设置最大递归深度
warnings.filterwarnings("ignore")  # 关闭警告
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

def GetRoot(path):
    strg = fs.FileStorage(path)
    zDb = ZODB.DB(strg)
    zConn = zDb.open()
    return zConn.root(), zDb

# 尝试从已保存的状态恢复
tempMaxDir = 0
drRoot = os.path.join('.', 'DataRepeater')
for subDir in os.listdir(drRoot):
    if os.path.isfile(os.path.join(drRoot, subDir)):
        os.remove(os.path.join(drRoot, subDir))
        continue
    try:
        datetime.datetime.strptime(subDir, "%Y%m%d")
        subDirNum = int(subDir)
        if subDirNum > tempMaxDir:
            tempMaxDir = subDirNum
    except:
        shutil.rmtree(os.path.join(drRoot, subDir))
if tempMaxDir > 0:
    root, db = GetRoot(os.path.join(drRoot, str(tempMaxDir), 'DataRepeater'))
    mkt = root['mkt']
    stg = root['stg']
    db.close()
else:
    from Init import mkt
    from Init import stg
# 再次更新各对象的数据库链接属性（因为可能因为更改数据库地址而重启程序）----
from Init import connStr
mkt.dbStr = connStr
for ds in mkt.dsList:
    ds.dbStr = connStr
# -------------------------------------------------------------
while mkt.FetchDate():
    print('--------------------------------------', end = '')
    print(datetime.datetime.now())
    print(stg.NetValStr())
    if datetime.datetime.now() < mkt.crtDate + datetime.timedelta(hours = 17, minutes=30):
        break
    while datetime.datetime.now() < mkt.crtDate + datetime.timedelta(hours = 17, minutes=30):
        time.sleep(300)
    print(mkt.crtDate)
    mkt.BeforeOpen()
    mkt.Open()
    mkt.Close()
    mkt.AfterClose()

    mkt.WriteLog('Begin to save status in ZODB')
    lf = mkt.logFile
    mkt.logFile = None  # 文件对象不能序列化，在保存状态之前先置空，下次在FetchDate中恢复
    # 保存状态-------------------------------------------
    # dtStr = mkt.crtDate.strftime('%Y%m%d')
    # drPathTemp = os.path.join(drRoot, '_' + dtStr)
    # drPath = os.path.join(drRoot, dtStr)
    # if os.path.exists(drPathTemp):
    #     shutil.rmtree(drPathTemp)
    # if os.path.exists(drPath):
    #     shutil.rmtree(drPath)
    # os.mkdir(drPathTemp)
    # root, db = GetRoot(os.path.join(drPathTemp, 'DataRepeater'))
    # root['mkt'] = mkt
    # root['stg'] = stg
    # root['saveTime'] = datetime.datetime.now()
    # ts.commit()
    # db.close()
    # # 重命名路径，去掉日期字符串前的下划线，以表示状态已经成功保存
    # os.rename(drPathTemp, drPath)
    # # 删除旧的状态记录
    # for subDir in os.listdir(drRoot):
    #     if subDir != dtStr:
    #         fullPath = os.path.join(drRoot, subDir)
    #         if os.path.isfile(fullPath):
    #             os.remove(fullPath)
    #         elif os.path.isdir(fullPath):
    #             shutil.rmtree(fullPath)
    # ------------------------------------------------------
    mkt.logFile = lf  # 恢复日志
    mkt.WriteLog('Success to save status in ZODB')

stg.AfterAll(mkt)

hah=0