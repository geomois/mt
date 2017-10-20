import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from collections import namedtuple
from dateutil.relativedelta import relativedelta as rd
import QuantLib as ql
import re,math,datetime


def shortDateProcess(dFrame,columns=['X','Y']):
    dat=dFrame.copy()
    for j in range(len(columns)):
        series = np.asarray(dat[columns[j]])
        for i in range(0,len(series)):
            new=re.findall(r'(\d+)(\w)',series[i])
            if len(new)>1:
                if new[0][1].lower()=='m':
                    num=int(new[1][0])
                    # series[i]=(int(new[1][0])/10)#discard the first part
                    #0.11 <- 11 months, 1.02<-14 months, 0.03 <- 3 months
                    series[i]=num/100 if num<12 else 1+(num%12)/100
                else:
                    series[i]=int(new[0][0])+(int(new[1][0])/10)
            elif len(new)==1:
                series[i]=int(new[0][0])
                if new[0][1].lower() == 'm':
                    series[i]=series[i]/100 if series[i]<12 else 1+(series[i]%12/100)
    return dat

def shortDateToQLDate(evaluationDate,series):
    assert type(evaluationDate) == datetime.date , "evaluationDate should be of type datetime.date"
    # return [evaluationDate + rd(months=int(num%1*100)) + rd(years=int(math.floor(num))) for num in series]
    qlDates=[]
    for num in series:
        d = evaluationDate + rd(months=int(num%1*100)) + rd(years=int(math.floor(num)))
        qlDates.append(ql.Date(d.day,d.month,d.year))
    return qlDates

def plotGroups(dat,XaxisLabels,groupByColumn='X',start= -1,end=10):
    g=dat.groupby(groupByColumn)
    ax=XaxisLabels
    counter = 0
    for i in g.groups:
        if start-counter<0 or start==-1:
            plt.plot(ax,g.get_group(i).sort_values('Date',ascending=True).Z,label=i)
        if counter>end+start and start!=-1:
            break
        counter+=1
    plt.legend()
    plt.show()

def writeToFile(dat,sortCol,groupCol,name='smth.csv'):
    g=dat.sort_values(sortCol,ascending=True).groupby(groupCol)#groupby unneeded
    g.head(len(dat)).copy().to_csv(name)

def cleanVolatilities(path="C:\\Users\\SB60MH\\Develop\\data\\SwaptionEURvol.csv",export=False,start=-1,end=10):
    dat=pd.read_csv(path,header=0)
    dat=dat.drop(['MDT_ID','MDE_ID','CURVE_ID_1','CURVE_ID_2','CURVE_ID_3','CURVE_UNIFIED','PNT_ID','GMDB_SYMBOL','Z1','MONEYNESS'],axis=1)
    dat=dat.rename(index=str , columns={'MTM_DATE':"Date"})
    dat.Date=pd.to_datetime(dat.Date)
    dat=shortDateProcess(dat)
    if(export):
        writeToFile(dat,['Date','X','Y'],'Date')
    dat=dat[dat.Date!='2016-02-24']
    dat=dat[dat.Date!='2016-02-23']
    g=dat.groupby(['X','Y'])
    ax=np.asarray(pd.to_datetime(g.get_group((1,0.1)).sort_values('Date',ascending=True).Date))
    plotGroups(dat,ax,['X','Y'],start,end)

def cleanYieldCurve(path="C:\\Users\\SB60MH\\Develop\\Data\\EUR6M.csv",export=False,start=-1,end=10):
    dat=pd.read_csv(path,header=0)
    dat=dat.drop(['MDT_ID','MDE_ID','CURVE_ID_1','CURVE_ID_2','CURVE_ID_3','CURVE_UNIFIED','PNT_ID','GMDB_SYMBOL','Z1','MONEYNESS'],axis=1)
    dat=dat.rename(index=str , columns={'MTM_DATE':"Date"})
    dat.Date=pd.to_datetime(dat.Date)
    dat=shortDateProcess(dat,'X')
    if(export):
        writeToFile(dat,['Date','X'],'Date')
    g=dat.groupby(['X'])
    ax=np.asarray(pd.to_datetime(g.get_group((15)).sort_values('Date',ascending=True).Date))
    plotGroups(dat,ax,'X',start,end)

def shortClean(dat):
    dat=dat.drop(['MDT_ID','MDE_ID','CURVE_ID_1','CURVE_ID_2','CURVE_ID_3','GMDB_SYMBOL','Z1','MONEYNESS'],axis=1)
    dat=dat.rename(index=str , columns={'MTM_DATE':"Date"})
    dat.Date=pd.to_datetime(dat.Date)
    return dat

def toCalibrationTuple(list):
    CalibrationData = namedtuple("CalibrationData","opt,swap,volatility")
    data=[CalibrationData(float(x),float(y),float(z)) for x,y,z in list]
    return data
