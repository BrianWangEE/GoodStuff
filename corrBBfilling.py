import os
import pandas as pd
import numpy as np
import math as ma

# define python's equivalent of R's match function
def match(a, b):
    return_list = []
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] not in b:
                return_list.append(np.nan)
                break
            else:
                if a[i] == b[j]:
                    return_list.append(j)
                    break
    return return_list


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
os.chdir("D:/ur directory")

DailyData = pd.read_csv('daily.csv', header=0)
WeeklyData = pd.read_csv('weekly.csv', header=0)
DailyData['Date'] = pd.to_datetime(DailyData['Date'], format='%m/%d/%Y')
WeeklyData['Date'] = pd.to_datetime(WeeklyData['Date'], format='%m/%d/%Y')

CMBX_list = ['CMBX AAA Series 4', 'CMBX AA Series 4', 'CMBX A Series 4', 'CMBX BBB Series 4', 'CMBX BB Series 4']
DailyData = DailyData[[item for item in DailyData.columns.tolist() if item not in CMBX_list]]
WeeklyData = WeeklyData[[item for item in WeeklyData.columns.tolist() if item not in CMBX_list]]

absreturn = ['Agency CMBS', '5Y tsy-swap basis', 'FANNIE MAE ']
Date_dropped_df = WeeklyData.drop(['Date'], axis=1)
Col_dropped_df = Date_dropped_df.drop([item for item in absreturn], axis=1)
WeeklyReturn_p1 = Col_dropped_df.diff()/Col_dropped_df.shift(1)
WeeklyReturn_p2 = WeeklyData[[item for item in absreturn]].diff()
WeeklyReturn = pd.concat([WeeklyReturn_p1.reset_index(drop=True), WeeklyReturn_p2], axis=1)
WeeklyReturn = WeeklyReturn.assign(Date=WeeklyData['Date'])
WeeklyReturn = WeeklyReturn.drop(WeeklyReturn.index[0])

Date_dropped_df2 = DailyData.drop(['Date'], axis=1)
Col_dropped_df2 = Date_dropped_df2.drop([item for item in absreturn], axis=1)
DailyReturn_p1 = Col_dropped_df2.diff()/Col_dropped_df2.shift(1)
DailyReturn_p2 = DailyData[[item for item in absreturn if item not in ['Agency CMBS']]].diff()
DailyReturn = pd.concat([DailyReturn_p1.reset_index(drop=True), DailyReturn_p2], axis=1)
DailyReturn = DailyReturn.assign(Date=DailyData['Date'])
DailyReturn = DailyReturn.drop(index=0)

tempWeeklyReturn = WeeklyReturn[WeeklyReturn['Date']<="2008-12-31"]
tempDailyReturn = DailyReturn[DailyReturn['Date']<="2008-12-31"]

tempDailyReturn_dropped = tempDailyReturn.drop(['Date'], axis=1)
tempDailyReturn_dropped = tempDailyReturn_dropped - tempDailyReturn_dropped.mean(axis=0, numeric_only=True, skipna=True)
tempDailyReturn_no_date = tempDailyReturn_dropped / tempDailyReturn_dropped.std(axis=0, numeric_only=True, skipna=True)

corrMatrix = tempWeeklyReturn.corr(method='pearson')
C = np.linalg.cholesky(corrMatrix)
C_XX_1 = C[:-1,:-1]
C_XS_1 = C[-1,:-1]
C_SS_1 = C[-1,-1]

s1 = np.matmul(np.matmul(C_XS_1, np.linalg.inv(C_XX_1)), tempDailyReturn_no_date.transpose())
std_1 = tempWeeklyReturn[['Agency CMBS']].std(axis=0)/ma.sqrt(5)
s2 = pd.Series(std_1).repeat(s1.shape[0]+1)

Output1 = pd.concat([tempDailyReturn['Date'], s1, s2.reset_index(drop=True)], axis=1, ignore_index=True)
Output1 = Output1[1:]

tempWeeklyReturn = WeeklyReturn[WeeklyReturn['Date']>"2008-12-31"]
tempDailyReturn = DailyReturn[DailyReturn['Date']>"2008-12-31"]
tempDailyReturn_dropped = tempDailyReturn.drop(['Date'], axis=1)
tempDailyReturn_dropped = tempDailyReturn_dropped - tempDailyReturn_dropped.mean(axis=0, numeric_only=True, skipna=True)
tempDailyReturn_no_date = tempDailyReturn_dropped / tempDailyReturn_dropped.std(axis=0, numeric_only=True, skipna=True)

corrMatrix = tempWeeklyReturn.corr(method='pearson')
C = np.linalg.cholesky(corrMatrix)
C_XX_2 = C[:-1,:-1]
C_XS_2 = C[-1,:-1]
C_SS_2 = C[-1,-1]

s1 = np.matmul(np.matmul(C_XS_2, np.linalg.inv(C_XX_2)), tempDailyReturn_no_date.transpose())
std_2 = tempWeeklyReturn[['Agency CMBS']].std(axis=0)/ma.sqrt(5)
s2 = pd.Series(std_2).repeat(s1.shape[0]+1)

Output2 = pd.concat([tempDailyReturn['Date'].reset_index(drop=True), s1.reset_index(drop=True),
                     s2.reset_index(drop=True)], axis=1, ignore_index=True)
Output2 = Output2[:-1]
Output = pd.concat([Output1.reset_index(drop=True), Output2.reset_index(drop=True)], axis=0).reset_index(drop=True)
Output.columns = ['Date','Realized','sd_CMBS']
Output['CSS'] = np.nan
Output.loc[Output['Date']<='2008-12-31', 'CSS'] = C_SS_1
Output.loc[Output['Date']>'2008-12-31', 'CSS'] = C_SS_2

Fill = DailyData[['Date', 'Agency CMBS']]
Fill = Fill.assign(Ind=pd.Series(range(1,DailyData.shape[0]+1)))
match_index_1 = match(Fill['Date'].tolist(), WeeklyReturn['Date'].tolist())
Fill = Fill.assign(WeeklyRt=WeeklyReturn['Agency CMBS'].reindex([i+1 for i in match_index_1]).reset_index(drop=True))
Fill = Fill.assign(preDate=Fill['Date'])
Fill['Date'] = Fill['Date'].astype(str)
for i in range(Fill.shape[0]):
    if pd.to_datetime(Fill['Date'][i]) <= min(WeeklyReturn['Date']):
        Fill = Fill.replace(Fill.at[i, 'preDate'],min(WeeklyData['Date']))
    else:
        Fill = Fill.replace(Fill.at[i, 'preDate'], max(WeeklyReturn['Date'][WeeklyReturn['Date']<Fill['Date'][i]]))

match_index_2 = match(Fill['preDate'].astype(str).tolist(), Fill['Date'].tolist())
Fill = Fill.assign(preInd=Fill['Ind'].reindex(match_index_2).reset_index(drop=True))
Fill = Fill.assign(deltaInd=Fill['Ind']-Fill['preInd'])
Fill = Fill.assign(m=Fill['WeeklyRt']/Fill['deltaInd'])
match_index_3 = match(Fill['preDate'].astype(str).tolist(),
                           Fill['preDate'].astype(str)[pd.notna(Fill['WeeklyRt'])].tolist())

Fill = Fill.assign(mu=Fill['m'][pd.notna(Fill['WeeklyRt'])].
                   reset_index(drop=True).reindex(match_index_3).reset_index(drop=True))
Fill = Fill.assign(dt=Fill['deltaInd'][pd.notna(Fill['WeeklyRt'])].
                   reset_index(drop=True).reindex(match_index_3).reset_index(drop=True))
Fill = Fill.assign(AgencyCMBSx=Fill['Agency CMBS'].reindex(match_index_2).reset_index(drop=True))
Fill = Fill.assign(AgencyCMBS=Fill['mu']*Fill['deltaInd']+Fill['AgencyCMBSx'])

Fill = Fill.drop(['m', 'AgencyCMBSx', 'Agency CMBS'], axis=1)
Fill.loc[Fill['Date']=='2008-01-04','deltaInd']=5
Fill.loc[Fill['Date']>='2009-12-28','AgencyCMBS']=65
Fill.loc[Fill['Date']>='2009-12-28','mu']=0
Fill.loc[Fill['Date']>='2009-12-28','dt']=4
Fill.loc[Fill['Date']<='2008-01-04','preDate']=pd.to_datetime('2008-01-02')

# Check Output, Fill dataframe consistency with the original code before the simulation.
# print(Fill)
np.random.seed(123)
Nsim = DailyReturn.shape[0]
Random_total = np.random.normal(0,1,10000*Nsim)
N = 10
Batch = [i for i in range(1,11)]

match_index_4 = match(Fill['Date'].tolist(), Output['Date'].astype(str).tolist())

for run in Batch:

    Temp_Result = pd.DataFrame(data={'Date':Fill['Date'][Fill['Date']>='2008-01-04'].reset_index(drop=True)})
    name_list = ['Date']

    for k in range(1,N+1):

        Output = Output.assign(Random=Output['CSS']*Random_total[(k*run-1)*Nsim:k*run*Nsim])
        Fill = Fill.assign(Simulated=Output['sd_CMBS']*(Output['Random']+Output['Realized']).reindex(match_index_4))
        Fill.at[np.isnan(Fill['Simulated']),'Simulated'] = 0
        CumSim = Fill.groupby('preDate')['Simulated'].sum()
        CumSim_df = pd.DataFrame(data={'preDate':CumSim.index,'WT':CumSim}).reset_index(drop=True)
        match_index_5 = match(Fill['preDate'].astype(str).tolist(), CumSim_df['preDate'].astype(str).tolist())
        Fill = Fill.assign(WT=CumSim_df['WT'].reindex(match_index_5).reset_index(drop=True))
        Fill = Fill.assign(Wt=0.0000)

        for i in range(Fill.shape[0]):
            criteria_1 = Fill['deltaInd']<=Fill.at[i,'deltaInd']
            criteria_2 = Fill['preDate']==Fill.at[i,'preDate']
            ss = Fill['Simulated'][criteria_1 & criteria_2]
            Fill.at[i,'Wt'] = Fill['Simulated'][criteria_1 & criteria_2].sum()

        r = Fill['AgencyCMBS']+(Fill['Wt']-Fill['WT']/Fill['dt']*Fill['deltaInd'])
        Fill = Fill.assign(Result=r)
        Result = Fill['Result'][Fill['Date']>='2008-01-04'].reset_index(drop=True)
        Temp_Result = pd.concat([Temp_Result, Result], axis=1)
        name_list.append(k)
        Temp_Result.columns = name_list
        print(k)

    Temp_Result.to_csv('Output_Batch_{Batch_num}.csv'.format(Batch_num=run), header=True, index=False)


