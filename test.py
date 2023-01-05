import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
#修改此处，让其图中能正常显示中文
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus']=False
from datetime import datetime
import baostock as bs
def get_data(code):
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day).strftime('%Y-%m-%d')
    end = end.strftime('%Y-%m-%d')
    # 登陆系统
    lg = bs.login()
    # 获取沪深A股历史K线数据
    rs_result = bs.query_history_k_data_plus(
            code,
            fields="date,open,high,low,close,volume",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="3")
    df_result = rs_result.get_data()
    # 退出系统
    bs.logout()
    df_result['date'] = df_result['date'].map(
                        lambda x: datetime.strptime(x,'%Y-%m-%d'))
    _res = df_result.set_index('date')
    res = _res.applymap(lambda x: float(x))
    return res
  liquor_list = ['sh.600207', 'sh.600438',
               'sh.600537', 'sh.600732']
#安彩高科[600225]，通威股份，亿晶光电，爱旭股份
company_name = ['安彩高科','通威股份','亿晶光电','爱旭股份']
for name, code in zip(company_name ,liquor_list):
    exec(f"{name}=get_data(code)") 
company_list = [安彩高科,通威股份,亿晶光电,爱旭股份]
for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
df = pd.concat(company_list, axis=0)
df.tail(10)
for i in company_name:
    print(i)
    df[df['company_name']==i].to_csv(f'{i}.csv',index=False)
    
    plt.figure(figsize=(15, 6))
plt.subplots_adjust(top=1.25, bottom=1.2)
for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['close'].plot()
    plt.ylabel('close')
    plt.xlabel(None)
    plt.title(f"收盘价： {company_name[i - 1]}")  
plt.tight_layout()
plt.figure(figsize=(15, 7))
plt.subplots_adjust(top=1.25, bottom=1.2)
for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['volume'].plot()
    plt.ylabel('volume')
    plt.xlabel(None)
    #修改下面一句代码，结合前面的提示，将图里的英文标题改成中文标题
    plt.title(f"成交量： {company_name[i - 1]}") 
plt.tight_layout()
ma_day = [10, 20, 50]
for ma in ma_day:
    for company in company_list:
        column_name = f"MA for {ma} days"
        company[column_name] = company['close'].rolling(ma).mean()
        fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(8)
fig.set_figwidth(15)
安彩高科[['close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,0])
axes[0,0].set_title('安彩高科移动平均线')

通威股份[['close', 'MA for 10 days', 
          'MA for 20 days', 'MA for 50 days']
          ].plot(ax=axes[0,1])
axes[0,1].set_title('通威股份移动平均线')

亿晶光电[['close', 'MA for 10 days', 
        'MA for 20 days', 'MA for 50 days']
        ].plot(ax=axes[1,0])
axes[1,0].set_title('亿晶光电移动平均线')

爱旭股份[['close', 'MA for 10 days', 
        'MA for 20 days', 'MA for 50 days']
        ].plot(ax=axes[1,1])
axes[1,1].set_title('爱旭股份移动平均线')
fig.tight_layout()
for company in company_list:
    company['Daily Return'] = company['close'].pct_change()
# 画出日收益率
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(8)
fig.set_figwidth(15)
#修改斜体字部分，改用循环实现
安彩高科['Daily Return'].plot(ax=axes[0,0], legend=True, 
                            linestyle='--', marker='o')
axes[0,0].set_title('安彩高科平均日回报率')
通威股份['Daily Return'].plot(ax=axes[0,1], legend=True, 
                               linestyle='--', marker='o')
axes[0,1].set_title('通威股份平均日回报率')
亿晶光电['Daily Return'].plot(ax=axes[1,0], legend=True, 
                            linestyle='--', marker='o')
axes[1,0].set_title('亿晶光电平均日回报率')
爱旭股份['Daily Return'].plot(ax=axes[1,1], legend=True, 
                            linestyle='--', marker='o')
axes[1,1].set_title('爱旭股份平均日回报率')
fig.tight_layout()
plt.figure(figsize=(12, 7))
company_name_c = ['安彩高科','通威股份','亿晶光电','爱旭股份']
for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)#2行2列一共四个子图，i从1开始到4，不能是0
    sns.histplot(company['Daily Return'].dropna(), bins=100, color='purple')
    plt.ylabel('Daily Return')
    plt.title(f'{company_name_c[i - 1]} 日回报率')
# 也可以这样绘制
# 天津松江['Daily Return'].hist()
plt.tight_layout();
index = 安彩高科.index
closing_df = pd.DataFrame()
for company, company_n in zip(company_list,company_name_c):
    temp_df = pd.DataFrame(index=company.index,
                           data = company['close'].values ,
                           columns=[company_n])
    closing_df = pd.concat([closing_df,temp_df],axis=1)
# 看看数据
closing_df.head() 
#下面是计算当前元素与先前元素的相差百分比，即日回报
liquor_rets = closing_df.pct_change()
liquor_rets.head()
sns.jointplot(data=liquor_rets,x='安彩高科',y='亿晶光电', kind='scatter',
              color='seagreen')
sns.jointplot(data=liquor_rets,x='通威股份',y='爱旭股份', kind='scatter',
              color='seagreen')
sns.pairplot(liquor_rets, kind='reg')
# 通过命名为returns_fig来设置我们的图形，
# 在DataFrame上调用PairPLot
#下面是分析四支股票日收益相关性(kde+散点图+直方图)
return_fig = sns.PairGrid(liquor_rets.dropna())
# 使用map_upper，我们可以指定上面的三角形是什么样的。
# 可以对return_fig调用fig.suptitle()函数设置标题。
return_fig.map_upper(plt.scatter, color='purple')
# 我们还可以定义图中较低的三角形，
# 包括绘图类型(kde)或颜色映射(blueppurple)
return_fig.map_lower(sns.kdeplot, cmap='cool_d')
# 最后，我们将把对角线定义为每日收益的一系列直方图
return_fig.map_diag(plt.hist, bins=30)
#下面是分析四支股票收盘价相关性(kde+散点图+直方图)
returns_fig = sns.PairGrid(closing_df)
# 可以对return_fig调用fig.suptitle()函数设置标题。
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)
sns.heatmap(liquor_rets.corr(),
            annot=True, cmap='summer')

#每日收盘价的快速相关图
sns.heatmap(closing_df.corr(),
            annot=True, cmap='summer')
rets = liquor_rets.dropna()
area = np.pi * 20
plt.figure(figsize=(10, 7))
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel('预期回报',fontsize=18)
plt.ylabel('风险',fontsize=18)

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    if label == '爱旭股份':#最后一只股票
        xytext=(-50,-50)
    else:
        xytext=(50,50)
    plt.annotate(label, xy=(x, y), xytext=xytext,
                 textcoords='offset points',
                 ha='right', va='bottom', fontsize=15,
                 arrowprops=dict(arrowstyle='->',
                                 color='gray',
                                 connectionstyle='arc3,rad=-0.3'))
        # 获取股票报价
df = 爱旭股份.loc[:,['open','high','low','close','volume']]
df.head()
# 创建一个只有收盘价的新数据帧
data = df.filter(['close'])
# 将数据帧转换为numpy数组
dataset = data.values
# 获取要对模型进行训练的行数
training_data_len = int(np.ceil( len(dataset) * .95 ))
# 数据标准化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
# 创建训练集，训练标准化训练集
train_data = scaled_data[0:int(training_data_len), :]
# 将数据拆分为x_train和y_train数据集
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
# 将x_train和y_train转换为numpy数组
x_train, y_train = np.array(x_train), np.array(y_train)
# Reshape数据
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
plt.figure(figsize=(16,6))
plt.title('历史收盘价',fontsize=20)
plt.plot(df['close'])
plt.xlabel('日期', fontsize=18)
plt.ylabel('收盘价 RMB (¥)', fontsize=18)
plt.show()
from keras.models import Sequential
from keras.layers import Dense, LSTM
# 建立LSTM模型
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')
# 训练模型
model.fit(x_train, y_train, batch_size=1, epochs =1)

# 创建测试数据集
# 创建一个新的数组，包含从索引的缩放值
test_data = scaled_data[training_data_len - 60: , :]
# 创建数据集x_test和y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
# 将数据转换为numpy数组
x_test = np.array(x_test)
# 重塑的数据
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
# 得到模型的预测值
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
# 得到均方根误差(RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,6))
plt.title('模型')
plt.xlabel('日期', fontsize=18)
plt.ylabel('收盘价 RMB (¥)', fontsize=18)
plt.plot(train['close'])
plt.plot(valid[['close', 'Predictions']])
plt.legend(['训练价格', '实际价格', '预测价格'], loc='lower right')
plt.show()
