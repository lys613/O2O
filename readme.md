


##赛题说明
本赛题提供用户在2016年1月1日至2016年6月30日之间真实线上线下消费行为，预测用户在2016年7月领取优惠券后15天以内的使用情况。 评测指标采用AUC，先对每个优惠券单独计算核销预测的AUC值，再对所有优惠券的AUC值求平均作为最终的评价标准。

赛题地址：https://tianchi.aliyun.com/competition/entrance/231593/information


##解决方案概述
赛题提供了三个数据表，分别为：线上真实销售数据、线下真实销售数据、线下测试数据。记录的时间区间是2016.01.01至2016.06.30,需要预测的是2016年7月份用户领取优惠劵后是否核销。根据这两份数据表，首先对数据集进行划分，然后提取了用户相关的特征、商家相关的特征，优惠劵相关的特征，用户与商家之间的交互特征，以及利用本赛题的leakage得到的其它特征。

划分方式如下：

 - **训练集1：**
   - 特征区间（提取feature） 20160101~20160413
   - 预测区间（提取label）   20160614~20160514
 - **训练集2：**
   - 特征区间（提取feature） 20160201~20160514
   - 预测区间（提取label）   20160515~20160615
 - **测试集：**
   - 特征区间（提取feature） 20160315~20160630
   - 预测区间（提取label）   20160701~20160731
  
用滑窗法划分数据，可以避免太久远的数据影响，如果只用一部分来模拟，数据量太少；充分利用数据，让模型学到更多的情况。


对offline数据集提取特征，一共52个特征

 -  **优惠券相关特征：**
   - 折扣率
   - 消费距离
   - 一个月的第几天
   - 满多少
   - 减多少
   - 第几天收到优惠券
   - 是否满减
   - 收到优惠券日期是否是周末
   - 收到优惠券日期是否周一
   - 收到优惠券日期是否周二
   - 收到优惠券日期是否周三
   - 收到优惠券日期是否周四
   - 收到优惠券日期是否周五
   - 收到优惠券日期是否周六
   - 收到优惠券日期是否周日

 -   **商户相关特征：**
    - 卖出商品总数
    - 使用了优惠券消费的商品总数 
    - 商品优惠券的总数量
    - 购物的最近距离
    - 购物的最远距离
    - 购物的平均距离
    - 购物距离的中位数
    - 商户优惠券转化率
    - 卖出的商品中使用优惠券的占比

 - **用户相关特征：**
  - 用户购买商品的种类  
  - 使用优惠券购买商品的用户距商店的最短距离
  - 使用优惠券购买商品的用户距商店的最大距离
  - 使用优惠券购买商品的用户距商店的平均距离
  - 使用优惠券购买商品的用户距商店的中位数距离
  - 每个用户使用优惠券购买的商品数量
  - 购买商品的总数 
  - 接收优惠券的总数 
  - 收到优惠券的日期和使用之间的平均距离天数
  - 收到优惠券的日期和使用之间的最小距离天数
  - 收到优惠券的日期和使用之间的最大距离天数
  - 用券购买率   
  - 用户优惠券转化率


  
 - **用户——商店相关特征**
	- 用户在特定商户购买率
	- 用户在特定商户购买商品总量
	- 用户收到特定商户的优惠劵数目
	- 用户在特定商户用券购买数目
	- 用户在特定商户有记录的总数   
	- 用户未使用优惠券购买的商品数目
	- 用户在特定商户的优惠券转化率
	- 用户在特定商户的用券购买率
	- 用户在一个商户未用券购买率


 - **其它特征：对dataset3也就是训练数据集（这里用到了赛题的leakage）**
  - 用户领取的特定优惠券数目
  - 用户领取的所有优惠券数目
  - 用户最早领取特定优惠券的天数
  - 用户最近领取特定优惠券的天数
  - 用户当天领取的所有优惠券数量
  - 用户当天领取不同优惠券数量






##模型训练
用XGBoost，GBDT，RandomForest进行训练，最后XGBoost结果最好