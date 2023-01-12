import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from numpy.linalg import norm
import time
from pyspark.sql.functions import split, explode,col
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

class Preprocess_data:
    def __init__(self,df,df_user_reach):
        self.df=df
        self.df_user_reach=df_user_reach
        
        
    def fn1(self,x):
        return (''.join([i if 32 < ord(i) < 126 else " " for i in x]))

    def fn2(self,x):
        return (''.join([i if i not in "!@#$%^&*()[]{};:,./<>?\|`~-=_+" else " " for i in x]))

    def fn3(self,x):
        return (''.join([i if i not in '!@#$%^&*()[]{};:,/<>?\|`~-=_+""\n' else " " for i in x]))
    
    def itemperse(self,d,char,dedup):
        idx=d[char]
        res=np.zeros(len(dedup))
        res[idx]=1
        return res

    def preprocessing_kw(self):
        data=list(self.df['keywords'])
        all_sentences = [l.split('\t')[0] for l in data]
        all_sentences = [self.fn1(sentence) for sentence in all_sentences]
        all_sentences = [self.fn2(sentence) for sentence in all_sentences]
        all_sentences = [sentence.lower() for sentence in all_sentences]
        all_sentences = [sentence.strip() for sentence in all_sentences]
        preprocessed_kw={data[i]:all_sentences[i] for i in range(len(data))}### user reach extraction

        self.df_user_reach['preprocessed_kw']=self.df_user_reach['keywords'].map(preprocessed_kw)
        self.df_user_reach=self.df_user_reach.groupby(['preprocessed_kw']).agg({'user_reach':'sum'}).round(2)


        self.df['kw_id']=self.df['keywords'].map(preprocessed_kw)
        df_item_user=self.df.groupby(['line_item_id','kw_id']).size().reset_index()[['line_item_id','kw_id']]
        df_item_user['rating']=1
        
        spark = SparkSession \
                    .builder \
                    .master("local") \
                    .appName("keyword targeting reco") \
                    .getOrCreate()
        sparkDF=spark.createDataFrame(df_item_user) 
        indexer = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(sparkDF.columns)-set(['rating'])) ]
        pipeline = Pipeline(stages=indexer)
        transformed = pipeline.fit(sparkDF).transform(sparkDF)
        #transformed=transformed.select(['line_item_id_index','kw_id_index','rating'])

        md=transformed.select(transformed['line_item_id'],transformed['line_item_id_index'],transformed['kw_id'],transformed['kw_id_index'])
        md=md.toPandas()
        dict1 =dict(zip(md['line_item_id_index'],md['line_item_id']))
        dict2=dict(zip(md['kw_id_index'],md['kw_id']))

        return transformed,self.df_user_reach,preprocessed_kw,dict1,dict2
    
    def preprocessing_kw_cs(self):
        data=list(self.df['keywords'])
        all_sentences = [l.split('\t')[0] for l in data]
        all_sentences = [self.fn1(sentence) for sentence in all_sentences]
        all_sentences = [self.fn2(sentence) for sentence in all_sentences]
        all_sentences = [sentence.lower() for sentence in all_sentences]
        all_sentences = [sentence.strip() for sentence in all_sentences]
        preprocessed_kw={data[i]:all_sentences[i] for i in range(len(data))}### user reach extraction

        self.df_user_reach['preprocessed_kw']=self.df_user_reach['keywords'].map(preprocessed_kw)
        self.df_user_reach=self.df_user_reach.groupby(['preprocessed_kw']).agg({'user_reach':'sum'}).round(2)


        self.df['kw_id']=self.df['keywords'].map(preprocessed_kw)
        df_item_user=self.df.groupby(['advertiser_id','kw_id']).size().reset_index()[['advertiser_id','kw_id']]
        df_item_user['rating']=1
        spark = SparkSession \
                    .builder \
                    .master("local") \
                    .appName("keyword targeting reco") \
                    .getOrCreate()
        sparkDF=spark.createDataFrame(df_item_user) 
        indexer = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(sparkDF.columns)-set(['rating'])) ]
        pipeline = Pipeline(stages=indexer)
        transformed = pipeline.fit(sparkDF).transform(sparkDF)
        #transformed=transformed.select(['line_item_id_index','kw_id_index','rating'])

        md=transformed.select(transformed['advertiser_id'],transformed['advertiser_id_index'],transformed['kw_id'],transformed['kw_id_index'])
        md=md.toPandas()
        dict1 =dict(zip(md['advertiser_id_index'],md['advertiser_id']))
        dict2=dict(zip(md['kw_id_index'],md['kw_id']))

        return transformed,self.df_user_reach,preprocessed_kw,dict1,dict2
    
    def preprocessing_us(self):
        df_item_user=self.df.groupby(['line_item_id','segments']).size().reset_index()[['line_item_id','segments']]
        df_item_user['rating']=1
        spark = SparkSession \
                    .builder \
                    .master("local") \
                    .appName("keyword targeting reco") \
                    .getOrCreate()
        sparkDF=spark.createDataFrame(df_item_user) 
        indexer = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(sparkDF.columns)-set(['rating'])) ]
        pipeline = Pipeline(stages=indexer)
        transformed = pipeline.fit(sparkDF).transform(sparkDF)
        #transformed=transformed.select(['line_item_id_index','kw_id_index','rating'])

        md=transformed.select(transformed['line_item_id'],transformed['line_item_id_index'],transformed['segments'],transformed['segments_index'])
        md=md.toPandas()
        dict1 =dict(zip(md['line_item_id_index'],md['line_item_id']))
        dict2=dict(zip(md['segments_index'],md['segments']))
        return transformed,self.df_user_reach,dict1,dict2
    
    def preprocessing_us_cs(self):
        df_item_user=self.df.groupby(['advertiser_id','segments']).size().reset_index()[['advertiser_id','segments']]
        df_item_user['rating']=1
        spark = SparkSession \
                    .builder \
                    .master("local") \
                    .appName("keyword targeting reco") \
                    .getOrCreate()
        sparkDF=spark.createDataFrame(df_item_user) 
        indexer = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(sparkDF.columns)-set(['rating'])) ]
        pipeline = Pipeline(stages=indexer)
        transformed = pipeline.fit(sparkDF).transform(sparkDF)
        #transformed=transformed.select(['line_item_id_index','kw_id_index','rating'])

        md=transformed.select(transformed['advertiser_id'],transformed['advertiser_id_index'],transformed['segments'],transformed['segments_index'])
        md=md.toPandas()
        dict1 =dict(zip(md['advertiser_id_index'],md['advertiser_id']))
        dict2=dict(zip(md['segments_index'],md['segments']))
        return transformed,self.df_user_reach,dict1,dict2
    
class targeting_tactic_model_KT:
    def __init__(self,df_tt_exp,user_reach_tt,camp_category='food'):
        self.df_tt_exp=df_tt_exp
        self.camp_category=camp_category
        self.user_reach=user_reach_tt
        spark = SparkSession \
            .builder \
            .master("local") \
            .appName("Protob Conversion to Parquet") \
            .getOrCreate()
        self.is_cold_started_trained_already=False
        self.is_base_model_trained=False
    def fit(self,line_item_id,is_needed=False):
        if not self.is_base_model_trained and line_item_id in list(self.df_tt_exp['line_item_id'].unique()) :
            self.targeting_tactic_reco_kw()
            self.is_base_model_trained=True
        if (not self.is_cold_started_trained_already) and is_needed:
            self.targeting_tactic_reco_cold_kw()
            self.is_cold_started_trained_already=True
        
    def targeting_tactic_reco_kw(self):

        self.df_tt_exp=self.df_tt_exp.loc[self.df_tt_exp['camp_category']==self.camp_category]
#         if self.line_item_id not in list(self.df_tt_exp['line_item_id'].unique()):
#             return self.targeting_tactic_reco_cold_kw()
        
        df_kwx,ur_kw,preprocessed_kw,d_lin_item,d_kw=Preprocess_data(self.df_tt_exp,self.user_reach).preprocessing_kw()
        d_reach={ur_kw.index[i]:ur_kw['user_reach'][i] for i in range(ur_kw.shape[0])}
        (training,test)=df_kwx.randomSplit([0.8, 0.2])
        als=ALS(maxIter=5,regParam=0.09,rank=25,userCol="line_item_id_index",itemCol="kw_id_index",ratingCol="rating",coldStartStrategy="drop",nonnegative=True)
        self.model=als.fit(training)

        evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
        predictions=self.model.transform(test)
        rmse=evaluator.evaluate(predictions)
        print("RMSE="+str(rmse))
        
        recs=self.model.recommendForAllUsers(15).toPandas()
        nrecs=recs.recommendations.apply(pd.Series) \
                    .merge(recs, right_index = True, left_index = True) \
                    .drop(["recommendations"], axis = 1) \
                    .melt(id_vars = ['line_item_id_index'], value_name = "recommendation") \
                    .drop("variable", axis = 1) \
                    .dropna() 
        nrecs=nrecs.sort_values('line_item_id_index')
        nrecs=pd.concat([nrecs['recommendation'].apply(pd.Series), nrecs['line_item_id_index']], axis = 1)
        nrecs.columns = [
                'kw_id_index',
                'Rating',
                'line_item_id_index'
             ]
        nrecs['line_item_id']=nrecs['line_item_id_index'].map(d_lin_item)
        nrecs['kw_id']=nrecs['kw_id_index'].map(d_kw)
        nrecs=nrecs.sort_values('line_item_id')
        nrecs.reset_index(drop=True, inplace=True)
        self.new_kw=nrecs[['line_item_id','kw_id','Rating']]

        self.new_kw['User_reach']=self.new_kw['kw_id'].map(d_reach)
        self.df_tt_exp['preprocessed_kw']=self.df_tt_exp['keywords'].map(preprocessed_kw)
        print('Training is finished')
        print('--'*70)
        print('--'*70)
        return
    def recommend_kw(self,line_item_id,average_freq_cap,number_of_impression):
        existed_kw=list(self.df_tt_exp.loc[self.df_tt_exp['line_item_id']==line_item_id]['preprocessed_kw'].values)

        reco_camp_id=self.new_kw[(self.new_kw['line_item_id']==line_item_id)].sort_values('User_reach',ascending=False).reset_index()[['line_item_id','kw_id','Rating','User_reach']]

        reco_camp_id=reco_camp_id[~reco_camp_id['kw_id'].isin(existed_kw)]
        reco_camp_id['impression_provided']=reco_camp_id['User_reach']*average_freq_cap

        reco_camp_id['cum_impression'] = reco_camp_id.impression_provided.cumsum()
        #reco_camp_id=reco_camp_id.loc[reco_camp_id['cum_impression'] <= number_of_impression]
        reco_camp_id=reco_camp_id.reset_index()[['line_item_id','kw_id','Rating','impression_provided','cum_impression']]
        idx=reco_camp_id.loc[reco_camp_id['cum_impression'] <= number_of_impression].index[-1]
        #print('idx: ',idx,number_of_impression)
        if reco_camp_id['cum_impression'][idx]==number_of_impression:
            return reco_camp_id.loc[:str(int(idx))]
        return reco_camp_id.loc[:str(int(idx)+1)] if reco_camp_id.shape[0]>idx else reco_camp_id.loc[:str(int(idx))]
    
    def targeting_tactic_reco_cold_kw(self):

#         if self.advertiser_id not in list(df_kw_exp['advertiser_id'].unique()):
#             print('No recommendation can be provided since Advertiser has not past campaigns history and also line item is new campaign with no data')
#             return

        print('Line item id has no historical data and thus recommending based on the advertiser past campaigns data')
        df_kwx,ur_kw,preprocessed_kw,d_lin_item,d_kw=Preprocess_data(self.df_tt_exp,self.user_reach).preprocessing_kw_cs()
        d_reach={ur_kw.index[i]:ur_kw['user_reach'][i] for i in range(ur_kw.shape[0])}
        (training,test)=df_kwx.randomSplit([0.8, 0.2])
        als=ALS(maxIter=5,regParam=0.09,rank=25,userCol="advertiser_id_index",itemCol="kw_id_index",ratingCol="rating",coldStartStrategy="drop",nonnegative=True)
        self.model_cs=als.fit(training)

        evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
        predictions=self.model_cs.transform(test)
        rmse=evaluator.evaluate(predictions)
        print("RMSE="+str(rmse))

        recs=self.model_cs.recommendForAllUsers(15).toPandas()
        nrecs=recs.recommendations.apply(pd.Series) \
                    .merge(recs, right_index = True, left_index = True) \
                    .drop(["recommendations"], axis = 1) \
                    .melt(id_vars = ['advertiser_id_index'], value_name = "recommendation") \
                    .drop("variable", axis = 1) \
                    .dropna() 
        nrecs=nrecs.sort_values('advertiser_id_index')
        nrecs=pd.concat([nrecs['recommendation'].apply(pd.Series), nrecs['advertiser_id_index']], axis = 1)
        nrecs.columns = [
                'kw_id_index',
                'Rating',
                'advertiser_id_index'
             ]
        nrecs['advertiser_id']=nrecs['advertiser_id_index'].map(d_lin_item)
        nrecs['kw_id']=nrecs['kw_id_index'].map(d_kw)
        nrecs=nrecs.sort_values('advertiser_id')
        nrecs.reset_index(drop=True, inplace=True)
        self.new_kw_cs=nrecs[['advertiser_id','kw_id','Rating']]

        self.new_kw_cs['User_reach']=self.new_kw_cs['kw_id'].map(d_reach)
        self.df_tt_exp['preprocessed_kw']=self.df_tt_exp['keywords'].map(preprocessed_kw)
        print('Training is finished')
        print('--'*70)
        print('--'*70)
        return 
    def recommend_kw_cs(self,advertiser_id,average_freq_cap,number_of_impression):
        reco_camp_id=self.new_kw_cs[(self.new_kw_cs['advertiser_id']==advertiser_id)].sort_values('User_reach',ascending=False).reset_index()[['advertiser_id','kw_id','Rating','User_reach']]

        reco_camp_id['impression_provided']=reco_camp_id['User_reach']*average_freq_cap

        reco_camp_id['cum_impression'] = reco_camp_id.impression_provided.cumsum()
        #reco_camp_id=reco_camp_id.loc[reco_camp_id['cum_impression'] <= number_of_impression]

        reco_camp_id=reco_camp_id.reset_index()[['advertiser_id','kw_id','Rating','impression_provided','cum_impression']]
        idx=reco_camp_id.loc[reco_camp_id['cum_impression'] <= number_of_impression].index[-1]
        if reco_camp_id['cum_impression'][idx]==number_of_impression:
            return reco_camp_id.loc[:str(int(idx))]
        return reco_camp_id.loc[:str(int(idx)+1)] if reco_camp_id.shape[0]>idx else reco_camp_id.loc[:str(int(idx))]
    def recommendation_kw(self,line_item_id,advertiser_id,average_freq_cap=1,number_of_impression=20000):
        if advertiser_id not in list(self.df_tt_exp['advertiser_id'].unique()):
            print('No recommendation can be provided since Advertiser has not past campaigns history and also line item is new campaign with no data')
            return
        if line_item_id not in list(self.df_tt_exp['line_item_id'].unique()):
            if (not self.is_cold_started_trained_already):
                self.fit(line_item_id,is_needed=True)
            return self.recommend_kw_cs(advertiser_id,average_freq_cap,number_of_impression)
        
        return self.recommend_kw(line_item_id,average_freq_cap,number_of_impression)

    
class targeting_tactic_model_BT:
    def __init__(self,df_tt_exp,user_reach_tt,camp_category='food'):
        self.df_tt_exp=df_tt_exp
        self.user_reach=user_reach_tt
        self.camp_category=camp_category
        self.is_cold_started_trained_already=False
        self.is_base_model_trained=False
    
    def fit(self,line_item_id,is_needed=False):
        if not self.is_base_model_trained and line_item_id in list(self.df_tt_exp['line_item_id'].unique()) :
            self.targeting_tactic_reco_us()
            self.is_base_model_trained=True
        if (not self.is_cold_started_trained_already) and is_needed:
            self.targeting_tactic_reco_cold_us()
            self.is_cold_started_trained_already=True
    def targeting_tactic_reco_us(self):
        df_us_exp=self.df_tt_exp.loc[self.df_tt_exp['camp_category']==self.camp_category]
        # if self.line_item_id not in list(self.df_tt_exp['line_item_id'].unique()):
        #     return self.targeting_tactic_reco_cold_us()

        df_usx,ur_us,d_lin_item,d_us=Preprocess_data(self.df_tt_exp,self.user_reach).preprocessing_us()
        d_reach={ur_us['user_segment'][i]:ur_us['user_reach'][i] for i in range(ur_us.shape[0])}

        (training,test)=df_usx.randomSplit([0.8, 0.2])
        als=ALS(maxIter=5,regParam=0.09,rank=25,userCol="line_item_id_index",itemCol="segments_index",ratingCol="rating",coldStartStrategy="drop",nonnegative=True)
        self.model=als.fit(training)

        evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
        predictions=self.model.transform(test)
        rmse=evaluator.evaluate(predictions)
        print("RMSE="+str(rmse))

        recs=self.model.recommendForAllUsers(15).toPandas()
        nrecs=recs.recommendations.apply(pd.Series) \
                    .merge(recs, right_index = True, left_index = True) \
                    .drop(["recommendations"], axis = 1) \
                    .melt(id_vars = ['line_item_id_index'], value_name = "recommendation") \
                    .drop("variable", axis = 1) \
                    .dropna() 
        nrecs=nrecs.sort_values('line_item_id_index')
        nrecs=pd.concat([nrecs['recommendation'].apply(pd.Series), nrecs['line_item_id_index']], axis = 1)
        nrecs.columns = [
                'segments_index',
                'Rating',
                'line_item_id_index'
             ]
        nrecs['line_item_id']=nrecs['line_item_id_index'].map(d_lin_item)
        nrecs['segments']=nrecs['segments_index'].map(d_us)
        nrecs=nrecs.sort_values('line_item_id')
        nrecs.reset_index(drop=True, inplace=True)
        self.new_us=nrecs[['line_item_id','segments','Rating']]

        self.new_us['User_reach']=self.new_us['segments'].map(d_reach)
        print('Training is finished')
        print('--'*70)
        print('--'*70)
        return

    def recommend_us(self,line_item_id,average_freq_cap,number_of_impression):

        existed_us=list(self.df_tt_exp.loc[self.df_tt_exp['line_item_id']==line_item_id]['segments'].values)

        reco_camp_id=self.new_us[(self.new_us['line_item_id']==line_item_id)].sort_values('User_reach',ascending=False).reset_index()[['line_item_id','segments','Rating','User_reach']]

        reco_camp_id=reco_camp_id[~reco_camp_id['segments'].isin(existed_us)]
        reco_camp_id['impression_provided']=reco_camp_id['User_reach']*average_freq_cap

        reco_camp_id['cum_impression'] = reco_camp_id.impression_provided.cumsum()
        #reco_camp_id=reco_camp_id.loc[reco_camp_id['cum_impression'] <= number_of_impression]

        reco_camp_id=reco_camp_id.reset_index()[['line_item_id','segments','Rating','impression_provided','cum_impression']]
        idx=reco_camp_id.loc[reco_camp_id['cum_impression'] <= number_of_impression].index[-1]
        if reco_camp_id['cum_impression'][idx]==number_of_impression:
            return reco_camp_id.loc[:str(int(idx))]
        return reco_camp_id.loc[:str(int(idx)+1)] if reco_camp_id.shape[0]>idx else reco_camp_id.loc[:str(int(idx))]

    def targeting_tactic_reco_cold_us(self):
  
        # if self.advertiser_id not in list(self.df_tt_exp['advertiser_id'].unique()):
        #     print('No recommendation can be provided since Advertiser has not past campaigns history and also line item is new campaign with no data')
        #     return

        print('Line item has no historical data and thus recommending based on the advertiser past campaigns data')
        df_usx,ur_us,d_lin_item,d_us=Preprocess_data(self.df_tt_exp,self.user_reach).preprocessing_us_cs()
    #     d_reach={ur_us.index[i]:ur_us['user_reach'][i] for i in range(ur_us.shape[0])}
        d_reach={ur_us['user_segment'][i]:ur_us['user_reach'][i] for i in range(ur_us.shape[0])}
        (training,test)=df_usx.randomSplit([0.8, 0.2])
        als=ALS(maxIter=5,regParam=0.09,rank=25,userCol="advertiser_id_index",itemCol="segments_index",ratingCol="rating",coldStartStrategy="drop",nonnegative=True)
        self.model_cs=als.fit(training)

        evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
        predictions=self.model_cs.transform(test)
        rmse=evaluator.evaluate(predictions)
        print("RMSE="+str(rmse))

        recs=self.model_cs.recommendForAllUsers(15).toPandas()
        nrecs=recs.recommendations.apply(pd.Series) \
                    .merge(recs, right_index = True, left_index = True) \
                    .drop(["recommendations"], axis = 1) \
                    .melt(id_vars = ['advertiser_id_index'], value_name = "recommendation") \
                    .drop("variable", axis = 1) \
                    .dropna() 
        nrecs=nrecs.sort_values('advertiser_id_index')
        nrecs=pd.concat([nrecs['recommendation'].apply(pd.Series), nrecs['advertiser_id_index']], axis = 1)
        nrecs.columns = [
                'segments_index',
                'Rating',
                'advertiser_id_index'
             ]
        nrecs['advertiser_id']=nrecs['advertiser_id_index'].map(d_lin_item)
        nrecs['segments']=nrecs['segments_index'].map(d_us)
        nrecs=nrecs.sort_values('advertiser_id')
        nrecs.reset_index(drop=True, inplace=True)
        self.new_us_cs=nrecs[['advertiser_id','segments','Rating']]

        self.new_us_cs['User_reach']=self.new_us_cs['segments'].map(d_reach)   
        print('Training is finished')
        print('--'*70)
        print('--'*70)
        return
    def recommend_us_cs(self,advertiser_id,average_freq_cap,number_of_impression):
        reco_camp_id=self.new_us_cs[(self.new_us_cs['advertiser_id']==advertiser_id)].sort_values('User_reach',ascending=False).reset_index()[['advertiser_id','segments','Rating','User_reach']]
        reco_camp_id['impression_provided']=reco_camp_id['User_reach']*average_freq_cap

        reco_camp_id['cum_impression'] = reco_camp_id.impression_provided.cumsum()
        #reco_camp_id=reco_camp_id.loc[reco_camp_id['cum_impression'] <= number_of_impression]

        reco_camp_id=reco_camp_id.reset_index()[['advertiser_id','segments','Rating','impression_provided','cum_impression']]
        idx=reco_camp_id.loc[reco_camp_id['cum_impression'] <= number_of_impression].index[-1]
        if reco_camp_id['cum_impression'][idx]==number_of_impression:
            return reco_camp_id.loc[:str(int(idx))]
        return reco_camp_id.loc[:str(int(idx)+1)] if reco_camp_id.shape[0]>idx else reco_camp_id.loc[:str(int(idx))]
    def recommendation_kw(self,line_item_id,advertiser_id,average_freq_cap=1,number_of_impression=20000):
        if advertiser_id not in list(self.df_tt_exp['advertiser_id'].unique()):
            print('No recommendation can be provided since Advertiser has not past campaigns history and also line item is new campaign with no data')
            return
        if line_item_id not in list(self.df_tt_exp['line_item_id'].unique()):
            if (not self.is_cold_started_trained_already):
                self.fit(line_item_id,is_needed=True)
            return self.recommend_us_cs(advertiser_id,average_freq_cap,number_of_impression)
        
        return self.recommend_us(line_item_id,average_freq_cap,number_of_impression)
    