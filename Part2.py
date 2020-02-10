from collections import OrderedDict
import time
 

from quantopian.algorithm import attach_pipeline, pipeline_output, order_optimal_portfolio
from quantopian.pipeline import Pipeline, CustomFactor
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.optimize import TargetWeights
from quantopian.pipeline.factors import Returns
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.data.sentdex import sentiment
from quantopian.pipeline.data.psychsignal import twitter_withretweets as twitter_sentiment

import pandas as pd
import numpy as np
 

from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model

 
num_holding_days = 5 
days_for_fundamentals_analysis = 20
upper_percentile = 20
lower_percentile = 30
 
MAX_GROSS_EXPOSURE = 1.0
MAX_POSITION_CONCENTRATION = 0.05

 
def initialize(context):
    set_commission(commission.PerShare(cost=0.001, min_trade_cost=0))
    set_asset_restrictions(security_lists.restrict_leveraged_etfs)
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open(minutes=1))
    attach_pipeline(make_pipeline(), 'my_pipeline')
 
        
class Predictor(CustomFactor):
    value = Fundamentals.ebit.latest / Fundamentals.enterprise_value.latest
    quality = Fundamentals.roe.latest
    sentiment_score = SimpleMovingAverage(
        inputs=[stocktwits.bull_minus_bear],
        window_length=3,
    )
    Fundamentals.ebit

    universe = QTradableStocksUS()
    value_winsorized = value.winsorize(min_percentile=0.05, max_percentile=0.95)
    quality_winsorized = quality.winsorize(min_percentile=0.05, max_percentile=0.95)
    sentiment_score_winsorized = sentiment_score.winsorize(min_percentile=0.05,max_percentile=0.95)
    
    mean_sentiment_5day = SimpleMovingAverage(inputs=[sentiment.sentiment_signal], window_length=5).winsorize(min_percentile=0.05, max_percentile=0.95)
    positive_sentiment_pct = (twitter_sentiment.bull_scored_messages.latest/ twitter_sentiment.total_scanned_messages.latest).winsorize(min_percentile=0.05, max_percentile=0.95)
    workingCapital = Fundamentals.working_capital_per_share.latest.winsorize(min_percentile=0.05, max_percentile=0.95)

    factor_dict = OrderedDict([
              ('Asset_Growth_2d' , Returns(window_length=2)),
              ('Asset_Growth_3d' , Returns(window_length=3)),
              ('Asset_Growth_4d' , Returns(window_length=4)),
              ('Asset_Growth_5d' , Returns(window_length=5)),
              ('Asset_Growth_6d' , Returns(window_length=6)),
              ('Asset_Growth_7d' , Returns(window_length=7)),
              ('Asset_Growth_8d' , Returns(window_length=8)),
              ('Asset_Growth_9d' , Returns(window_length=9)),
              ('Asset_Growth_10d' , Returns(window_length=10)),
              ('Asset_Growth_15d' , Returns(window_length=15)),
              ('Asset_Growth_10d' , Returns(window_length=10)),
              ('Asset_Growth_20d' , Returns(window_length=20)),
              ('Return' , Returns(inputs=[USEquityPricing.open],window_length=5))
              ])
 
    columns = list(factor_dict.keys())
    inputs = list(factor_dict.values())
 
    def compute(self, today, assets, out, *inputs):
        inputs = OrderedDict([(self.columns[i] , pd.DataFrame(inputs[i]).fillna(0,axis=1).fillna(0,axis=1)) for i in range(len(inputs))])
        num_secs = len(inputs['Return'].columns)
        y = inputs['Return'].shift(-num_holding_days)
        y=y.dropna(axis=0,how='all')
        
        for index, row in y.iterrows():
            
             upper = np.nanpercentile(row, upper_percentile)            
             lower = np.nanpercentile(row, lower_percentile)
             auxrow = np.zeros_like(row)
             
             for i in range(0,len(row)):
                if row[i] <= lower: 
                    auxrow[i] = -1
                elif row[i] >= upper: 
                    auxrow[i] = 1 
        
             y.iloc[index] = auxrow
            
        y=y.stack(dropna=False)
        x = pd.concat([df.stack(dropna=False) for df in list(inputs.values())], axis=1).fillna(0)
        
        ## Run Model
        model = linear_model.BayesianRidge()
        #model = GaussianNB()
        model_x = x[:-num_secs*(num_holding_days)]
        model.fit(model_x, y)
        
        out[:] =  model.predict(x[-num_secs:])
 
def make_pipeline():
 
    universe = QTradableStocksUS()
 
    pipe = Pipeline(columns={'Model': Predictor(window_length=days_for_fundamentals_analysis, mask=universe)},screen = universe)
 
    return pipe


def rebalance(context,data):
    
    start_time = time.time()
    pipeline_output_df = pipeline_output('my_pipeline').dropna(how='any')
    todays_predictions = pipeline_output_df.Model
    target_weight_series = todays_predictions.sub(todays_predictions.mean())
    target_weight_series = target_weight_series/target_weight_series.abs().sum()
    order_optimal_portfolio(objective=TargetWeights(target_weight_series),constraints=[])
    print('Full Rebalance Computed Seconds: '+'{0:.2f}'.format(time.time() - start_time))
    print("Number of total securities trading: "+ str(len(target_weight_series[target_weight_series > 0])))
    print("Leverage: " + str(context.account.leverage))