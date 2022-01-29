def CustomerLifeTimeValue(df,penalizer_coef=0.01,months_to_predict=3, discount_rate=0.01):
    '''
    args = df must have Frequency(count), Recency(days), Monetary($) and T (Transation period in days)
    output = [['ID','predicted_clv','manual_predicted_clv']]
    '''

	# Discount rate converts future cash flows (that is revenue/profits) into today’s money for the firm
	# discount_rate=0.01 ----> monthly discount rate ~ 12.7% annually  
    import lifetimes
    import pandas as pd
    import numpy as np
    #       Filter out customer those who have never visited again 
    df = df[df['Frequency']>=1]
    bgf = lifetimes.BetaGeoFitter(penalizer_coef=penalizer_coef)
    bgf.fit(df['Frequency'], df['Recency'], df['T'])

    # Compute the customer alive probability
    df['probability_alive'] = bgf.conditional_probability_alive(df['Frequency'], df['Recency'], df['T'])

    # Predict future transaction for the next 90 (months_to_predict*30) days based on historical data
    transaction_date = months_to_predict*30
    df['pred_num_txn'] = round(bgf.conditional_expected_number_of_purchases_up_to_time(transaction_date, 
                                                                                       df['Frequency'],
                                                                                       df['Recency'],
                                                                                       df['T']),2)


    df_repeated_customers = df.copy()
    # Modeling the monetary value using Gamma-Gamma Model from Lifetimes python library 
    ggf = lifetimes.GammaGammaFitter(penalizer_coef=penalizer_coef)
    ggf.fit(df_repeated_customers['Frequency'],
     df_repeated_customers['Monetary'])

    df_repeated_customers['exp_avg_sales'] = ggf.conditional_expected_average_profit(df_repeated_customers['Frequency'],
                                     df_repeated_customers['Monetary'])

    # predicted_clv --> predicted_annual_lifetime_value
    # Predicting Customer Lifetime Value for the next 3 months
    df_repeated_customers['predicted_clv'] = ggf.customer_lifetime_value(bgf,
                                     df_repeated_customers['Frequency'],
                                     df_repeated_customers['Recency'],
                                     df_repeated_customers['T'],
                                     df_repeated_customers['Monetary'],
                                     time=months_to_predict,     # lifetime in months
                                     freq='D',   # frequency in which the data is present(transaction_date)      
                                     discount_rate=discount_rate) # discount rate

    # Manual predict clv = Predicted no. of transactions * Expected avg sales 
    df_repeated_customers['manual_predicted_clv'] = (df_repeated_customers['pred_num_txn'] *
                                                     df_repeated_customers['exp_avg_sales'])

	#     if the clv is nan impute with mean
	#     df_repeated_customers['predicted_clv'].fillna(df_repeated_customers['predicted_clv'].mean(), inplace=True)
	#     df_repeated_customers['manual_predicted_clv'].fillna(df_repeated_customers['manual_predicted_clv'].mean(), inplace=True)
    df_repeated_customers = df_repeated_customers.round(2)

    return df_repeated_customers[['ID','predicted_clv','manual_predicted_clv']]