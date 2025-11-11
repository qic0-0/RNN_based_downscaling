# -*- coding: utf-8 -*-
"""prophet_linear_adjust_yearly_to_hourly.ipynb

Modified for yearly to hourly downscaling
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import norm


def p_values_from_bounds(y_true, yhat, yhat_lower, yhat_upper, interval_width=0.80):
    z_alpha = norm.ppf(0.5 + interval_width / 2.0)
    estimated_std = (yhat_upper - yhat_lower) / (2 * z_alpha)
    z_scores = (y_true - yhat) / estimated_std
    p_values = norm.cdf(z_scores)

    return p_values

def compute_p_values(y_true, y_pred, cov_matrix):
    std_devs = np.sqrt(np.diag(cov_matrix))
    z_scores = (y_true - y_pred) / std_devs
    p_values = norm.cdf(z_scores)
    return p_values


def linear_accuracy_adjusting(model, fitted, Z, manual, n_sample = 50000):
    if manual:
        if model.mcmc_samples > 0:
            raise NotImplementedError("have not write that function")
        else:
            if model.growth == 'linear':
                cov_matrix = covariance_MAP_linear(model, forecast_df = fitted)
            else:
                cov_matrix = covariance_MAP_nonlinear(model, forecast_df = fitted)
    else:
        model.uncertainty_samples = n_sample
        if model.growth == 'linear':
            simulated = model.predictive_samples(fitted[['ds']])
        else:
            simulated = model.predictive_samples(fitted[['ds', 'cap']])
        sim_matrix = simulated['yhat'].T
        cov_matrix = np.cov(sim_matrix, rowvar=False)

    X = fitted['yhat'].values
    Z_hat = np.sum(X)
    Z_value = Z[Z['ds'] == fitted['ds'].dt.year.unique()[0]]['y'].values[0]

    sigma_dot = np.sum(cov_matrix, axis=1)
    sigma_ddot = np.sum(cov_matrix)
    A = sigma_dot / sigma_ddot
    X_adj = X + A * (Z_value - Z_hat)
    X_lower_adj = fitted['yhat_lower'].values + A * (Z_value - Z_hat)
    X_upper_adj = fitted['yhat_upper'].values + A * (Z_value - Z_hat)

    p_values = compute_p_values(fitted['yhat'], X_adj, cov_matrix)

    return X_adj, X_lower_adj, X_upper_adj, p_values

def covariance_MAP_linear(model, forecast_df):
    """
    return closed form solution of covaraince matrix of MAP with linear trend mode
    """
    horizon = forecast_df.shape[0]
    T = model.history.shape[0]
    S = model.params['delta'].shape[1]
    lambda_ = np.mean(np.abs(model.params['delta']))
    var_delta = (S / T) * 2 * lambda_ ** 2

    sigma_obs = np.mean(model.history['y'] - model.predict(model.history)['yhat'])
    sigma2 = np.var(model.history['y'] - model.predict(model.history)['yhat'])

    cov_matrix = np.zeros((horizon, horizon))
    for i in range(horizon):
        for j in range(horizon):
            t_i = T + i + 1
            t_j = T + j + 1
            shared_changepoints = min(i + 1, j + 1)
            cov_matrix[i, j] = t_i * t_j * shared_changepoints * var_delta
    for i in range(horizon):
        cov_matrix[i, i] += sigma2

    return cov_matrix


def logistic_derivative(g_t, C_t):
    return g_t * (1 - g_t / C_t)

def covariance_MAP_nonlinear(model, forecast_df):
    """
    return closed form solution of covaraince matrix of MAP with non-linear trend mode
    """
    horizon = forecast_df.shape[0]
    T = model.history.shape[0]
    S = model.params['delta'].shape[1]
    lambda_ = np.mean(np.abs(model.params['delta']))
    var_delta = (S / T) * 2 * lambda_ ** 2

    yhat = forecast_df['yhat'].values[:horizon]
    cap = forecast_df['cap'].values[:horizon]
    u_deriv = logistic_derivative(yhat, cap)

    cov_matrix = np.zeros((horizon, horizon))
    for i in range(horizon):
        for j in range(horizon):
            shared_steps = min(i + 1, j + 1)
            t_i = T + i + 1
            t_j = T + j + 1
            cov_matrix[i, j] = (
                u_deriv[i] * u_deriv[j] * t_i * t_j * shared_steps * var_delta
            )
    return cov_matrix

def forecast_next_year_hourly(df, date_start, date_end, yearly_demand, manual = False, bayesian_samples = 0, linear_trend = True,
                             daily=False, weekly=False, yearly=False, monthly=False):

    start = pd.to_datetime(date_start)
    end = pd.to_datetime(date_end)

    df_train = df.loc[(df['ds'] >= start) & (df['ds'] < end)].copy()
    df_test = df.loc[(df['ds'] >= end) & (df['ds'] < end + pd.DateOffset(years=1))].copy()

    if linear_trend:
        growth = 'linear'
    else:
        growth = 'logistic'

    model = Prophet(daily_seasonality=daily, weekly_seasonality=weekly, yearly_seasonality=yearly,
                    mcmc_samples = bayesian_samples, growth = growth)
    if monthly:
        model.add_seasonality(name='monthly', period=365.25 / 12, fourier_order=5)
    model.fit(df_train)

    forecast = model.predict(df_test[['ds']]).sort_values('ds').reset_index()

    fitted = pd.merge(df_test, forecast[['yhat','ds','yhat_lower','yhat_upper',]], on='ds', how='inner').sort_values('ds').reset_index(drop=True)

    fitted['adjusted_y'],fitted['adjusted_y_lower'],fitted['adjusted_y_upper'], fitted['p_value_adj'] = linear_accuracy_adjusting(model, fitted, yearly_demand, manual, n_sample = 50000)

    fitted['p_value'] = p_values_from_bounds(fitted['y'], fitted['yhat'], fitted['yhat_lower'],  fitted['yhat_upper'], interval_width=model.interval_width)

    return fitted

def plot_future(result, nhead = 24):
    result1 = result.iloc[:nhead,:]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 6))
    plt.plot(result1['ds'], result1['yhat'], label='Prediction', color='blue')
    plt.fill_between(result1['ds'], result1['yhat_lower'], result1['yhat_upper'],
                     color='blue', alpha=0.2, label='Confidence Interval')
    plt.plot(result1['ds'], result1['y'], label='True Value', color='black', linestyle='--')
    plt.xlabel('Datetime')
    plt.ylabel('Value')
    plt.title('Prediction with Confidence Interval vs True Value')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.plot(result1['ds'], result1['adjusted_y'], label='Adjusted Prediction', color='blue')
    plt.fill_between(result1['ds'], result1['adjusted_y_lower'], result1['adjusted_y_upper'],
                     color='blue', alpha=0.2, label='Confidence Interval')
    plt.plot(result1['ds'], result1['y'], label='True Value', color='black', linestyle='--')
    plt.xlabel('Datetime')
    plt.ylabel('Value')
    plt.title('Adjusted Prediction with Confidence Interval vs True Value')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def rmse_function(df):
    df['ds'] = pd.to_datetime(df['ds'])
    df['year'] = df['ds'].dt.year
    results = []
    for year, group in df.groupby('year'):
        rmse_yhat = np.sqrt(np.mean((group['y'] - group['yhat'])**2))
        rmse_adjusted = np.sqrt(np.mean((group['y'] - group['adjusted_y'])**2))
        results.append({
            'year': year,
            'RMSE_yhat': rmse_yhat,
            'RMSE_adjusted_y': rmse_adjusted
        })
    return pd.DataFrame(results)


def plot_rmse(rmse_df):
    plt.figure(figsize=(12, 6))

    plt.plot(rmse_df['year'], rmse_df['RMSE_yhat'], label='RMSE (yhat)', marker='o')
    plt.plot(rmse_df['year'], rmse_df['RMSE_adjusted_y'], label='RMSE (adjusted_y)', marker='s')
    plt.xlabel('Year')
    plt.ylabel('RMSE')
    plt.legend()
    plt.tight_layout()

    plt.show()

def bonferroni_global_test(df, alpha=0.05):
    df['ds'] = pd.to_datetime(df['ds'])
    df['year'] = df['ds'].dt.year

    results = []

    for year, group in df.groupby('year'):
        m = len(group)

        # Bonferroni correction: p * m
        min_p_yhat = group['p_value'].min()
        min_p_adj = group['p_value_adj'].min()

        global_signif_yhat = (min_p_yhat * m) <= alpha
        global_signif_adjusted = (min_p_adj * m) <= alpha

        results.append({
            'year': year,
            'min_p_yhat': min_p_yhat,
            'min_p_adjusted': min_p_adj,
            'bonferroni_significant_yhat': global_signif_yhat,
            'bonferroni_significant_adjusted_y': global_signif_adjusted
        })

    return pd.DataFrame(results)
