"""## Import Libraries"""
import argparse
import os
import csv
import random
import datetime as dt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

"""## Read Input Files"""
def read_input_files(generation, consumption, bidresult):
  gen_df = pd.read_csv(generation)
  con_df = pd.read_csv(consumption)
  bid_df = pd.read_csv(bidresult)
  return gen_df, con_df, bid_df

"""## Combine gen_df & con_df"""
def combine_input(gen_df, con_df):
  df = gen_df
  df['consumption'] = con_df['consumption']
  return df

"""## Featuring"""
def featuring(df):
  df['time'] = pd.to_datetime(df['time'])
  df['year'] = pd.DatetimeIndex(df['time']).year
  df['month'] = pd.DatetimeIndex(df['time']).month
  df['day'] = pd.DatetimeIndex(df['time']).day
  df['hour'] = pd.DatetimeIndex(df['time']).hour
  df['weekday'] = pd.DatetimeIndex(df['time']).weekday
  return df

"""## Create MinMaxScaler"""
def create_sc():
  sc = MinMaxScaler()
  df = pd.DataFrame()

  for i in range(0, 50):
    tmp = pd.read_csv('./data/target' + str(i) + '.csv')
    tmp = featuring(tmp)
    tmp = tmp.drop(['time'], axis=1)
    df = df.append(tmp, ignore_index=True)

  df = sc.fit_transform(df)

  return sc

"""## Model Definition"""
class LSTM(nn.Module):

  def __init__(self, num_classes, input_size, hidden_size, num_layers):
    super(LSTM, self).__init__()

    self.num_classes = num_classes
    self.num_layers = num_layers
    self.input_size = input_size
    self.hidden_size = hidden_size

    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                        num_layers=num_layers, batch_first=True)

    self.fc = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    h_0 = Variable(torch.zeros(
        self.num_layers, x.size(0), self.hidden_size))

    c_0 = Variable(torch.zeros(
        self.num_layers, x.size(0), self.hidden_size))

    # Propagate input through LSTM
    ula, (h_out, _) = self.lstm(x, (h_0, c_0))

    h_out = h_out.view(-1, self.hidden_size)

    out = self.fc(h_out)

    return out

"""## Load Models"""
def load_models():
  gen_model = torch.load('./model/generation')
  con_model = torch.load('./model/consumption')
  return gen_model, con_model

"""## Make Model Input"""
def make_model_input(gen_df, con_df, sc):
  tmp = combine_input(gen_df, con_df)
  tmp = featuring(tmp)
  tmp = tmp.drop(['time'], axis=1)
  tmp = sc.fit_transform(tmp)
  tmp  = Variable(torch.Tensor(np.array([tmp])))
  return tmp

"""## MinMaxScaler Inverse"""
def inverse_data(data, sc, type):
  tmp = []
  inversed = []

  # reshape for sc input
  for day_data in data:
    for hour_data in day_data:
      if type == 'gen':
        expand_hour_data = np.array([hour_data, 0, 0, 0, 0, 0, 0])
      else:
        expand_hour_data = np.array([0, hour_data, 0, 0, 0, 0, 0])
      tmp.append(expand_hour_data)
  tmp = np.array(tmp)

  # sc inverse
  tmp = sc.inverse_transform(tmp)

  # reshape for output
  for hour_data in tmp:
    if type == 'gen':
      inversed.append(hour_data[0])
    else:
      inversed.append(hour_data[1])
  inversed = np.array(inversed)
  inversed = inversed.reshape(data.shape[0], data.shape[1])

  return inversed

"""## Predict"""
def predict(model, model_input, sc, type):
  model.eval()
  pred = model(model_input)
  pred = pred.data.numpy()
  pred = inverse_data(pred, sc, type)
  return pred[0]

"""## Trade"""
def trade(gen_pred, con_pred, base_time):
  col = ['time', 'action', 'target_price', 'target_volume']
  trade_df = pd.DataFrame(columns=col)
  for i in range(24):
    time = base_time + dt.timedelta(hours=i)
    delta = round(gen_pred[i] - con_pred[i], 2)
    action = ''
    if delta < 0:
      # generation < consumption -> buy
      action = 'buy'
    elif delta > 0:
      action = 'sell'
    else:
      action = ''

    if action == 'buy':
      tmp = [time, action, 3, abs(delta)]
      tmp = pd.DataFrame([tmp], columns=col)
      trade_df = trade_df.append(tmp)
    if action == 'sell':
      tmp = [time, action, 1, abs(delta)]
      tmp = pd.DataFrame([tmp], columns=col)
      trade_df = trade_df.append(tmp)

  return trade_df.reset_index(drop=True)

"""## Output Trades"""
def output_trade(trade_df, output_path):
  trade_df.to_csv(output_path, index=False)

"""## Configuration"""
def config():
  parser = argparse.ArgumentParser()
  parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
  parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
  parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
  parser.add_argument("--output", default="output.csv", help="output the bids path")

  return parser.parse_args()


"""## Main Function"""
if __name__ == "__main__":
  args = config()
  sc = create_sc()
  gen_df, con_df, bid_df = read_input_files(args.generation, args.consumption, args.bidresult)
  model_input = make_model_input(gen_df, con_df, sc)
  gen_model, con_model = load_models()
  gen_pred = predict(gen_model, model_input, sc, 'gen')
  con_pred = predict(con_model, model_input, sc, 'con')
  base_time = gen_df['time'][167] + dt.timedelta(hours=1)
  trade_df = trade(gen_pred, con_pred, base_time)
  output_trade(trade_df, args.output)