import pandas as pd


def read_dataset(caminho = 'datasets/WINJ20_1min.csv'):
  data_frame = pd.read_csv(caminho) # data frame

  data_frame['Data'] = pd.to_datetime(data_frame['Data'], format='%d/%m/%Y',errors='ignore')
  data_frame['Hora'] = pd.to_datetime(data_frame['Hora']).dt.time

  data_frame['Variação'] = data_frame['Fechamento'].sub(data_frame['Abertura'])
  data_frame['Variação(%)'] = 100*data_frame['Fechamento'].sub(data_frame['Abertura']).div(data_frame['Fechamento'])
  #data_frame['Média'] = (data_frame['Fechamento'].add(data_frame['Abertura']).add(data_frame['Máximo']).add(data_frame['Mínimo'])).div(4)
  data_frame = data_frame.sort_values(by=['Data', 'Hora'])

  return data_frame
