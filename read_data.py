import pandas as pd
import MetaTrader5 as mt5


def read_dataset(caminho = 'datasets/WINJ20_1min.csv'):
  data_frame = pd.read_csv(caminho) # data frame

  data_frame['Data'] = pd.to_datetime(data_frame['Data'], format='%d/%m/%Y',errors='ignore')
  data_frame['Hora'] = pd.to_datetime(data_frame['Hora']).dt.time

  data_frame['Variação'] = data_frame['Fechamento'].sub(data_frame['Abertura'])
  data_frame['Variação(%)'] = 100*data_frame['Fechamento'].sub(data_frame['Abertura']).div(data_frame['Fechamento'])
  #data_frame['Média'] = (data_frame['Fechamento'].add(data_frame['Abertura']).add(data_frame['Máximo']).add(data_frame['Mínimo'])).div(4)
  data_frame = data_frame.sort_values(by=['Data', 'Hora'])

  return data_frame

def read_ultimo_candle(ativo = "WIN$"):

    if not mt5.initialize():
        print("MT5 Falhou!")
        mt5.shutdown()
        return
    
    valor = mt5.copy_rates_from_pos(ativo, mt5.TIMEFRAME_M1, 0, 1)
    
    df_ultimo = pd.DataFrame({'Abertura': [valor[0][1]],
                          'Mínimo': [valor[0][2]], 
                          'Máximo': [valor[0][3]],
                          'Fechamento': [valor[0][4]]})
    return df_ultimo
