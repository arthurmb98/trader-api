import pandas as pd
import read_data as rd
import train_models as tm
import testes as t
from fastapi import FastAPI

modelos = tm.train_models(rd.read_dataset())

app = FastAPI()

# Configurações
limiar_gap = 10 # gap_maximo = 10 # pontos
erro_ordem = 100 # pontos 

@app.get("/")
def root():
    return "Daytrade 1min candle predict api."

@app.get("/teste")
def teste_dataset():
    return t.teste_model()

@app.get("/sinal")
def get_future_candle():

    df_ultimo = rd.read_ultimo_candle()

    predicao_abertura = modelos[0].predict(df_ultimo)
    predicao_minimo = modelos[1].predict(df_ultimo)
    predicao_maximo = modelos[2].predict(df_ultimo)
    predicao_fechamento = modelos[3].predict(df_ultimo)
    
    df = pd.DataFrame(data={'Abertura': [str(predicao_abertura)],'Mínimo': [str(predicao_minimo)], 'Máximo': [str(predicao_maximo)], 'Fechamento': [str(predicao_fechamento)]})

    # Ultimo valor real
    fechamento_ultimo = df_ultimo['Fechamento'][0].astype(float)
    
    print()
    print("Fechamento atual: ", fechamento_ultimo)
    print()
    print("Fechamento predito: ", predicao_fechamento[0])
    print()
    
    json = {
        "Tipo de Ordem": "NÃO Efetuar ordem!",
        "Valor da Ordem": 0,
        "Candle Futuro": df,
        "Fechamento passado:": str(fechamento_ultimo)
    }
    
    predicao_var = (predicao_abertura - predicao_fechamento).astype(float)
    predicao_gap = abs(predicao_abertura.astype(float) - fechamento_ultimo)
    
    if(predicao_gap < limiar_gap):
        if(predicao_var >= 0): # se o candle predito for positivo (branco) 
            # efetua uma ordem de venda
            ordem_compra = False
            ordem = predicao_minimo[0][0].astype(float) + erro_ordem
        else: # se o candle for negativo (preto)
            # efetua uma ordem de compra
            ordem_compra = True
            ordem = float( predicao_maximo[0][0].astype(float) - erro_ordem )
        if(ordem_compra):
            print("Efetuar ordem de COMPRA!")
            print("Compra em: ", ordem)
            print()
            json["Tipo de Ordem"] = "COMPRA"
            json["Valor da Ordem"] = ordem
        else:
            print("Efetuar ordem de VENDA!")
            print("Venda em: ", ordem)
            print()
            json["Tipo de Ordem"] = "VENDA"
            json["Valor da Ordem"] = ordem
    else:
        json['Tipo de Ordem'] = "NÃO Efetuar ordem!"
        json['Valor da Ordem'] = ""
        print("NÃO Efetuar ordem!")
        print()
        print("Gap: ", predicao_abertura - fechamento_ultimo)
        print()
        print("Variação: ", predicao_fechamento - predicao_abertura)   
    
    return json
    