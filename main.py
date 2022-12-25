import pandas as pd
import read_dataset
import train_models
from fastapi import FastAPI

modelos = train_models.train_models(read_dataset.read_dataset())

app = FastAPI()

# Configurações

limiar_gap = 15 # gap_maximo = 10 # pontos
erro_ordem = 120 # pontos

@app.get("/api/{fechamento_ultimo}")
def get_future_candle(fechamento_ultimo: float):

    df_ultimo = pd.DataFrame({'Abertura': [fechamento_ultimo],
                          'Mínimo': [fechamento_ultimo], 
                          'Máximo': [fechamento_ultimo],
                          'Fechamento': [fechamento_ultimo]})

    predicao_abertura = modelos[0].predict(df_ultimo)
    predicao_minimo = modelos[1].predict(df_ultimo)
    predicao_maximo = modelos[2].predict(df_ultimo)
    predicao_fechamento = modelos[3].predict(df_ultimo)
    
    df = pd.DataFrame(data={'Abertura': [float(predicao_abertura)],'Máximo': [float(predicao_maximo)],'Mínimo': [float(predicao_minimo)],'Fechamento': [float(predicao_fechamento)]})

    print()
    print("Fechamento atual: ", fechamento_ultimo)
    print()
    print("Fechamento predito: ", predicao_fechamento[0])
    print()
    
    json = {
        "Tipo de Ordem": "NÃO Efetuar ordem!",
        "Valor da Ordem": "",
        "Candle Futuro": df
    }
    
    print(json)
    
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
            ordem = predicao_maximo[0][0].astype(float) - erro_ordem 
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
            json["Valor da Ordem"] = str(ordem)
    else:
        json['Tipo de Ordem'] = "NÃO Efetuar ordem!"
        json['Valor da Ordem'] = ""
        print("NÃO Efetuar ordem!")
        print()
        print("Gap: ", predicao_abertura - fechamento_ultimo)
        print()
        print("Variação: ", predicao_fechamento - predicao_abertura)
        
    
    return json
    
    
@app.get("/api/dataset")
def get_future_candle():
    return {"Dados de treinamento": read_dataset.read_dataset().head()}
