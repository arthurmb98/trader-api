import pandas as pd
import read_data
import train_models
import testes
from fastapi import FastAPI

modelos = train_models.train_models(read_data.read_dataset())

app = FastAPI()

# Configurações

limiar_gap = 15 # gap_maximo = 10 # pontos
erro_ordem = 120 # pontos

@app.get("/")
def root():
    return "Daytrade 1min candle predict api."

@app.get("/teste")
def teste_dataset():
    return testes.teste_model()

@app.get("/api/")
def get_future_candle():

    df_ultimo = read_data.read_ultimo_candle()

    predicao_abertura = modelos[0].predict(df_ultimo)
    predicao_minimo = modelos[1].predict(df_ultimo)
    predicao_maximo = modelos[2].predict(df_ultimo)
    predicao_fechamento = modelos[3].predict(df_ultimo)
    
    df = pd.DataFrame(data={'Abertura': [float(predicao_abertura)],'Mínimo': [float(predicao_minimo)], 'Máximo': [float(predicao_maximo)], 'Fechamento': [float(predicao_fechamento)]})

    # Ultimo valor real
    fechamento_ultimo = df_ultimo['Fechamento'][0].astype(float)
    
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
    