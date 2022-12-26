import read_data as rd
import train_models as tm


def teste_model(data_test = rd.read_dataset("datasets/WINM20_1min.csv"), banca = 100, qnt_contratos = 1):
    
    data_test = data_test[len(data_test)-480:len(data_test)]
    modelos = tm.train_models(data_test, False)
    
    # Configurações
    erro_ordem_teste = 100 # pontos # com o valor 115, tem-se um acerto de 78%
    
    limite_stop = 100
    limite_gain = 50
    
    #
    gap_minimo = 0 # pontos # com o valor 29, tem-se um acerto de 78%
    gap_maximo = 100 # pontos # com o valor 100, tem-se um acerto de 91%

    var_minima = 0 # pontos # com o valor 1.2, tem-se um acerto de 81%
    var_maximo = 2000 # pontos # com o valor 2.1, tem-se um acerto de 80% 

    #
    valor_contrato = 1 # reais
    qnt_operacoes = 0
    qnt_ganhos = 0
    qnt_perdas = 0
    qnt_vendas = 0
    qnt_compras = 0

    ganho = banca
    ordem_execucao = False

    ordem_compra_teste = False
    ordem_teste = 0

    media_var_perda = 0
    media_var_ganho = 0 
    media_gap_perda = 0
    media_gap_ganho = 0
    
    features_entrada = ['Abertura','Mínimo','Máximo','Fechamento']
    
    for i in range(1, len(data_test)-1):
        x_teste = data_test[features_entrada].iloc[i-1:i]
        minimo_teste = data_test['Mínimo'].iloc[i:i+1]
        maximo_teste = data_test['Máximo'].iloc[i:i+1]
        
        prev_abe = modelos[0].predict(x_teste)
        prev_min = modelos[1].predict(x_teste)
        prev_max = modelos[2].predict(x_teste)
        prev_fec = modelos[3].predict(x_teste)
        
        prev_var = abs(float(prev_fec) - float(prev_abe))
        prev_gap = abs(float(x_teste['Fechamento']) - float(prev_abe))
        if(prev_abe.astype(float) <= prev_fec): # se o candle predito for positivo (branco) 
            # efetua uma ordem de venda
            ordem_compra_teste = False
            qnt_vendas = qnt_vendas + 1
            ordem_teste = prev_min + erro_ordem_teste
        else: # se o candle for negativo (preto)
            # efetua uma ordem de compra
            ordem_compra_teste = True
            ordem_teste = prev_max - erro_ordem_teste 
            qnt_compras = qnt_compras + 1
            
        if(ordem_compra_teste):
            stop_ordem = ordem_teste - limite_stop
            gain_ordem = ordem_teste + limite_gain
        else:
            stop_ordem = ordem_teste + limite_stop
            gain_ordem = ordem_teste - limite_gain       
        
        if(var_minima <= prev_var and var_maximo >= prev_var and prev_gap >= gap_minimo and prev_gap <= gap_maximo and ganho >= limite_stop/5): # caso a minha banca não tenha zerado, continua        
            if(ordem_compra_teste):
                if(minimo_teste.to_numpy()[0] <= ordem_teste): # se entrar na ordem compra
                    if(ordem_execucao == False):
                        qnt_operacoes = qnt_operacoes + 1
                        ordem_execucao = True
            else:
                if(maximo_teste.to_numpy()[0] >= ordem_teste): # se entrar na ordem venda
                    if(ordem_execucao == False):
                        qnt_operacoes = qnt_operacoes + 1
                        ordem_execucao = True
                
            if(ordem_execucao):
                if(ordem_compra_teste):
                    if(minimo_teste.to_numpy()[0] <= stop_ordem): # stop da ordem compra
                        ganho = ganho - qnt_contratos*limite_stop/5 - valor_contrato*qnt_contratos
                        qnt_perdas = qnt_perdas + 1
                        ordem_execucao = False
                        media_var_perda = media_var_perda + prev_var
                        media_gap_perda = media_gap_perda + prev_gap
                    elif(maximo_teste.to_numpy()[0] >= gain_ordem): # gain da ordem compra
                        ganho = ganho + qnt_contratos*limite_gain/5 - valor_contrato*qnt_contratos
                        qnt_ganhos = qnt_ganhos + 1
                        ordem_execucao = False
                        media_var_ganho = media_var_ganho + prev_var
                        media_gap_ganho = media_gap_ganho + prev_gap
                        
                else:
                    if(maximo_teste.to_numpy()[0] >= stop_ordem): # stop da ordem venda
                        ganho = ganho - qnt_contratos*limite_stop/5 - valor_contrato*qnt_contratos
                        qnt_perdas = qnt_perdas + 1
                        ordem_execucao = False
                        media_var_perda = media_var_perda + prev_var
                        media_gap_perda = media_gap_perda + prev_gap
                    elif(minimo_teste.to_numpy()[0] <= gain_ordem): # gain da ordem venda
                        ganho = ganho + qnt_contratos*limite_gain/5 - valor_contrato*qnt_contratos
                        qnt_ganhos = qnt_ganhos + 1
                        ordem_execucao = False
                        media_var_ganho = media_var_ganho + prev_var
                        media_gap_ganho = media_gap_ganho + prev_gap
 
    
    # medias
    if(qnt_perdas>0):
        media_var_perda = media_var_perda/qnt_perdas
        media_gap_perda = media_gap_perda/qnt_perdas
    if(qnt_ganhos>0):
        media_var_ganho = media_var_ganho/qnt_ganhos
        media_gap_ganho = media_gap_ganho/qnt_ganhos

    print("Total de candles: ", len(data_test))
    print("Total de operações: ", qnt_operacoes)
    print("Operações por candles(%): ", 100*qnt_operacoes/len(data_test))
    print()
    print("Quantidade de ganhos: ", qnt_ganhos)
    print("Quantidade de perdas: ", qnt_perdas)
    print()
    print("Média da variação das ganhos: ", media_var_ganho)
    print("Média da gap dos ganhos: ", media_gap_ganho)
    print("Média da variação das perdas: ", media_var_perda)
    print("Média do gap das perdas: ", media_gap_perda)
    print()
    print("Lucro total: R$", ganho - banca)
    print()
    taxa_acerto = 0
    if(qnt_operacoes > 0):
        taxa_acerto = 100*qnt_ganhos/qnt_operacoes
        print("Taxa de acerto (%): ", taxa_acerto)
        print()
        
    json = {
        "Total de candles": len(data_test),
        "Total de operações:": qnt_operacoes,
        "Operações por candles(%):": 100*qnt_operacoes/len(data_test),
        "Quantidade de ganhos:": qnt_ganhos,
        "Quantidade de perdas:": qnt_perdas,
        "Média da variação das ganhos:": media_var_ganho,
        "Média da gap dos ganhos:": media_gap_ganho,
        "Média da variação das perdas:": media_var_perda,
        "Média do gap das perdas:": media_gap_perda,
        "Banca inicial: R$": banca,
        "Lucro total: R$": ganho - banca,
        "Taxa de acerto (%):": taxa_acerto 
    }
        
    return json