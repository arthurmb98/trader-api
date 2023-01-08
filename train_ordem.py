from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

# definição da função para fit para o model de 'ordem' do candle atual, para predição do próximo candle
def fit_model_ordem(data_training, show_error = True):
    
    regression_ordem = linear_model.LinearRegression()
    
    # Definição da entrada
    entrada = data_training[['Fechamento']]
    
    # Definição da saída
    ordem = data_training[['Ordem']]
    
    # Predição de Mínimo
    
    X_train_ordem, X_test_ordem, y_train_ordem, y_test_ordem = train_test_split(entrada, ordem, test_size=0.2, shuffle=False)
    
    mod_ordem = regression_ordem.fit(X_train_ordem, y_train_ordem)
    
    # print dos erros e score do model  
    if(show_error):
        score_ordem = mod_ordem.score(X_test_ordem, y_test_ordem)
        y_pred_ordem = mod_ordem.predict(X_test_ordem)
        rmse_ordem = mean_squared_error(y_test_ordem, y_pred_ordem)**(1/2)
        var_score_ordem = explained_variance_score(y_test_ordem, y_pred_ordem)
        print("---------------------------------------------------")
        print('Ordem score: ', score_ordem)
        print()
        print("Erros de Ordem")
        print()
        print("erro(%): ", 100*(abs(y_test_ordem - y_pred_ordem)/y_test_ordem).mean())
        print()
        print("media do erro: ", (y_test_ordem - y_pred_ordem).mean())
        print()
        print("media do modulo do erro: ", (abs(y_test_ordem - y_pred_ordem)).mean())
        print()
        print("raiz do erro quadratico médio: ", rmse_ordem)
        print()
        print("variancia score: ", var_score_ordem)
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print()
    
    return mod_ordem
