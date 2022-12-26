from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

# definição da função para fit para o model de 'fechamento' do candle atual, para predição do próximo candle
def fit_model_fechamento_saida_atrasada(data_training, show_error = True):

    regression_fechamento = linear_model.LinearRegression()
    
    # Definição da entrada
    entrada = data_training[['Abertura','Mínimo','Máximo','Fechamento']]
    
    features = entrada.iloc[0:len(entrada)-1]
    
    # Definição da saída
    fechamento = data_training[['Fechamento']]
    
    # Predição de Máximo
    
    target_fechamento = fechamento.iloc[1:len(data_training)]
    X_train_fechamento, X_test_fechamento, y_train_fechamento, y_test_fechamento = train_test_split(features, target_fechamento, test_size=0.2, shuffle=False)
    
    mod_fechamento = regression_fechamento.fit(X_train_fechamento, y_train_fechamento)
    
    # print dos erros e score do model   
    if(show_error):
        score_fechamento = mod_fechamento.score(X_test_fechamento, y_test_fechamento) 
        y_pred_fechamento = mod_fechamento.predict(X_test_fechamento)
        print("---------------------------------------------------")
        print('Fechamento score: ', score_fechamento)
        print()
        rmse_fechamento = mean_squared_error(y_test_fechamento, y_pred_fechamento)**(1/2)
        var_score_fechamento = explained_variance_score(y_test_fechamento, y_pred_fechamento)
        print("---------------------------------------------------")
        print("Erros de Fechamento")
        print()
        print("erro(%): ", 100*(abs(y_test_fechamento - y_pred_fechamento)/y_test_fechamento).mean())
        print()
        print("media do erro: ", (y_test_fechamento - y_pred_fechamento).mean())
        print()
        print("media do modulo do erro: ", (abs(y_test_fechamento - y_pred_fechamento)).mean())
        print()
        print("raiz do erro quadratico médio: ", rmse_fechamento)
        print()
        print("variancia score: ", var_score_fechamento)
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print()
    
    return mod_fechamento
