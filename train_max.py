from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

# definição da função para fit para o model de 'maximo' do candle atual, para predição do próximo candle
def fit_model_maximo_saida_atrasada(data_training, show_error = True):

    regression_maximo = linear_model.LinearRegression()
    
    # Definição da entrada
    entrada = data_training[['Abertura','Mínimo','Máximo','Fechamento']]
    
    features = entrada.iloc[0:len(entrada)-1]
    
    # Definição da saída
    maximo = data_training[['Máximo']]
    
    # Predição de Máximo
    
    target_maximo = maximo.iloc[1:len(data_training)]
    X_train_maximo, X_test_maximo, y_train_maximo, y_test_maximo = train_test_split(features, target_maximo, test_size=0.2, shuffle=False)
    
    mod_maximo = regression_maximo.fit(X_train_maximo, y_train_maximo)
    
    # print dos erros e score do model
    if(show_error):
        score_maximo = mod_maximo.score(X_test_maximo, y_test_maximo) 
        y_pred_maximo = mod_maximo.predict(X_test_maximo)
        print("---------------------------------------------------")
        print('Máximo score: ', score_maximo)
        print()
        rmse_maximo = mean_squared_error(y_test_maximo, y_pred_maximo)**(1/2)
        var_score_maximo = explained_variance_score(y_test_maximo, y_pred_maximo)
        print("---------------------------------------------------")
        print("Erros de Máximo")
        print()
        print("erro(%): ", 100*(abs(y_test_maximo - y_pred_maximo)/y_test_maximo).mean())
        print()
        print("media do erro: ", (y_test_maximo - y_pred_maximo).mean())
        print()
        print("media do modulo do erro: ", (abs(y_test_maximo - y_pred_maximo)).mean())
        print()
        print("raiz do erro quadratico médio: ", rmse_maximo)
        print()
        print("variancia score: ", var_score_maximo)
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print()
    
    return mod_maximo
