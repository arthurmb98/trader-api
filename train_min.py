from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

# definição da função para fit para o model de 'minimo' do candle atual, para predição do próximo candle
def fit_model_minimo_saida_atrasada(data_training, show_error = True):
    
    regression_minimo = linear_model.LinearRegression()
    
    # Definição da entrada
    entrada = data_training[['Abertura','Mínimo','Máximo','Fechamento']]
    
    features = entrada.iloc[0:len(entrada)-1]
    
    # Definição da saída
    minimo = data_training[['Mínimo']]
    
    # Predição de Mínimo
    
    target_minimo = minimo.iloc[1:len(data_training)]
    X_train_minimo, X_test_minimo, y_train_minimo, y_test_minimo = train_test_split(features, target_minimo, test_size=0.2, shuffle=False)
    
    mod_minimo = regression_minimo.fit(X_train_minimo, y_train_minimo)
    
    # print dos erros e score do model  
    if(show_error):
        score_minimo = mod_minimo.score(X_test_minimo, y_test_minimo)
        y_pred_minimo = mod_minimo.predict(X_test_minimo)
        rmse_minimo = mean_squared_error(y_test_minimo, y_pred_minimo)**(1/2)
        var_score_minimo = explained_variance_score(y_test_minimo, y_pred_minimo)
        print("---------------------------------------------------")
        print('Mínimo score: ', score_minimo)
        print()
        print("Erros de Mínimo")
        print()
        print("erro(%): ", 100*(abs(y_test_minimo - y_pred_minimo)/y_test_minimo).mean())
        print()
        print("media do erro: ", (y_test_minimo - y_pred_minimo).mean())
        print()
        print("media do modulo do erro: ", (abs(y_test_minimo - y_pred_minimo)).mean())
        print()
        print("raiz do erro quadratico médio: ", rmse_minimo)
        print()
        print("variancia score: ", var_score_minimo)
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print()
    
    return mod_minimo
