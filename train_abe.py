from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

# definição da função para fit para o model de 'abertura' do candle atual, para predição do próximo candle
def fit_model_abertura_saida_atrasada(data_training, show_error = True):
    
    regression_abertura = linear_model.LinearRegression()
    
    # Definição da entrada
    entrada = data_training[['Abertura','Mínimo','Máximo','Fechamento']]
    
    features = entrada.iloc[0:len(entrada)-1]
    
    # Definição da saída
    abertura = data_training['Abertura']
    
    # Predição de Abertura
    
    target_abertura = abertura.iloc[1:len(data_training)]
    X_train_abertura, X_test_abertura, y_train_abertura, y_test_abertura = train_test_split(features, target_abertura, test_size=0.2, shuffle=False)
    
    mod_abertura = regression_abertura.fit(X_train_abertura, y_train_abertura)
 
    # print dos erros e score do model  
    if(show_error):
        score_abertura = mod_abertura.score(X_test_abertura, y_test_abertura)
        y_pred_abertura = mod_abertura.predict(X_test_abertura)
        rmse_abertura = mean_squared_error(y_test_abertura, y_pred_abertura)**(1/2)
        var_score_abertura = explained_variance_score(y_test_abertura, y_pred_abertura)
        print("---------------------------------------------------")
        print('Abertura score: ', score_abertura)
        print()
        print("Erros de Abertura")
        print()
        print("erro(%): ", 100*(abs(y_test_abertura - y_pred_abertura)/y_test_abertura).mean())
        print()
        print("media do erro: ", (y_test_abertura-y_pred_abertura).mean())
        print()
        print("media do modulo do erro: ", (abs(y_test_abertura-y_pred_abertura)).mean())
        print()
        print("raiz do erro quadratico médio: ", rmse_abertura)
        print()
        print("variancia score: ", var_score_abertura)
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print()
    
    return mod_abertura
