import train_abe
import train_fec
import train_max
import train_min


def train_models(data_frame, show_errors = True):
    model_abertura = train_abe.fit_model_abertura_saida_atrasada(data_frame, show_errors)
    model_minimo =train_min.fit_model_minimo_saida_atrasada(data_frame, show_errors)
    model_maximo = train_max.fit_model_maximo_saida_atrasada(data_frame, show_errors)
    model_fechamento =  train_fec.fit_model_fechamento_saida_atrasada(data_frame, show_errors)
    
    return model_abertura, model_minimo, model_maximo, model_fechamento
    