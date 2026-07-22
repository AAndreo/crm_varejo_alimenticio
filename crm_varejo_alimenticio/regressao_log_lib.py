import pandas as pd # importando a biblioteca de manipulação de dados
import numpy as np # biblioteca para calculo
import scipy.stats as stats # biblioteca para modelagem
import statsmodels.api as sm # biblioteca para a regressão logística
import statistics
import seaborn as sns
import math
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def sumario_analise_var_numerica_por_percentil(data, x, y, q=10):
    """
    Ordena a variável x, divide em percentis e sumariza estatísticas.

    Parâmetros:
        data (pd.DataFrame): O banco de dados contendo as variáveis.
        x (str): O nome da variável independente (explanatória).
        y (str): O nome da variável dependente (resposta).
        q (int): O número de percentis (default: 10).

    Retorno:
        pd.DataFrame: DataFrame com as estatísticas por percentil, incluindo:
                      - Percentil
                      - n (número de linhas)
                      - Min de x
                      - Max de x
                      - p (média de y)
                      - logito de p
    """
    # retirar os valores nulos
    data = data[[x, y]].dropna().reset_index(drop=True)

    # Certificar-se de que a variável y está no formato numérico
    data[y] = pd.to_numeric(data[y], errors='coerce')

    # Ordenar os dados pela variável x
    data = data.sort_values(by=x).reset_index(drop=True)

    # Criar os percentis
    _ , bins_edge = pd.qcut(data[x], q=q, retbins=True, duplicates='drop')
    data['percentil'] = pd.qcut(data[x], q=q, labels=[str(i) for i in range(1, len(bins_edge))], retbins=False, duplicates='drop')

    # Sumarizar as estatísticas por percentil
    summary = data.groupby('percentil').agg(
        n=(x, 'count'),
        min_x=(x, 'min'),
        max_x=(x, 'max'),
        p=(y, 'mean')
    ).reset_index()

    # Calcular o logito de p
    summary['logito_p'] = np.log(summary['p'] / (1 - summary['p']))

    # Ajuste para lidar com casos onde p é 0 ou 1
    epsilon = 1e-10  # Pequeno valor para ajustar 0 e 1
    summary['logito_p'] = np.log(np.clip(summary['p'], epsilon, 1 - epsilon) / 
                                 (1 - np.clip(summary['p'], epsilon, 1 - epsilon)))
    return summary

def analise_var_numerica_por_percentil(data, x, y, q=10, grafico='none'):
    """
    Ordena a variável x, divide em percentis e sumariza estatísticas.

    Parâmetros:
        data (pd.DataFrame): O banco de dados contendo as variáveis.
        x (str): O nome da variável independente (explanatória).
        y (str): O nome da variável dependente (resposta).
        q (int): O número de percentis (default: 10).
        grafico (str): Opção de gráfico: 'p', 'logito', 'ambos', 'none' (default: 'none').

    Retorno:
        pd.DataFrame: DataFrame com as estatísticas por percentil, incluindo:
                      - Percentil
                      - n (número de linhas)
                      - Min de x
                      - Max de x
                      - p (média de y)
                      - logito de p

    Exemplo de uso
        >> data = pd.DataFrame({'x': np.random.uniform(0, 100, 1000), 
        'y': np.random.randint(0, 2, 1000)})
        >> resultado = analise_var_numerica_por_percentil(data, 'x', 'y', q=10, grafico='ambos')
        >> print(resultado)
    """
    # retirar os valores nulos
    data = data[[x, y]].dropna().reset_index(drop=True)
    
    # Certificar-se de que a variável y está no formato numérico
    data[y] = pd.to_numeric(data[y], errors='coerce')

    # Ordenar os dados pela variável x
    data = data.sort_values(by=x).reset_index(drop=True)

    # Criar os percentis
    _ , bins_edge = pd.qcut(data[x], q=q, retbins=True, duplicates='drop')
    data['percentil'] = pd.qcut(data[x], q=q, labels=[str(i) for i in range(1, len(bins_edge))], retbins=False, duplicates='drop')

    # Sumarizar as estatísticas por percentil
    summary = data.groupby('percentil').agg(
        n=(x, 'count'),
        min_x=(x, 'min'),
        max_x=(x, 'max'),
        p=(y, 'mean')
    ).reset_index()

    # Calcular o logito de p
    summary['logito_p'] = np.log(summary['p'] / (1 - summary['p']))

    # Ajuste para lidar com casos onde p é 0 ou 1
    epsilon = 1e-10  # Pequeno valor para ajustar 0 e 1
    summary['logito_p'] = np.log(np.clip(summary['p'], epsilon, 1 - epsilon) / 
                                 (1 - np.clip(summary['p'], epsilon, 1 - epsilon)))


    # Opções de gráfico
    if grafico in ['p', 'logito', 'ambos']:
        plt.figure(figsize=(12, 6))

        if grafico == 'p':
            plt.scatter(summary['percentil'], summary['p'], color='blue')
            plt.title('Gráfico de Percentil x p')
            plt.xlabel('Percentil')
            plt.ylabel('p (média de y)')
            plt.grid(True)
            plt.show()

        elif grafico == 'logito':
            plt.scatter(summary['percentil'], summary['logito_p'], cmap='mako_r')
            plt.title(f'Gráfico de Percentil x Logito de p | {x}')
            plt.xlabel('Percentil')
            plt.ylabel('Logito de p')
            plt.grid(True)
            plt.show()

        elif grafico == 'ambos':
            # Gráficos lado a lado
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

            # Gráfico Percentil x p
            axes[0].scatter(summary['percentil'], summary['p'], color='blue')
            axes[0].set_title('Percentil x p')
            axes[0].set_xlabel('Percentil')
            axes[0].set_ylabel('p (média de y)')
            axes[0].grid(True)

            # Gráfico Percentil x Logito de p
            axes[1].scatter(summary['percentil'], summary['logito_p'], cmap='mako_r')
            axes[1].set_title('Percentil x Logito de p')
            axes[1].set_xlabel('Percentil')
            axes[1].set_ylabel('Logito de p')
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

    return summary


def plot_IC95(model)-> None:

    summary_df = model.summary2().tables[1].reset_index()
    summary_df.rename(columns={"Coef.": "coef", "[0.025": "lower", "0.975]": "upper"}, inplace=True)
    summary_df = summary_df[summary_df["index"] != "Intercept"]  # remove intercepto
    fig, ax = plt.subplots(figsize=(8,12))

    y_positions = range(len(summary_df))

    ax.errorbar(
        summary_df["coef"],
        y_positions,
        xerr=[
            summary_df["coef"] - summary_df["lower"],
            summary_df["upper"] - summary_df["coef"]
        ],
        fmt="o",
        capsize=5,
        ecolor='blue',
        label="Coeficiente"
    )

    ax.axvline(0, color="red", linestyle="--")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary_df["index"])
    ax.set_title("Intervalos de Confiança dos Coeficientes")

    plt.tight_layout()
    plt.legend()
    plt.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.7)
    plt.show()

    return None


def selecionar_pvalor_forward(var_dependente, var_independente, base, signif):
    """   
    Esta função realiza uma seleção forward stepwise com base no p-valor das variáveis independentes.
    A cada passo, adiciona a variável independente com o menor p-valor ao modelo, desde que o p-valor 
    seja menor que o nível de significância especificado.
    
    Parâmetros:
    var_dependente (str): Nome da variável dependente.
    var_independente (list): Lista de variáveis independentes a serem avaliadas.
    base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    signif (float): Nível de significância para a inclusão das variáveis (por exemplo, 0.05).
    
    Retorna: 
    pd.DataFrame: DataFrame contendo as variáveis selecionadas e seus respectivos p-valores.
    
    Exemplo de uso:
        >>> import pandas as pd
        >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
        >>> colunas_pvalor = selecionar_pvalor_forward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df, signif=0.05)
        >>> colunas_pvalor
    
    criada por Mateus Rocha - time ASN.Rocks
    """
    

    preditoras = []
    pvalor_preditoras = []
    Y = base[var_dependente]
    while True and var_independente != []:
        lista_pvalor = []
        lista_variavel = []
        for var in var_independente:
            X = sm.add_constant(base[ [var] +  preditoras ])
            
            modelo = sm.GLM(Y,X,family=sm.families.Binomial()).fit()
            
            if( preditoras == []):    
                
                pvalor = modelo.pvalues[1]
                variavel = modelo.pvalues.index[1]
            
            else:
                
                pvalor = modelo.pvalues.drop(preditoras)[1]
                variavel = modelo.pvalues.drop(preditoras).index[1]
                
            lista_pvalor.append(pvalor)
            lista_variavel.append(variavel)          
        
        if( lista_pvalor[ np.argmin(lista_pvalor) ] < signif ):
            preditoras.append( lista_variavel[np.argmin(lista_pvalor)] )
            pvalor_preditoras.append(lista_pvalor[ np.argmin(lista_pvalor) ])
            var_independente.remove( lista_variavel[ np.argmin(lista_pvalor)] )
        else:
            break
    info_final = pd.DataFrame({ 'var': preditoras, 'pvalor': pvalor_preditoras})
    return info_final


def selecionar_aic_forward(var_dependente, var_independente, base):
    """   
    Esta função realiza uma seleção forward stepwise com base no critério de informação de Akaike (AIC).
    A cada passo, adiciona a variável independente que minimiza o AIC ao modelo.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    
    Retorna: 
        pd.DataFrame: DataFrame contendo as combinações de variáveis selecionadas e seus respectivos AICs, 
        ordenados do menor para o maior AIC.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_aicforward = selecionar_aic_forward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df)
            >>> colunas_aicforward
    
    criada por Mateus Rocha - time ASN.Rocks
    """    

    preditoras = []
    aic_preditoras = []
    Y = base[var_dependente]
    lista_final = []
    aic_melhor = float('inf')
    
    while True and var_independente != []:
        lista_aic = []
        lista_variavel = []
        lista_modelos =[]
        if(var_independente == []):
            break
        for var in var_independente:
            X = sm.add_constant(base[ [var] +  preditoras ])
            aic = sm.GLM(Y,X,family=sm.families.Binomial()).fit().aic
            variavel = var
                
            lista_aic.append(aic)
            
            lista_variavel.append(var)
            
            lista_modelos.append( [var] +  preditoras )
            
        if( lista_aic[ np.argmin(lista_aic) ] < aic_melhor ):
            
            lista_final.append(lista_modelos[ np.argmin(lista_aic)]  )
            
            preditoras.append( lista_variavel[np.argmin(lista_aic)] )
            
            aic_preditoras.append(lista_aic[ np.argmin(lista_aic) ])
            
            var_independente.remove( lista_variavel[ np.argmin(lista_aic)] )
            
            aic_melhor = lista_aic[ np.argmin(lista_aic) ] 
            
        else:
            break
        
    info_final = pd.DataFrame({ 'var': lista_final, 'aic': aic_preditoras}).sort_values(by = 'aic')
    return info_final


def selecionar_bic_forward(var_dependente, var_independente, base):
    
    """   
    Esta função realiza uma seleção forward stepwise com base no critério de informação bayesiano (BIC).
    A cada passo, adiciona a variável independente que minimiza o BIC ao modelo.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    
    Retorna: 
        pd.DataFrame: DataFrame contendo as combinações de variáveis selecionadas e seus respectivos BICs, 
        ordenados do menor para o maior BIC.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_bicforward = selecionar_bic_forward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df)
            >>> colunas_bicforward
    
    criada por Mateus Rocha - time ASN.Rocks
    """
    preditoras = []
    bic_preditoras = []
    Y = base[var_dependente]
    lista_final = []
    bic_melhor = float('inf')
    
    while True and var_independente != []:
        lista_bic = []
        lista_variavel = []
        lista_modelos =[]
        if(var_independente == []):
            break
        for var in var_independente:
            X = sm.add_constant(base[ [var] +  preditoras ])
            bic = sm.GLM(Y,X,family=sm.families.Binomial()).fit().bic
            variavel = var
                
            lista_bic.append(bic)
            
            lista_variavel.append(var)
            
            lista_modelos.append( [var] +  preditoras )
            
        if( lista_bic[ np.argmin(lista_bic) ] < bic_melhor ):
            
            lista_final.append(lista_modelos[ np.argmin(lista_bic)]  )
            
            preditoras.append( lista_variavel[np.argmin(lista_bic)] )
            
            bic_preditoras.append(lista_bic[ np.argmin(lista_bic) ])
            
            var_independente.remove( lista_variavel[ np.argmin(lista_bic)] )
            
            aic_melhor = lista_bic[ np.argmin(lista_bic) ] 
            
        else:
            break
        
    info_final = pd.DataFrame({ 'var': lista_final, 'bic': bic_preditoras}).sort_values(by = 'bic')
    return info_final


def selecionar_pvalor_backward(var_dependente, var_independente, base, signif):
    """   
    Esta função realiza uma seleção backward stepwise com base no p-valor das variáveis independentes.
    A cada passo, remove a variável independente com o maior p-valor do modelo, 
    desde que seja maior que o nível de significância especificado.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
      signif (float): Nível de significância para a inclusão das variáveis (por exemplo, 0.05).
      
    Retorna: 
        pd.DataFrame: DataFrame contendo as variáveis restantes após a seleção backward.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_pvalorbackward = selecionar_pvalor_backward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), signif = 0.05 ,base=df)
            >>> colunas_pvalorbackward
    
    criada por Mateus Rocha - time ASN.Rocks
    """

    Y = base[var_dependente]
    
    while True and var_independente != []:
        
        X_geral = sm.add_constant(base[var_independente])
        
        modelo = sm.GLM(Y,X_geral,family=sm.families.Binomial()).fit()
        
        pvalor_geral = modelo.pvalues
        
        variavel_geral = modelo.pvalues.index
        
        if(pvalor_geral[ np.argmax(pvalor_geral) ] > signif ):
            var_independente.remove( variavel_geral[ np.argmax(pvalor_geral) ] )
        else:
            break
    
    
    
    info_final = pd.DataFrame({ 'var': var_independente})
    return info_final


def selecionar_aic_backward(var_dependente, var_independente, base):
    """   
    Esta função realiza uma seleção backward stepwise com base no critério de informação de Akaike (AIC).
    A cada passo, adiciona a variável independente que minimiza o AIC ao modelo.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    
    Retorna: 
        pd.DataFrame: DataFrame contendo as combinações de variáveis selecionadas e seus respectivos AICs, 
        ordenados do menor para o maior AIC.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_aicbackward = selecionar_aic_backward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df)
            >>> colunas_aicbackward
    
    criada por Mateus Rocha - time ASN.Rocks
    """
    Y = base[var_dependente]
    
    preditoras_finais = []
    
    aic_final = []
    
    while True and var_independente != []:
        
        lista_aic = []
        lista_preditoras = []

        X_geral = sm.add_constant(base[var_independente])
        
        aic_geral = sm.GLM(Y,X_geral,family=sm.families.Binomial()).fit().aic
    
        aic_final.append(aic_geral)
        
        preditoras_finais.append(base[var_independente].columns.to_list())
        
        for var in var_independente:
            
            lista_variaveis = var_independente.copy()
            lista_variaveis.remove(var)
            
            X = sm.add_constant(base[ lista_variaveis ])
            aic = sm.GLM(Y,X,family=sm.families.Binomial()).fit().aic    
            
            lista_aic.append(aic)
            
            lista_preditoras.append(var)
            
        if(lista_aic[ np.argmin(lista_aic) ] < aic_geral ):
            var_independente.remove( lista_preditoras[ np.argmin(lista_aic) ] )
            
        else:
            break
    
    
    info_final = pd.DataFrame({ 'var': preditoras_finais, 'aic':aic_final }).sort_values(by = 'aic')
    return info_final

def selecionar_bic_backward(var_dependente, var_independente, base):
    """   
    Esta função realiza uma seleção backward stepwise com base no critério de informação bayesiano (BIC).
    A cada passo, adiciona a variável independente que minimiza o BIC ao modelo.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
    
    Retorna: 
        pd.DataFrame: DataFrame contendo as combinações de variáveis selecionadas e seus respectivos BICs, 
        ordenados do menor para o maior BIC.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_bicbackward = selecionar_bic_backward(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base=df)
            >>> colunas_bicbackward
    
    criada por Mateus Rocha - time ASN.Rocks
    """
    Y = base[var_dependente]
    
    preditoras_finais = []
    
    bic_final = []
    
    while True and var_independente != []:
        
        lista_bic = []
        lista_preditoras = []

        X_geral = sm.add_constant(base[var_independente])
        
        bic_geral = sm.GLM(Y,X_geral,family=sm.families.Binomial()).fit().bic
    
        bic_final.append(bic_geral)
        
        preditoras_finais.append(base[var_independente].columns.to_list())
        
        for var in var_independente:
            
            lista_variaveis = var_independente.copy()
            lista_variaveis.remove(var)
            
            X = sm.add_constant(base[ lista_variaveis ])
            bic = sm.GLM(Y,X,family=sm.families.Binomial()).fit().bic    
            
            lista_bic.append(bic)
            
            lista_preditoras.append(var)
            
        if(lista_bic[ np.argmin(lista_bic) ] < bic_geral ):
            var_independente.remove( lista_preditoras[ np.argmin(lista_bic) ] )
            
        else:
            break
    
    
    info_final = pd.DataFrame({ 'var': preditoras_finais, 'bic':bic_final }).sort_values(by = 'bic')
    return info_final

def stepwise( var_dependente , var_independente , base, metrica, signif = 0.05, epsilon = 0.0001):
      
    """   
    Esta função realiza a seleção stepwise de variáveis, usando os métodos forward e backward 
    com base em uma métrica específica (AIC, BIC ou p-valor).
    O processo consiste em primeiro aplicar a seleção forward com a métrica escolhida e, 
    em seguida, a backward, ajustando o modelo até que a diferença entre as métricas seja menor 
    que um valor de tolerância (epsilon).
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
      metrica (str): A métrica a ser usada no processo de seleção (pode ser 'aic', 'bic', ou 'pvalor').
      signif (float): Nível de significância usado para a seleção por p-valor (padrão 0.05).
      epsilon (float): Diferença mínima aceitável entre as métricas forward e backward para parar o processo (padrão 0.0001).
    Retorna: 
         Resultado da seleção de variáveis com base no método e métrica escolhidos.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_stepwise = stepwise(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base = df ,metrica='aic', signif=0.05)
            >>> colunas_stepwise
    
    criada por Mateus Rocha - time ASN.Rocks
    """

    lista_var = var_independente
    
    metrica_forward = 0
    
    metrica_backward = 0
    
    while True:
    
        if(metrica == 'aic'):
            resultado = selecionar_aic_forward(var_dependente = var_dependente, var_independente = var_independente, base = base)

            if (len(resultado) == 1):
                return resultado
            
            resultado_final = selecionar_aic_backward(var_dependente = var_dependente, var_independente = resultado['var'].to_list()[0], base = base)

            if(len(resultado_final) == 1):
                return resultado_final

            metrica_forward = resultado['aic'].to_list()[0]

            metrica_backward = resultado_final['aic'].to_list()[0]


        elif(metrica == 'bic'):
            resultado = selecionar_bic_forward(var_dependente = var_dependente, var_independente = var_independente, base = base)

            if (len(resultado) == 1):
                return resultado

            resultado_final = selecionar_bic_backward(var_dependente = var_dependente, var_independente = resultado['var'].to_list()[0], base = base)

            if(len(resultado_final) == 1):
                return resultado_final

            metrica_forward = resultado['bic'].to_list()[0]

            metrica_backward = resultado_final['bic'].to_list()[0]

        elif(metrica == 'pvalor'):
            resultado = selecionar_pvalor_forward(var_dependente = var_dependente, var_independente = var_independente, base = base, signif = signif)

            if (len(resultado) == 1):
                return resultado

            resultado_final = selecionar_pvalor_backward(var_dependente = var_dependente, var_independente = resultado['var'].to_list(), base = base, signif = signif)

            if(len(resultado_final) == 1):
                return resultado_final

            return resultado_final

        if( abs(metrica_forward - metrica_backward) < epsilon ):
            break
        else:
            var_independente = set(resultado_final['var'].to_list() + lista_var)

    
def step( var_dependente , var_independente , base, metodo, metrica, signif = 0.05):
    """   
    Esta função realiza a seleção de variáveis usando os métodos forward, backward ou stepwise, 
    com base em uma métrica escolhida (AIC, BIC ou p-valor).O usuário pode escolher o método de 
    seleção (forward, backward ou both) e a métrica desejada para o critério de inclusão ou exclusão de variáveis.
    
    Parâmetros:
      var_dependente (str): Nome da variável dependente.
      var_independente (list): Lista de variáveis independentes a serem avaliadas.
      base (pd.DataFrame): Conjunto de dados contendo as variáveis dependentes e independentes.
      metrica (str): A métrica a ser usada no processo de seleção (pode ser 'aic', 'bic', ou 'pvalor').
      metodo (str): Método de seleção ('forward', 'backward' ou 'both').
      signif (float): Nível de significância usado para a seleção por p-valor (padrão 0.05).
    Retorna: 
        Resultado da seleção de variáveis com base no método e métrica escolhidos.
        
    Exemplo de uso:
            >>> import pandas as pd
            >>> df = pd.read_csv('https://raw.githubusercontent.com/Zack1803/Body-Fat-Prediction-Dataset/refs/heads/main/bodyfat.csv')
            >>> colunas_step = step(var_dependente='BodyFat', var_independente=df.drop('BodyFat', axis = 1).columns.to_list(), base = df, metodo = 'forward' ,metrica='aic', signif=0.05)
            >>> colunas_step
    
    criada por Mateus Rocha - time ASN.Rocks
    """

    if( metodo == 'forward' and metrica == 'aic' ):
        resultado = selecionar_aic_forward(var_dependente = var_dependente, var_independente = var_independente, base = base)
    elif(metodo == 'forward' and metrica == 'bic' ):
        resultado = selecionar_bic_forward(var_dependente = var_dependente, var_independente = var_independente, base = base)
    elif(metodo == 'forward' and metrica == 'pvalor' ):
        resultado = selecionar_pvalor_forward(var_dependente = var_dependente, var_independente = var_independente, base = base, signif = signif)
    elif( metodo == 'backward' and metrica == 'aic' ):
        resultado = selecionar_aic_backward(var_dependente = var_dependente, var_independente = var_independente, base = base)
    elif(metodo == 'backward'and metrica == 'bic' ):
        resultado = selecionar_bic_backward(var_dependente = var_dependente, var_independente = var_independente, base = base)
    elif(metodo == 'backward' and metrica == 'pvalor' ):
        resultado = selecionar_pvalor_backward(var_dependente = var_dependente, var_independente = var_independente, base = base, signif = signif)
    elif(metodo == 'both'):
        resultado = stepwise( var_dependente = var_dependente , var_independente = var_independente , base = base, metrica = metrica, signif = signif)
        
    return resultado

def calcula_lift(target: pd.Series, p_chapeu: pd.Series, n_percentis: int =10) -> pd.DataFrame:
    df_lift = pd.DataFrame({
        'target': target,
        'p_chapeu': p_chapeu
    }).sort_values(by='p_chapeu', ascending=False).reset_index(drop=True)

    df_lift['percentil'] = pd.qcut(df_lift.index, n_percentis, labels=[str(i) for i in range(1, n_percentis + 1)])

    total_de_1s_acumulado_por_percentil = df_lift.groupby('percentil')['target'].sum().cumsum()
    total_de_linhas_acumulado_por_percentil = df_lift.groupby('percentil')['target'].count().cumsum()
    # taxa de resposta = precisão => media do target por percentil acumulado.
    taxa_resposta = (total_de_1s_acumulado_por_percentil / total_de_linhas_acumulado_por_percentil)
    # lift = precisao / media geral do target
    lift = (total_de_1s_acumulado_por_percentil / total_de_linhas_acumulado_por_percentil) / df_lift['target'].mean()
    # percentual de quanto o modelo é melhor por percentil, do que um cenário aleatorio (media geral target)
    percentual = ((lift - 1) * 100)

    dic_lift = {'taxa_resposta': taxa_resposta, 'lift': lift, '%': percentual}

    return pd.DataFrame(dic_lift)

def plot_logit_smooth(df, var, target='ABAIXOPESO', frac=0.4):
    
    df_temp = df[[var, target]].dropna().copy()
    
    # Ordenar pela variável
    df_temp = df_temp.sort_values(var)
    
    # Criar bins (percentis)
    df_temp['bin'] = pd.qcut(df_temp[var], q=20, duplicates='drop')
    
    # Agregação
    agg = df_temp.groupby('bin').agg({
        var: 'mean',
        target: 'mean'
    }).reset_index()
    
    # Evitar infinito no logito
    agg[target] = agg[target].clip(0.001, 0.999)
    
    # Logito
    agg['logit'] = np.log(agg[target] / (1 - agg[target]))
    
    # LOWESS smoothing
    smooth = lowess(agg['logit'], agg[var], frac=frac)
    
    # Plot
    plt.figure(figsize=(8,5))
    plt.scatter(agg[var], agg['logit'], alpha=0.6, label='Original')
    plt.plot(smooth[:, 0], smooth[:, 1], color='red', linewidth=2, label='LOWESS')
    
    plt.title(f'{var} vs Logito (Suavizado)')
    plt.xlabel(var)
    plt.ylabel('Logito')
    plt.legend()
    plt.grid()
    
    plt.show()