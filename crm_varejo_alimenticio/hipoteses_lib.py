# Bibliotecas para testes de hipoteses.
import scipy.stats as stats
from scipy.stats import f_oneway, sem, t, ttest_ind, chi2_contingency, norm, contingency
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
import statsmodels.stats.api as sms
import pingouin as pg

import pandas as pd

def teste_chi2(dataframe: pd.DataFrame, var_categorica1: str , var_categorica2: str) -> None:
    """Calcula o teste de hipoteses qui-quadrado - variáveis categóricas Associação : proporção : +2 categorias : 2 amostras.
    Args: *vars_cat
        var1 : pandas series : variável categórica do dataframe.
        var2 : pandas series : variável categórica do dataframe.        
    """
        
    # var1, var2 = vars_cat
    table = pd.crosstab(dataframe[var_categorica1], dataframe[var_categorica2])
    alpha = 0.05
    
    # print('***** Realizando a teste de hipóteses qui-quadrado *****')
    # print('* Premissas:')
    # print('- As observações devem ser contagens ou frequências.')
    # print('- Cada observação deve pertencer a uma única categoria.')
    # print('- A amostra deve ser relativamente grande, com pelo menos 5 observações em cada célula e, no caso de poucos grupos, pelo menos 10.') 
    # print('- Os dados devem vir de uma amostra aleatória.') 
    # print('- As categorias das variáveis devem ser mutuamente exclusivas.')
    # print('')
    
    # Hipóteses
    print('----- Hipóteses -----')
    print(f'H0 : Não existe associação entre as variáveis {var_categorica1} e {var_categorica2}. As frequencias são estatisticamente iguais.')
    print(f'H1 : Existe associação entre as variáveis {var_categorica1} e {var_categorica2}. As frequencias são estatisticamente diferentes.')
    print(f'Nível de significância de {alpha}.')
    print('---------------------')
    print('')
        
    # Executando o Teste qui-quadrado
    chi2, p_valor, _, _ = chi2_contingency(table)
    print(f"Valor do teste qui-quadrado: {chi2}")
    print(f"P_valor: {p_valor}")
    print('')
    print('----- Conclusão -----')
    if p_valor < alpha:
        print(f"- p_valor: {p_valor:.10f} < alpha: {alpha} => Rejeitamos a hipótese nula. Existe associação entre as variáveis {var_categorica1} e {var_categorica2}.")
        print(f"- Tamanho do Efeito - Cramér´s V: {contingency.association(table,method='cramer'):.10f}")        
    else:
        print(f"- p_valor: {p_valor:.10f} > alpha: {alpha} => Não rejeitamos a hipótese nula. Não existe associação entre as variáveis {var_categorica1} e {var_categorica2}.")
        print(f"- Tamanho do Efeito - Cramér´s V: {contingency.association(table,method='cramer'):.10f}")   