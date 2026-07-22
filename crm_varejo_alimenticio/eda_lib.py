# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy.stats import shapiro

# Bibliotecas para testes de hipoteses.
import scipy.stats as stats
from scipy.stats import f_oneway, sem, t, ttest_ind, chi2_contingency, norm, contingency
from scipy.stats import variation, pearsonr, kurtosis, skew, shapiro, kruskal
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import pingouin as pg


# Biblioteca com funções para Análise Exploratória de Dados (EDA).

class Eda_lib:

    def __init__(self, dataframe: pd.DataFrame, nome: str, colormap: str) -> None:
        self.dataframe = dataframe
        self.colormap = colormap
        self.nome = nome
        # self._linhas: int = self.linhas
        # self._colunas: int = self.colunas
        # self._palette: plt.colors.ListedColormap = self.palette

    def __str__(self) -> str:
        texto = f'O dataset {self._nome} possui {self._linhas} registros e {self._colunas} colunas.' + '\n' + '------------------------------------------------------------------------' + '\n' + self.dataframe.dtypes.to_string()
        return texto

    @property
    def dataframe(self):
        return self._dataframe
    
    @property
    def palette(self):
        return self._palette
    
    @property
    def colormap(self) -> str:
        return self._colormap
    
    @property
    def nome(self):
        return self._nome
    
    @property
    def linhas(self):
        return self._linhas
    
    @property
    def colunas(self):
        return self._colunas
    
    @dataframe.setter
    def dataframe(self, dataframe: pd.DataFrame):
        if dataframe is None:
            raise ValueError('Falta definir o dataframe')
        else:
            self._dataframe = dataframe
            self._linhas = dataframe.shape[0]
            self._colunas = dataframe.shape[1]            

    @colormap.setter
    def colormap(self, colormap: str): 
        if colormap is None:
            raise ValueError('Falta definir as cores para os gráficos')
        else:
            self._colormap = colormap
            self._palette = sns.color_palette(colormap, n_colors=16)
            sns.set_palette(self._palette)

    @nome.setter
    def nome(self, nome: str): 
        if nome is None:
            raise ValueError('Falta definir o nome do dataframe.') 
        else:
            self._nome = nome

    def info(self) -> pd.Series:
        return self.dataframe.dtypes
    
    def valores_ausentes(self) -> pd.DataFrame:
        """
        Verifica e informa a existencia de dados nulos.
        :return: um dataframe contendo as colunas com as respectivas qtdes e percentuais de valores ausentes
        """
        valores_ausentes=self.dataframe.isna().sum()
        porcentagem=(valores_ausentes / self.dataframe.shape[0]) * 100
        data = pd.DataFrame({'valores_ausentes': valores_ausentes, 'porcentagem': porcentagem}).sort_values(by='valores_ausentes', ascending=False)
         
        return data.style.background_gradient(cmap=self.colormap, subset=["porcentagem"])

    def to_datetime(self, colunas: list) -> None:
        """
        Converte determinadas colunas para o tipo de dados (datetime)
        :param colunas: lista de colunas
        :return: informação das colunas com o tipo de dados alterado
        """
        for col in colunas:
            self.dataframe[col] = pd.to_datetime(self.dataframe[col])

        return print(self.dataframe[colunas].info())

    # função para exibir as frequencias das variaveis categoricas
    def frequencia_cat(self, coluna: str) -> pd.DataFrame:
        """
        Exibe as frequências e porcentagens das categorias de uma determinada coluna
        :param coluna: Coluna do dataframe que possue as categorias
        :return: Um dataframe com as informações     
        """
        # data = self.dataframe[coluna].agg(['value_counts',lambda x : (x.value_counts(dropna=True) / self.dataframe.shape[0])*100]).reset_index().rename(columns={'index':coluna,'value_counts':'frequencia', '<lambda>':'porcentagem'}).sort_values(by='porcentagem', ascending=True)
        data = self.dataframe[coluna].value_counts(dropna=True).to_frame().reset_index().rename(columns=({'count':'frequencia'}))
        data['porcentagem'] = (data['frequencia'] / data['frequencia'].sum())*100
        # .agg(['value_counts',lambda x : (x.value_counts(dropna=True) / self.dataframe.shape[0])*100]).reset_index().rename(columns={'index':coluna,'value_counts':'frequencia', '<lambda>':'porcentagem'}).sort_values(by='porcentagem', ascending=True)
        return data.style.background_gradient(cmap=self.colormap, high=.5, subset=["porcentagem"])     

    def _frequencia_cat(self, coluna: str) -> pd.DataFrame:
        data = self.dataframe[coluna].value_counts(dropna=True).to_frame().reset_index().rename(columns=({'count':'frequencia'}))
        data['porcentagem'] = (data['frequencia'] / data['frequencia'].sum())*100
        return data.sort_values(by='porcentagem', ascending=True)

    def taxa_resposta(self, var_categorica: str, target: str) -> pd.DataFrame:
        """Função retornar a taxa de resposta (média de y) entre os grupos de uma variavel categorica e a variável alvo.
        dataframe : pandas dataframe 
        var_categorica : pandas series : variável categórica do dataframe.
        target : pandas series : variável alvo do dataframe.        
        """
        data = self.dataframe.groupby(by=var_categorica, as_index=False).agg(frequencia=(target, 'count'),
                                                                                      taxa_resposta=(target, 'mean'))

        return data.style.background_gradient(cmap=self.colormap, high=.5, subset=["taxa_resposta"])

    def _freq_porcentagem_entre_grp_target(self, var_categorica: str, target: str) -> pd.DataFrame:
        data = self.dataframe.groupby([var_categorica], as_index=False)[target].value_counts(dropna=True).rename(columns=({'count':'frequencia'}))
        data['porcentagem'] = self.dataframe.groupby(by=var_categorica)[target].value_counts(dropna=True, normalize=True).values*100  
        return data.sort_values(by=[var_categorica,target])

    def freq_porcentagem_entre_grp_target(self, var_categorica: str, target: str) -> pd.DataFrame:
        """Função para agrupar as variaveis categoricas de interesse e retornar um dataframe.
        dataframe : pandas dataframe 
        var_categorica : pandas series : variável categórica do dataframe.
        target : pandas series : variável alvo do dataframe.        
        """
        data = self.dataframe.groupby([var_categorica], as_index=False)[target].value_counts(dropna=True).rename(columns=({'count':'frequencia'}))
        data['porcentagem'] = self.dataframe.groupby(by=var_categorica)[target].value_counts(dropna=True, normalize=True).values*100  

        return data.sort_values(by=[var_categorica,target]).style.background_gradient(cmap='mako_r', high=.5, subset=["porcentagem"])

    def shapiro_test(self, coluna: str)-> None:
        # Teste de Normalidade dos dados.
        # H0: Os dados são normais.
        # H1: Os dados não são normais.
        # Nivel de significancia (alpha): 0.05
        """
        Teste de Normalidade dos dados (Saphiro)
        :param coluna: Coluna do dataframe que possue as categorias
        """
        p_valor_shapiro = shapiro(self.dataframe[coluna].dropna())[1]

        # Interpretação
        print('----- Teste de Normalidade dos dados (Shapiro) -----')
        print('H₀ - Hipótese Nula: Os dados seguem uma distribuição normal.')
        print('H₁ - Hipótese Alternativa: Os dados não seguem uma distribuição normal.')
        if p_valor_shapiro > 0.05:
            print(f'p_valor: {p_valor_shapiro} > 0.05. Os dados de {coluna} seguem uma distribuição normal (Hipótese nula não rejeitada).')
        else:
            print(f'p_valor: {p_valor_shapiro} < 0.05. Os dados de {coluna} não seguem uma distribuição normal (Hipótese nula rejeitada).', flush=True)
        return None

    def resumo_estatistico(self, coluna: str) -> None:
        """
        Exibe um resumo estatístico de uma determinada coluna
        :param coluna: Coluna do dataframe
        :return: Um dataframe com as informações estatísticas da coluna 
        """
        def li(x):
            q1 = x.quantile(.25)
            q3 = x.quantile(.75)
            li = q1 - 1.5 * (q3-q1)
            return li

        def ls(x):
            q1 = x.quantile(.25)
            q3 = x.quantile(.75)
            ls = q3 + 1.5 * (q3-q1)
            return ls

        def q25(x):
            return x.quantile(.25)

        def q75(x):
            return x.quantile(.75)        
        
        print('*************************** Resumo Estatístico **************************')
        print('')
        print(f'A variável {coluna} possui {self.dataframe[coluna].count()} registros.')
        print('')
        print('--- Medidas Tendencia Central -------------------------------------------')
        # Medidas Tendencia Central
        print(self.dataframe[coluna].agg({'Média':'mean', 'Mediana':'median'}).to_frame().apply(lambda s: s.apply('{0:.2f}'.format)).T)
        # print('-------------------------------------------------------------------------')
        print('')
        print('--- Medidas de Dispersão ------------------------------------------------')
        # Medidas de dispersão
        print(self.dataframe[coluna].agg({'dp':'std','var':'var','CV%':lambda x: (x.std()/x.mean())*100
                                    , 'Skew': lambda x: x.skew(), 'min':'min', 'max':'max', 'Alcance':lambda x: (x.max() - x.min())}).to_frame().apply(lambda s: s.apply('{0:.2f}'.format)).T)
        # print('-------------------------------------------------------------------------')
        print('')
        print('--- Medidas Separatrizes ------------------------------------------------')
        # Medidas Separatrizes
        print(self.dataframe[coluna].agg({'Q1(25%)':q25, 'Q2(50%)': 'median', 'Q3(75%)':q75, 'Limite inferior':li, 'Limite Superior':ls}).to_frame().T)
        print('')
        # Normalidade dos dados
        self.shapiro_test(coluna=coluna)
          

        return None


    def plot_distribuicao(self, coluna: str) -> None:
        """
        Plota um gráfico com a distribuição dos valores de uma determinada coluna
        :param coluna: Coluna do dataframe que possue as categorias
        """
   
        f, (ax_box, ax_scatter, ax_hist) = plt.subplots(3, sharex=True, gridspec_kw={"height_ratios": (.25, .20, .55)})
        bins = np.histogram_bin_edges(self.dataframe.loc[~self.dataframe[coluna].isna(), coluna], bins='auto')

        sns.boxplot(data=self.dataframe, x=coluna, meanprops={'marker' : 'D', 'markeredgecolor' : 'black', 'markersize' : 6},
                    showmeans=True, showfliers=True, showbox=True, showcaps=True, fill=True, linecolor='k', ax=ax_box)
        sns.stripplot(data=self.dataframe, x=coluna, jitter=0.3, alpha=0.5, ax=ax_scatter)
        sns.histplot(data=self.dataframe, x=coluna,  bins=bins, shrink=0.95, stat='density', ax=ax_hist)
        sns.kdeplot(data=self.dataframe, x=coluna, fill=True, alpha=0.5, ax=ax_hist)

        ax_box.set(yticks=[])
        ax_hist.set(yticks=[])
        ax_scatter.set(yticks=[])
        ax_hist.set(ylabel='')
        ax_hist.set(xlabel=f'Distribuição de {coluna}')

        sns.despine(ax=ax_hist, left=True)
        sns.despine(ax=ax_box, left=True)
        sns.despine(ax=ax_scatter, left=True)

        plt.show() 

        self.resumo_estatistico(coluna=coluna)
        
        return None
   
    def plot_frequencia_cat(self, coluna: str, xlabel: str=np.nan, title: str=np.nan, orient: str='v') -> None:
        """
        Plota um gráfico com as frequências e porcentagens das categorias de uma determinada coluna
        :param coluna: Coluna do dataframe que possue as categorias
        :param xlabel: label do eixo X
        :param title: Título do gráfico 
        """
        if pd.isna(xlabel):
            xlabel = coluna

        if pd.isna(title):
            title = 'Frequencia de ' + coluna

        data = self._frequencia_cat(coluna)

        plt.figure(figsize=(8,4))

        if str.lower(orient) == 'v':
            ax = sns.barplot(data=data, x=coluna, y='frequencia', order=data[coluna], orient=orient, palette=self.palette)
            plt.ylabel('Frequencia')
            plt.xlabel(xlabel)
            plt.xticks(rotation=45, fontsize=8)

            for container in ax.containers:
                # ax.bar_label(container, labels = [f'{x.get_height():.2f}%' for x in container])
                ax.bar_label(container, labels = [f'{x.get_height():.0f}' for x in container])

        elif str.lower(orient) == 'h':
            ax = sns.barplot(data=data, x='frequencia', y=coluna, orient=orient, palette=self.palette)
            plt.ylabel(xlabel)
            plt.xlabel('%')

            for container in ax.containers:
                ax.bar_label(container, labels = [f'{x.get_width():.0f}' for x in container])
    
        plt.title(title)
        sns.despine()
        plt.show()
        return None

    def plot_boxplot(self, var_numerica:str, var_categorica:str) -> None:
        fig, ax = plt.subplots(figsize=(10,5),dpi=75)
        sns.boxplot(data=self.dataframe, x=var_categorica, y=var_numerica, hue=var_categorica, palette=self.palette,
                    meanprops={'marker' : 'D', 'markeredgecolor' : 'black', 'markersize' : 6},
                    showmeans=True, showfliers=True, showbox=True, showcaps=True, fill=True, linecolor='k')        
        plt.title('Comparação entre grupos')
        return None
    
    def plot_freq_porcentagem_entre_grp_target(self, var_categorica: str, target: str, xlabel: str=np.nan, title: str=np.nan, orient: str='v') -> None:
        """
        Plota um gráfico com as frequências e porcentagens das categorias de uma determinada coluna
        :param dataframe : pandas dataframe 
        :param var_categorica: Coluna do dataframe que possue as categorias
        :param target: Coluna do dataframe alvo
        :param xlabel: label do eixo X
        :param title: Título do gráfico 
        """
        if pd.isna(xlabel):
            xlabel = var_categorica

        if pd.isna(title):
            title = 'Frequência entre os grupos de ' + var_categorica + ' e ' + target
      
        plt.figure(figsize=(8,4))

        if str.lower(orient) == 'v':
            ax = sns.countplot(data=self.dataframe, x=var_categorica, hue=target, palette=self.palette, gap=0.1, dodge=True)            
            plt.xticks(rotation=45, fontsize=8)
            ax.set_ylabel('frequência')
            legend = ax.get_legend()
            legend.set_bbox_to_anchor((1, 1))

            for container in ax.containers:
                ax.bar_label(container, labels = [f'{v.get_height():.0f}' if v.get_height() > 0 else '' for v in container], label_type='edge', fontsize=8)

        elif str.lower(orient) == 'h':
            ax = sns.countplot(data=self.dataframe, y=var_categorica, hue=target, palette=self.palette, gap=0.1, dodge=True)            
            plt.xticks(rotation=45, fontsize=8)
            ax.set_xlabel('frequência')
            legend = ax.get_legend()
            legend.set_bbox_to_anchor((1, 1))

            for container in ax.containers:
                ax.bar_label(container, labels = [f'{v.get_width():.0f}' if v.get_width() > 0 else '' for v in container], label_type='edge', fontsize=8)

        plt.title(title)
        sns.despine()
        plt.show()

        display(self._freq_porcentagem_entre_grp_target(var_categorica=var_categorica, target=target))
        return None
    
    def plot_IC95(self, model)-> None:

        summary_df = model.summary2().tables[1].reset_index()
        summary_df.rename(columns={"Coef.": "coef", "[0.025": "lower", "0.975]": "upper"}, inplace=True)
        summary_df = summary_df[summary_df["index"] != "Intercept"]  # remove intercepto
        fig, ax = plt.subplots(figsize=(8,4))

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

    def pearson_corr(self, target:str, var_numericas:list, threshold:float) -> None:
        
        vars = self.dataframe[var_numericas]
        # Plotando o heatmap
        plt.figure(figsize=(16, 8))
        sns.heatmap(vars.corr(numeric_only=True), annot=True, cmap=self._colormap, fmt=".2f",  linecolor='white', linewidths=.3)
        plt.title("Matriz de Correlação")
        plt.show()
        alpha = 0.05

        corr = vars.corr()

        high_corr = (
            corr.where(abs(corr) > 0.7)
                .stack()
                .reset_index()
)
        high_corr.columns = ['X1', 'X2', 'Corr']
        high_corr = high_corr[high_corr['X1'] != high_corr['X2']]
        high_corr.loc[high_corr['Corr'].notna()]


        col_corr = set(high_corr.loc[high_corr['Corr'].notna()]['X1'])  # Set das colunas correlacionadas
                     
        if len(col_corr) > 0:   # Verifica se existem variáveis com correlação acima do threshold   
            print('')
            print(f'Threshold (corte): {threshold}')                         
            print(high_corr.loc[high_corr['Corr'].notna()])

            if self.dataframe[target].dtypes.name not in ['object', 'category', 'str']: # Verifica se o y eh categorico
                print('') 
                print('----- Correlação das variáveis X com y -----')
                list_var = list(col_corr)
                list_var.append(target)

                # Plotando o heatmap
                plt.figure(figsize=(8, 4))
                sns.heatmap(self.dataframe[list_var].corr()[target].to_frame(), annot=True, cmap=self._colormap, fmt=".2f",  linecolor='white', linewidths=.5)
                plt.title("Matriz de Correlação")
                plt.show()
        else :
            print(f'Utilizando o threshold (corte): {threshold}, não foram encontrados casos de multicolinearidade.')
        print('')


