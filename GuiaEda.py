import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_rows = 10


class AnalisisDatosExploratorio:
    
    def __init__(self, path, sep=None, decimal=".", index_col=0):
        self.__df = self.__cargar_datos(path, sep, decimal, index_col)
        
    @property
    def df(self):
        return self.__df
    
    @df.setter
    def df(self, valor):
        if not isinstance(valor, pd.DataFrame):
            raise TypeError("Debe ser un DataFrame")
        self.__df = valor
    
    def __cargar_datos(self, path, sep=None, decimal=".", index_col=0):
        try:
            if sep is None:
                with open(path, 'r') as f:
                    primera_linea = f.readline()
                    sep = ',' if ',' in primera_linea else ';'
            
            return pd.read_csv(path, sep=sep, decimal=decimal, index_col=index_col)
        except FileNotFoundError:
            raise FileNotFoundError(f"Archivo no encontrado: {path}")
        except Exception as e:
            raise Exception(f"Error al cargar: {str(e)}")
    
    def info_basica(self):
        print("=" * 60)
        print("DATASET INFO")
        print("=" * 60)
        print(f"Shape: {self.__df.shape[0]} filas × {self.__df.shape[1]} columnas")
        print(f"\nTipos:\n{self.__df.dtypes}")
        print(f"\nNulos:\n{self.__df.isnull().sum()}")
        print(f"\nPrimeras filas:\n{self.__df.head()}")
        print("=" * 60)
    
    def estadisticas_descriptivas(self):
        df_num = self.__df.select_dtypes(include=np.number)
        
        print("\n" + "=" * 60)
        print("ESTADÍSTICAS")
        print("=" * 60)
        print(f"\n{df_num.describe()}")
        print(f"\nDesv Est:\n{df_num.std(ddof=0)}")
        print(f"\nAsimetría:\n{df_num.skew()}")
        print("=" * 60)
        
        return {
            'media': df_num.mean(),
            'mediana': df_num.median(),
            'std': df_num.std(ddof=0),
            'var': df_num.var(ddof=0),
            'min': df_num.min(),
            'max': df_num.max(),
        }
    
    def filtrar_numerico(self):
        self.__df = self.__df.select_dtypes(include=np.number)
        return self
    
    def one_hot_encoding(self):
        self.__df = pd.get_dummies(self.__df)
        return self
    
    def visualizar_boxplot(self, figsize=(15, 8), mostrar=True):
        df_num = self.__df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        df_num.boxplot(ax=ax)
        ax.set_title("Boxplot", fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if mostrar:
            plt.show()
        return fig, ax
    
    def visualizar_densidad(self, figsize=(12, 8), mostrar=True):
        df_num = self.__df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        df_num.plot(kind='density', ax=ax)
        ax.set_title("Densidad", fontsize=12, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        if mostrar:
            plt.show()
        return fig, ax
    
    def visualizar_histograma(self, figsize=(14, 8), bins=20, mostrar=True):
        df_num = self.__df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        df_num.hist(bins=bins, ax=ax)
        fig.suptitle("Histogramas", fontsize=12, fontweight='bold')
        plt.tight_layout()
        if mostrar:
            plt.show()
        return fig, ax
    
    def correlaciones(self):
        df_num = self.__df.select_dtypes(include=np.number)
        corr = df_num.corr()
        print("\n" + "=" * 60)
        print("CORRELACIONES")
        print("=" * 60)
        print(corr)
        print("=" * 60)
        return corr
    
    def visualizar_correlacion(self, figsize=(12, 8), mostrar=True):
        df_num = self.__df.select_dtypes(include=np.number)
        corr = df_num.corr()
        
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        paleta = sns.diverging_palette(220, 10, as_cmap=True).reversed()
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=paleta, square=True, 
                    annot=True, fmt='.2f', cbar_kws={'label': 'Corr'}, ax=ax)
        ax.set_title("Correlaciones", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if mostrar:
            plt.show()
        
        return fig, ax
    
    def analisis_completo(self):
        self.info_basica()
        self.estadisticas_descriptivas()
        self.correlaciones()
        print("\nGenerando gráficos...")
        self.visualizar_boxplot()
        self.visualizar_densidad()
        self.visualizar_histograma()
        self.visualizar_correlacion()
        print("✓ Análisis completo")

    
