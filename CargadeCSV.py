#%%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from funpymodeling.exploratory import status

#%%

# cargar el dataframe con la info del csv de manera generica
df = pd.read_csv('Tema_10.csv')

# %%
df.head()
status(df)
# %%

#Conversión de tipos de variables

# convertir a datetime el campo date y los valores vacios asignarles NaT (Formato de dataset MM-DD-YYYY )
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
 
 # Convertir a Catergorica el tipo de variable de quarter
 df['quarter'] = 

# 
# %%
#Anális  de Quarter
df['quarter'].value_counts()

fechas_q5 = df[df['quarter'] == 'Quarter5']['date'].unique()
print(fechas_q5)
# %%
################## Data Profiling ###########

# Importaciones

from ydata_profiling import ProfileReport

# Se lee el csv
df = pd.read_csv(‘dataset.csv')
# Se ejecuta reporte
profile = ProfileReport(df, title="Reporte", explorative=True)
# Se guarda reporte como html
profile.to_file(output_file="Profiling_html.html")
# Se guarda reporte como json
profile.to_file(output_file="Profiling_json.json”)





# %%
########## Análisis SMV ###################
#Verificar escala de los valores SMV
df.plot(x="date", y=["smv"])
#De acá sale que hay puntos muy altos a eliminar o mover la coma de lugar (dividr entre 10)
def dividir_si_mayor_a_100_inplace(df, columna):
    df[columna] = df[columna].apply(lambda x: x / 10 if x > 100 else x)

#No se si tiene sentido cambiar a 0 los valores faltantes. Si tiene sentido es con esta funci[on]
df['smv'].fillna(0, inplace=True)
# %%
########## Análisis WIP ###################

df.plot(x="date", y=["wip"])
# Aca vi que hay uno o m[as puntos con un valor muy alto]
#ver el valor m[aximo]
df.max('wip')

# %%
#hay 8 valores repetidos, podr[ia ser aunque raro]
valores_repetidos = df['wip'].value_counts()[df['wip'].value_counts() > 1]
print(valores_repetidos)



# %%
