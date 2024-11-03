#%%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from funpymodeling.exploratory import status

#%%

df = pd.read_csv(r'C:\Users\mtsll\Desktop\Análisis de datos\GIT\Proyecto_analitica_de_datos\Tema 10.csv')


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
