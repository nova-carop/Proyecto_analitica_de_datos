# %%
# %%
!pip install funpymodeling
# %%
# importar librerias
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from funpymodeling.exploratory import status
import numpy as np

# cargar el dataframe con la info del csv
df = pd.read_csv(r'C:\Users\mtsll\Desktop\Análisis de datos\GIT\Proyecto_analitica_de_datos\Tema 10.csv')
#df = pd.read_csv(r'C:\Users\Fernanda\Desktop\CPI - Analisis datos\Proyecto\Tema 10.csv')
# %%
# analizar info general del df
df.head()
df.shape
status(df)
# %%
# identificar las filas que estan duplicadas en todas las columnas
df_duplicados = df[df.duplicated()]
df_duplicados
df_duplicados.shape

# eliminar filas duplicadas en el DataFrame original
df.drop_duplicates(inplace=True)

# Restablecer el índice
df.reset_index(drop=True, inplace=True)
df.shape
# %%
##################################### ANALISIS EXPLORATORIO VARIABLES ####################################################

############# DATE
# convertir a datetime el campo date y los valores vacios asignarles NaT (formato: MM-DD-YYYY)
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
print(df['date'].dtypes)

# realizar la interpolacion de la columna date para rellenar los valores faltantes
df['date'] = df['date'].interpolate()

# verificar que no hay fechas vacias
df[df['date'].isna()]

############# QUARTER
df['quarter'].value_counts()

# genero la columna dia a partir del campo date
df['dia'] = df['date'].dt.day

# Generar un boxplot con los dias para cada quarter
data_boxplot_dias_quarter = df[['quarter','dia']]
plt.figure(figsize=(8, 6))
sns.boxplot(x='quarter', y='dia', data=data_boxplot_dias_quarter)
plt.title('Distribucion dias-quarter')
plt.xlabel('quarter')
plt.ylabel('dia')
plt.show() 

# determinar los dias min y max de cada quarter
dia_min_q1 = df[df['quarter'] == 'Quarter1']['dia'].min()
dia_max_q1 = df[df['quarter'] == 'Quarter1']['dia'].max()
dia_min_q2 = df[df['quarter'] == 'Quarter2']['dia'].min()
dia_max_q2 = df[df['quarter'] == 'Quarter2']['dia'].max()
dia_min_q3 = df[df['quarter'] == 'Quarter3']['dia'].min()
dia_max_q3 = df[df['quarter'] == 'Quarter3']['dia'].max()
dia_min_q4 = df[df['quarter'] == 'Quarter4']['dia'].min()
dia_max_q4 = df[df['quarter'] == 'Quarter4']['dia'].max()
print(f'dia_min_q1   {dia_min_q1}')
print(f'dia_max_q1   {dia_max_q1}')
print(f'dia_min_q2   {dia_min_q2}')
print(f'dia_max_q2   {dia_max_q2}')
print(f'dia_min_q3   {dia_min_q3}')
print(f'dia_max_q3   {dia_max_q3}')
print(f'dia_min_q4   {dia_min_q4}')
print(f'dia_max_q4   {dia_max_q4}')

# funcion para rellenar los valores NaN en quarter, en base al valor de dia
def assign_quarter(row):
    if pd.isna(row['quarter']): # solo aplica si quarter es NaN
        day=row['dia']
        if 1 <= day <= 7:
            return 'Quarter1'
        elif 8 <= day <= 14:
            return 'Quarter2'
        elif 15 <= day <= 21:
            return 'Quarter3'
        elif day > 21:
            return 'Quarter4'
    return row['quarter']

# aplicar la funcion "assign_quarter" en la columna quarter para cada fila
df['quarter'] = df.apply(assign_quarter, axis=1)

df['quarter'].value_counts()
df[df['quarter'].isna()]

# quarter: el mes es dividido en 4 por lo que no tiene sentido el dato Quarter5
# analizar que fechas contiene Quarter5
fechas_q5 = df[df['quarter'] == 'Quarter5']['date'].unique()
fechas_q5 

# funcion para modificar los valores de la columna quarter que figuran con el valor Quarter5
def modify_quarter5(row):
    if row['quarter']=='Quarter5': # solo aplica si quarter es Quarter5
        day=row['dia']
        if 1 <= day <= 7:
            return 'Quarter1'
        elif 8 <= day <= 14:
            return 'Quarter2'
        elif 15 <= day <= 21:
            return 'Quarter3'
        elif day > 21:
            return 'Quarter4'
        elif pd.isna(day):
            return np.nan
    return row['quarter']

# aplicar la funcion "modify_quarter5" en la columna quarter para cada fila
df['quarter'] = df.apply(modify_quarter5, axis=1)

# verificar que solo hay quarter 1 a quarter 5
df['quarter'].value_counts()
# verificar que no hay vacios en la columna quarter
df[df['quarter'].isna()]

# Generar un boxplot con los dias para cada quarter con el ajuste realizado
data_boxplot_dias_quarter_2 = df[['quarter','dia']]
plt.figure(figsize=(8, 6))
sns.boxplot(x='quarter', y='dia', data=data_boxplot_dias_quarter_2)
plt.title('Distribucion dias-quarter ajustado')
plt.xlabel('quarter')
plt.ylabel('dia')
plt.show()

############# DAY
df[df['day'].isna()]

# funcion para rellenar los valores NaN en day, en base al valor de date
def assign_day(row):
    if pd.isna(row['day']) and not pd.isna(row['date']): # solo aplica si day es NaN y date no es NaT
        return row['date'].day_name()
    return row['day']

# aplicar la funcion assign_day en la columna day para cada fila
df['day'] = df.apply(assign_day, axis=1)

# verificar si hay vacios
df[df['day'].isna()]

############# DEPARTMENT
df['department'].value_counts()

# Elimina los espacios en blanco al principio y al final de la cadena
df['department'] = df['department'].str.strip()

# verificar el cambio anterior
df['department'].value_counts()

# verificar si hay valores vacios en department
df[df['department'].isna()].shape

############# TEAM
df['team'].value_counts()

# cambiar el tipo a "int" y asignar NaN a valores vacios o al valor "invalid_value" 
df['team'] = pd.to_numeric(df['team'], errors='coerce').astype('Int64') 

# verificar el cambio anterior y analizar los valores de team
df['team'].value_counts().sort_index()

# verificar si hay valores vacios en team
df[df['team'].isna()].shape

# Crear boxplot de team-department
data_boxplot_team_dep = df[['department','team']]
plt.figure(figsize=(8, 6))
sns.boxplot(x='department', y='team', data=data_boxplot_team_dep)
plt.title('Distribucion team-department')
plt.xlabel('Department')
plt.ylabel('Team')
plt.show()

# PENDIENTE VER COMO COMPLETAR INTERPOLAR VALORES VACIOS EN DEPARTMENT Y TEAM

# %%
