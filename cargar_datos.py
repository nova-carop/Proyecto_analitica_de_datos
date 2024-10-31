import pandas as pd
from limpieza_datos import limpiar_datos  

#Se carga el archivo csv y se pasa a dataframe

def cargar_csv(ruta, encoding='utf-8', na_values=None):
    df = pd.read_csv(ruta, encoding=encoding, na_values=na_values)
    return df

# Código principal
if __name__ == "__main__":

    #Poner el archivo en la carpeta de entraga y usar esa ruta fija
    ruta = 'C:\Users\carol\OneDrive\Escritorio\UM\Segundo Semestre 2024\Proyecto Posgrado\garments_worker_productivity.csv' 
    df = cargar_csv(ruta, na_values=['-', 'N/A', 'n/a'])
    
    #Muestra el dataframe para ver si se cargo bien (Es solo para depurgacion)
    print("DataFrame original:")
    print(df.head())
    
    # Llamar a la función de limpieza y pasar el DataFrame
    df_limpio = limpiar_datos(df)
    print("\nDataFrame después de la limpieza:")
    print(df_limpio.head())
