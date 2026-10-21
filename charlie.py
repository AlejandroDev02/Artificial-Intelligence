import numpy as np

#definir el conjunto de referencia ("base de datos" historica)
#simulamos 5 muestras, cada una con 3 características (x, y, z)
conjunto_referencia = np.array([
    [1.0, 2.0, 0.5],
    [4.0, 6.0, 1.2],
    [-2.0, 3.5, 0.0],
    [8.0, 1.0, -1.5],
    [0.5, 0.5, 0.5]
])

#definir el conjunto de muestras dadas (los puntos nuevos a evaluar)
muestras_dadas = np.array([
    [1.2, 1.8, 0.6],  # Fisicamente cerca de la muestra 0 del conjunto de referencia
    [7.5, 1.5, -1.0]  # Fisicamente cerca de la muestra 3 del conjunto de referencia
])

#funcion para encontrar la muestra más cercana
def encontrar_mas_cercanas(referencias, nuevas_muestras):
    resultados = []
    
    
    #iterar sobre el conjunto de muestras dadas
    for i, muestra in enumerate(nuevas_muestras):
        
        #calcular distancia euclidiana entre esta muestra y TODAS las de referencia
        #axis=1 indica que el calculo se hace fila por fila
        distancias = np.linalg.norm(referencias - muestra, axis=1)
        
        #encontrar el indice de la distancia minima
        indice_mas_cercano = np.argmin(distancias)
        distancia_minima = distancias[indice_mas_cercano]
        
        resultados.append({
            'muestra_dada': i,
            'coordenadas': muestra,
            'indice_mas_cercano': indice_mas_cercano,
            'coordenadas_cercanas': referencias[indice_mas_cercano],
            'distancia': distancia_minima
        })
        
    return resultados

#mostrar resultados
print("--- Busqueda de Muestras Mas Cercanas ---")
resultados_busqueda = encontrar_mas_cercanas(conjunto_referencia, muestras_dadas)

for res in resultados_busqueda:
    print(f"\nPara la muestra dada {res['muestra_dada']} con valores {res['coordenadas']}:")
    print(f" -> La muestra mas cercana es la del indice [{res['indice_mas_cercano']}] ({res['coordenadas_cercanas']})")
    print(f" -> Distancia euclidiana: {res['distancia']:.4f}")