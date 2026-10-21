import numpy as np
from scipy.stats import multivariate_normal

#definir los parametros de cada clase según el enunciado

# Clase 1
mu1 = np.array([0, 0])
cov1 = np.array([[1, 0.5], 
                 [0.5, 1]])

# Clase 2
mu2 = np.array([4, 4])
cov2 = np.array([[1, -0.3], 
                 [-0.3, 1]])

# Clase 3
mu3 = np.array([-4, 3])
cov3 = np.array([[0.5, 0], 
                 [0, 2]])

#funcion clasificadora
def clasificar_gaussiana(x):
    #Calcular la funcion de densidad de probabilidad (pdf) para la muestra x en cada clase
    prob_clase1 = multivariate_normal.pdf(x, mean=mu1, cov=cov1)
    prob_clase2 = multivariate_normal.pdf(x, mean=mu2, cov=cov2)
    prob_clase3 = multivariate_normal.pdf(x, mean=mu3, cov=cov3)
    
    #Agrupar las probabilidades en una lista
    probabilidades = [prob_clase1, prob_clase2, prob_clase3]
    
    #Encontrar el indice de la probabilidad mas alta
    #np.argmax devuelve contando el 0
    clase_ganadora = np.argmax(probabilidades) + 1
    
    return clase_ganadora, probabilidades

#crear un conjunto de muestras de prueba (pares de coordenadas x1, x2)
muestras_prueba = np.array([
    [0.5, 0.2],   # Debería tender a la Clase 1 (cerca al 0,0)
    [3.5, 4.2],   # Debería tender a la Clase 2 (cerca al 4,4)
    [-3.8, 2.9],  # Debería tender a la Clase 3 (cerca al -4,3)
    [10, -10]     # Un punto muy lejano para ver cómo se comporta
])

#clasificar iterativamente
print("--- Resultados de Clasificacion Gaussiana ---")
for i, muestra in enumerate(muestras_prueba):
    clase, probs = clasificar_gaussiana(muestra)
    print(f"Muestra {i+1} {muestra}:")
    print(f"  -> Probabilidades: C1={probs[0]:.6f}, C2={probs[1]:.6f}, C3={probs[2]:.6f}")
    print(f"  -> Asignada a: Clase {clase}\n")