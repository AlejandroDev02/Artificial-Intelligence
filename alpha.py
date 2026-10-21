import numpy as np

#Definir vectores
w = np.array([1/3, 1, -3, -2, 5/4])
b = 2/3

#Funcion de clasificacion
def clasificar_muestra(x, w, b):
    #calcular producto punto entre la muestra x y el vector w y sumar sesbo
    f_x = np.dot(w, x) + b

    #evaluar el resultado para asignar la clase
    if f_x >= 0:
        return 1 #clase positiva
    else:
        return 0 #clase negativa


#vectores x muestra
conjunto_muestras = np.array([
    [3, 0, 1, -1, 4],   # Muestra 1
    [0, -2, 2, 1, 0],   # Muestra 2
    [-3, 1, -1, 0, 2]   # Muestra 3
])

#iterar por cada vector x y clasificar
print("Resultados clasificacion:")
for i, muestra in enumerate(conjunto_muestras):
    clase = clasificar_muestra(muestra, w, b)
    print(f"Muestra {i+1} {muestra} -> Pertenece a la Clase {clase}")