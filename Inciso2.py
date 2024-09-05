import numpy as np
import matplotlib.pyplot as plt
from modeloSIR import ModeloSIR

def simular_multiples_experimentos(M, N, beta, gamma, I0, dias, rad, Nexp):
    # Inicializar acumuladores para S, I, R
    acumuladoS = np.zeros((dias,))
    acumuladoI = np.zeros((dias,))
    acumuladoR = np.zeros((dias,))
    
    # Realizar Nexp simulaciones
    for _ in range(Nexp):
        modelo = ModeloSIR(M, N, beta, gamma, I0, dias, rad)
        modelo.simular()
        
        # Acumular los resultados de S, I, R de cada simulación
        for t in range(dias):
            S, I, R = modelo.historialSIR[t]
            acumuladoS[t] += S
            acumuladoI[t] += I
            acumuladoR[t] += R
    
    # Calcular el promedio de S, I, R
    promedioS = acumuladoS / Nexp
    promedioI = acumuladoI / Nexp
    promedioR = acumuladoR / Nexp
    
    # Graficar los resultados promedio
    plt.figure(figsize=(10, 6))
    plt.plot(promedioS, label='Susceptibles (Promedio)')
    plt.plot(promedioI, label='Infectados (Promedio)')
    plt.plot(promedioR, label='Recuperados (Promedio)')
    plt.xlabel('Días')
    plt.ylabel('Población')
    plt.legend()
    plt.title(f'Modelo SIR - Promedio de {Nexp} Simulaciones\n(beta={beta}, gamma={gamma})')
    plt.show()

def experimentar_con_parametros(M, N, I0, dias, rad, Nexp, betas, gammas):
    # Realizar simulaciones para diferentes valores de beta y gamma
    for beta in betas:
        for gamma in gammas:
            print(f"Simulando para beta={beta} y gamma={gamma}...")
            simular_multiples_experimentos(M, N, beta, gamma, I0, dias, rad, Nexp)

# Ejemplo de uso
if __name__ == '__main__':
    M, N = 50, 50  # Tamaño del grid
    I0 = 5         # Número inicial de infectados
    dias = 100     # Número de días de simulación
    rad = 1        # Radio de interacción
    Nexp = 50      # Número de experimentos
    
    # Valores de parámetros beta y gamma a experimentar
    betas = [0.2, 0.3, 0.4]
    gammas = [0.05, 0.1, 0.2]
    
    # Realizar experimentos
    experimentar_con_parametros(M, N, I0, dias, rad, Nexp, betas, gammas)
