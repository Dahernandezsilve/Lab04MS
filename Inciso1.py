from modeloSIR import ModeloSIR

def Inciso1():
    M, N = 50, 50  # Tamaño del grid
    beta = 0.1     # Probabilidad de contagio
    gamma = 0.25   # Probabilidad de recuperación
    I0 = 2         # Número inicial de infectados
    dias = 100     # Número de días de simulación
    rad = 1        # Radio de interacción

    modelo = ModeloSIR(M, N, beta, gamma, I0, dias, rad)
    modelo.simular()
    modelo.animarGrid()
    modelo.graficarSIR()
    
Inciso1()