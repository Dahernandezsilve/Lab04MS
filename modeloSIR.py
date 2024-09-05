import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ModeloSIR:
    def __init__(self, M, N, beta, gamma, I0, dias, rad):
        self.M = M
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.I0 = I0
        self.dias = dias
        self.rad = rad
        self.grid = self.inicializarGrid()
        self.historialGrid = []
        self.historialSIR = []

    # Método para inicializar el grid con una cantidad de infectados de manera aleatoria
    def inicializarGrid(self):
        grid = np.zeros((self.M, self.N), dtype=int)
        infectadas = random.sample([(i, j) for i in range(self.M) for j in range(self.N)], self.I0)
        for i, j in infectadas:
            grid[i, j] = 1  
        return grid

    # Método para obtener la vecindad de una celda (i, j)
    def obtenerVecindad(self, i, j):
        return self.grid[max(0, i - self.rad):min(self.M, i + self.rad + 1), max(0, j - self.rad):min(self.N, j + self.rad + 1)]

    # Método para actualizar el grid según las reglas del modelo SIR
    def actualizarGrid(self):
        nuevoGrid = self.grid.copy()
        for i in range(self.M):
            for j in range(self.N):
                if self.grid[i, j] == 0:  # Susceptible - Morado
                    vecindad = self.obtenerVecindad(i, j)
                    infectados = np.sum(vecindad == 1)
                    if infectados > 0:
                        prob_infeccion = 1 - (1 - self.beta) ** infectados
                        if random.random() < prob_infeccion:
                            nuevoGrid[i, j] = 1
                elif self.grid[i, j] == 1:  # Infectado - Amarillo
                    if random.random() < self.gamma:
                        nuevoGrid[i, j] = 2  # Recuperado - Verde
        self.grid = nuevoGrid

    # Método para simular el modelo SIR durante un número de días especificado
    def simular(self):
        for _ in range(self.dias):
            self.historialGrid.append(self.grid.copy())
            self.actualizarGrid()

            # Almacena la cantidad de S, I y R
            S = np.sum(self.grid == 0)
            I = np.sum(self.grid == 1)
            R = np.sum(self.grid == 2)
            self.historialSIR.append([S, I, R])

    # Método para mostrar la evolución temporal del modelo SIR
    def graficarSIR(self):
        sir = np.array(self.historialSIR)
        plt.plot(sir[:, 0], label='Susceptibles')
        plt.plot(sir[:, 1], label='Infectados')
        plt.plot(sir[:, 2], label='Recuperados')
        plt.xlabel('Días')
        plt.ylabel('Población')
        plt.legend()
        plt.title('Modelo SIR - Evolución Temporal')
        plt.show()


    # Método para mostrar la evolución del contagio
    def animarGrid(self):
        fig, ax = plt.subplots()
        img = ax.imshow(self.historialGrid[0], cmap='viridis', vmin=0, vmax=2)

        def actualizar(frame):
            img.set_data(self.historialGrid[frame])
            ax.set_title(f'Día {frame}')
            return [img]

        ani = animation.FuncAnimation(fig, actualizar, frames=len(self.historialGrid), blit=False)
        plt.show()


# Ejemplo de uso
if __name__ == '__main__':
    M, N = 50, 50  # Tamaño del grid
    beta = 0.3     # Probabilidad de contagio
    gamma = 0.1    # Probabilidad de recuperación
    I0 = 5         # Número inicial de infectados
    dias = 100     # Número de días de simulación
    rad = 1        # Radio de interacción

    modelo = ModeloSIR(M, N, beta, gamma, I0, dias, rad)
    modelo.simular()
    modelo.animarGrid()
    modelo.graficarSIR()
