import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ModeloSIR:
    def __init__(self, M, N, beta, gamma, posiciones_infectadas, dias, rad):
        self.M = M
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.posiciones_infectadas = posiciones_infectadas
        self.dias = dias
        self.rad = rad
        self.grid = self.inicializarGrid()
        self.historialGrid = []
        self.historialSIR = []

    # Método para inicializar el grid con posiciones infectadas predefinidas
    def inicializarGrid(self):
        grid = np.zeros((self.M, self.N), dtype=int)
        for i, j in self.posiciones_infectadas:
            grid[i, j] = 1  # Marcamos la célula como infectada
        return grid

    # Método para obtener la vecindad de una celda (i, j)
    def obtenerVecindad(self, i, j):
        return self.grid[max(0, i - self.rad):min(self.M, i + self.rad + 1), max(0, j - self.rad):min(self.N, j + self.rad + 1)]

    # Método para actualizar el grid según las reglas del modelo SIR
    def actualizarGrid(self):
        nuevoGrid = self.grid.copy()
        for i in range(self.M):
            for j in range(self.N):
                if self.grid[i, j] == 0:  # Susceptible
                    vecindad = self.obtenerVecindad(i, j)
                    infectados = np.sum(vecindad == 1)
                    if infectados > 0:
                        prob_infeccion = 1 - (1 - self.beta) ** infectados
                        if random.random() < prob_infeccion:
                            nuevoGrid[i, j] = 1
                elif self.grid[i, j] == 1:  # Infectado
                    if random.random() < self.gamma:
                        nuevoGrid[i, j] = 2  # Recuperado
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

# Generar un promedio de la dinámica espacial
def calcular_grid_promedio(historial_grids):
    return np.mean(historial_grids, axis=0)

# Generar el video de la dinámica promedio
def generar_video(historial_promedio, nombre_video='dinamica_promedio.mp4'):
    fig, ax = plt.subplots()
    img = ax.imshow(historial_promedio[0], cmap='viridis', vmin=0, vmax=2)

    def actualizar(frame):
        img.set_data(historial_promedio[frame])
        ax.set_title(f'Día {frame}')
        return [img]

    ani = animation.FuncAnimation(fig, actualizar, frames=len(historial_promedio), blit=False)
    writer = animation.FFMpegWriter(fps=5)
    ani.save(nombre_video, writer=writer)

def graficar_promedio_sir(grids_historial, dias, M, N, Nexp, intervalo=10):
    S_prom = np.zeros(dias)
    I_prom = np.zeros(dias)
    R_prom = np.zeros(dias)

    for t in range(dias):
        S_prom[t] = np.mean([np.sum(grid == 0) for grid in grids_historial[:, t]])
        I_prom[t] = np.mean([np.sum(grid == 1) for grid in grids_historial[:, t]])
        R_prom[t] = np.mean([np.sum(grid == 2) for grid in grids_historial[:, t]])

    # Ajustar el tamaño de la figura para evitar compresión
    plt.figure(figsize=(10, 6))  # Puedes ajustar el tamaño aquí
    
    # Graficar la evolución promedio
    plt.plot(S_prom, label='Susceptibles Promedio')
    plt.plot(I_prom, label='Infectados Promedio')
    plt.plot(R_prom, label='Recuperados Promedio')
    
    # Etiquetas de los ejes
    plt.xlabel('Días')
    plt.ylabel('Población Promedio')

    # Título y leyenda
    plt.legend()
    plt.title(f'Promedio de la Evolución SIR - Nexp Experimentos {Nexp}')

    # Configuración para mostrar etiquetas en el eje x cada "intervalo" días
    plt.xticks(np.arange(0, dias, step=intervalo))

    # Ajustar el layout para que las etiquetas no se corten
    plt.tight_layout()

    # Mostrar la gráfica
    plt.show()


def guardar_snapshots(historial_promedio, dias_snapshot, nombre_base='snapshot'):
    for dia in dias_snapshot:
        plt.figure(figsize=(6, 6))
        plt.imshow(historial_promedio[dia], cmap='viridis', vmin=0, vmax=2)
        plt.title(f'Snapshot del Día {dia}')
        plt.colorbar(label='Estado (0=Susceptible, 1=Infectado, 2=Recuperado)')
        plt.savefig(f'{nombre_base}_dia_{dia}.png')  # Guardar la imagen
        plt.close()


# Ejemplo de uso
if __name__ == '__main__':
    M, N = 50, 50  # Tamaño del grid
    beta = 0.3     # Probabilidad de contagio
    gamma = 0.1    # Probabilidad de recuperación
    posiciones_infectadas = [(10, 10), (20, 20), (30, 30), (40, 40), (25, 25)]  # Lista de posiciones infectadas
    dias = 100     # Número de días de simulación
    rad = 1        # Radio de interacción
    Nexp = 10      # Número de experimentos

    grids_historial = []

    # Realizar Nexp experimentos
    for exp_num in range(Nexp):
        print(f'Ejecutando experimento {exp_num + 1}/{Nexp}')
        modelo = ModeloSIR(M, N, beta, gamma, posiciones_infectadas, dias, rad)
        modelo.simular()
        grids_historial.append(modelo.historialGrid)

    grids_historial = np.array(grids_historial)
    
    # Calcular el grid promedio
    grid_promedio = calcular_grid_promedio(grids_historial)

    # Generar el video de la dinámica espacial promedio
    generar_video(grid_promedio)

    # Días específicos para capturar snapshots (por ejemplo, cada 10 días)
    dias_snapshot = np.arange(0, dias, step=10)

    # Generar y guardar los snapshots
    guardar_snapshots(grid_promedio, dias_snapshot)

    # Graficar la evolución promedio de S, I, R
    graficar_promedio_sir(grids_historial, dias, M, N, Nexp, intervalo=10)
