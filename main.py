import numpy as np
import pygame
from numba import njit, prange

# Parametry symulacji
N = 50  # Rozmiar siatki
grid = np.zeros((N, N, 4), dtype=np.int32)

# Funkcja inicjalizująca siatkę z przegrodą i cząsteczkami
def initialize_grid():
    grid = np.zeros((N, N, 4), dtype=np.int32)
    grid[:, N // 3:N // 3 + 1, :] = -1  # Przegroda
    grid[N // 3 - 2:N // 3 + 2, N // 3, :] = 0  # Otwór w przegrodzie

    # Wypełniamy cząsteczkami lewą stronę
    grid[:, :N // 3, 0] = np.random.choice([0, 1], size=(N, N // 3))
    grid[:, :N // 3, 1] = np.random.choice([0, 1], size=(N, N // 3))
    grid[:, :N // 3, 2] = np.random.choice([0, 1], size=(N, N // 3))
    grid[:, :N // 3, 3] = np.random.choice([0, 1], size=(N, N // 3))
    return grid

grid = initialize_grid()
@njit(parallel=True)
def update(grid):
    """Pełna aktualizacja siatki z prostym odbijaniem przy kolizjach."""
    N = grid.shape[0]
    new_grid = np.zeros((N, N, 4), dtype=np.int32)

    # Pierwsza faza: Ruch cząsteczek bez kolizji
    for i in prange(N):
        for j in range(N):
            if grid[i, j, 0] == -1:  # Ściany
                new_grid[i, j, :] = -1
                continue

            # Ruch w górę
            if grid[i, j, 0] == 1:
                if i > 0 and grid[i - 1, j, 0] != -1:
                    new_grid[i - 1, j, 0] += 1
                else:
                    new_grid[i, j, 1] += 1  # Odbicie w dół

            # Ruch w dół
            if grid[i, j, 1] == 1:
                if i < N - 1 and grid[i + 1, j, 1] != -1:
                    new_grid[i + 1, j, 1] += 1
                else:
                    new_grid[i, j, 0] += 1  # Odbicie w górę

            # Ruch w lewo
            if grid[i, j, 2] == 1:
                if j > 0 and grid[i, j - 1, 2] != -1:
                    new_grid[i, j - 1, 2] += 1
                else:
                    new_grid[i, j, 3] += 1  # Odbicie w prawo

            # Ruch w prawo
            if grid[i, j, 3] == 1:
                if j < N - 1 and grid[i, j + 1, 3] != -1:
                    new_grid[i, j + 1, 3] += 1
                else:
                    new_grid[i, j, 2] += 1  # Odbicie w lewo

    for i in prange(N):
        for j in range(N):
            if new_grid[i, j, 0] > 0 and new_grid[i, j, 2] > 0:
                new_grid[i, j, 1], new_grid[i, j, 3] = new_grid[i, j, 0], new_grid[i, j, 2]
                new_grid[i, j, 0], new_grid[i, j, 2] = 0,0

            if new_grid[i, j, 1] > 0 and new_grid[i, j, 3] > 0:
                new_grid[i, j, 0], new_grid[i, j, 2] = new_grid[i, j, 1], new_grid[i, j, 3]
                new_grid[i, j, 1], new_grid[i, j, 3] = 0,0

    return new_grid




# Funkcja renderująca
def draw_grid(screen, grid):
    for i in range(N):
        for j in range(N):
            if grid[i, j, 0] == 1 or grid[i, j, 1] == 1 or \
               grid[i, j, 2] == 1 or grid[i, j, 3] == 1:
                pygame.draw.rect(screen, (255, 255, 255), (j*10, i*10, 10, 10))
            elif grid[i, j, 0] == -1:
                pygame.draw.rect(screen, (0, 0, 255), (j*10, i*10, 10, 10))

# Pętla symulacji
pygame.init()
screen = pygame.display.set_mode((N * 10, N * 10))
pygame.display.set_caption("Symulacja gazu - Numba")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))
    grid = update(grid)
    draw_grid(screen, grid)
    pygame.display.flip()
pygame.quit()
