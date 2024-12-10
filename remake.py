import numpy as np
import numba
import pygame
from numba import njit, prange

gridSize = 50
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
lightBlue = (0, 150, 212)

windowWidth = 1400
windowHeight = 1000

cellSize = 10
cellDensity = 0.3

gridWidth = windowWidth // cellSize
gridHeight = windowHeight // cellSize

class LgaVisualizer:
    def __init__(self, lga, screen):
        self.lga = lga
        self.screen = screen
        self.previousState = None

    def drawGrid(self, state):
        for i in range(gridHeight):
            for j in range(gridWidth):
                color = lightBlue if state[i, j] > 0 else white
                if self.lga.wall[i, j] == -1:
                    color = red
                pygame.draw.rect(self.screen, color, (j * cellSize, i * cellSize, cellSize, cellSize))

    def update(self):
        self.screen.fill(white)
        state = self.lga.getState()
        self.drawGrid(state)
        self.previousState = state.copy()

class Wall:
    def __init__(self, grid, holePositions=None):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])
        self.holePositions = holePositions
        self.createWallWithHole()

    def createWallWithHole(self):
        wall = np.zeros((self.height, self.width), dtype=int)
        quarterX = self.width // 4

        wall[:, quarterX] = -1

        for hole in self.holePositions:
            wall[hole, quarterX] = 0

        return wall
@njit
def random_initialize(density):
    return [1 if np.random.rand() < density else 0 for _ in range(4)]

# Funkcja kolizji
@njit
def collide(input_data):
    output = input_data.copy()
    if input_data[0] == 1 and input_data[2] == 1 and input_data[1] == 0 and input_data[3] == 0:
        output[0], output[2] = 0, 0
        output[1], output[3] = 1, 1
    elif input_data[1] == 1 and input_data[3] == 1 and input_data[0] == 0 and input_data[2] == 0:
        output[1], output[3] = 0, 0
        output[0], output[2] = 1, 1
    return output

@njit(parallel=True)
def streaming(input_grid, output_grid, wall, height, width):
    new_input_grid = np.zeros_like(input_grid)
    new_output_grid = np.zeros_like(output_grid)

    for i in prange(height):
        for j in prange(width):
            if wall[i, j] == -1:
                continue

            if i > 0 and wall[i - 1, j] != -1:
                new_input_grid[i - 1, j, 0] = output_grid[i, j, 0]
            else:
                new_input_grid[i, j, 2] = output_grid[i, j, 0]

            if i < height - 1 and wall[i + 1, j] != -1:
                new_input_grid[i + 1, j, 2] = output_grid[i, j, 2]
            else:
                new_input_grid[i, j, 0] = output_grid[i, j, 2]

            if j > 0 and wall[i, j - 1] != -1:
                new_input_grid[i, j - 1, 3] = output_grid[i, j, 3]
            else:
                new_input_grid[i, j, 1] = output_grid[i, j, 3]

            if j < width - 1 and wall[i, j + 1] != -1:
                new_input_grid[i, j + 1, 1] = output_grid[i, j, 1]
            else:
                new_input_grid[i, j, 3] = output_grid[i, j, 1]

    return new_input_grid, new_output_grid

class Cell:
    def __init__(self):
        self.input = [0, 0, 0, 0]
        self.output = [0, 0, 0, 0]

    def initialize(self, density=0.1):
        self.input = random_initialize(density)

    def collide(self):
        self.output = collide(self.input)

class Lga:
    def __init__(self, height, width, density=0.1):
        self.height = height
        self.width = width

        self.input_grid = np.zeros((height, width, 4), dtype=np.int32)
        self.output_grid = np.zeros((height, width, 4), dtype=np.int32)

        self.wall = self.createWall(height, width)

        for i in range(height):
            for j in range(width // 4):
                self.input_grid[i, j] = random_initialize(density)

    def createWall(self, height, width):
        wall = np.zeros((height, width), dtype=np.int32)
        quarter_x = width // 4
        wall[:, quarter_x] = -1
        for hole in range(height // 2 - 3, height // 2 + 4):
            wall[hole, quarter_x] = 0
        return wall

    def update(self):
        for i in prange(self.height):
            for j in prange(self.width):
                self.output_grid[i, j] = collide(self.input_grid[i, j])

        self.input_grid, self.output_grid = streaming(self.input_grid, self.output_grid, self.wall, self.height, self.width)

    def getState(self):
        return np.sum(self.input_grid, axis=2)


pygame.init()
screen = pygame.display.set_mode((windowWidth, windowHeight))
pygame.display.set_caption("LGA RB")
clock = pygame.time.Clock()

lga = Lga(height=gridHeight, width=gridWidth, density=cellDensity)

visualizer = LgaVisualizer(lga, screen)

running = True
animating = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if animating:
        lga.update()

    visualizer.update()

    pygame.display.flip()

pygame.quit()
