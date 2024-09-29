from pydantic import BaseModel
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import numpy as np

GridDimensions = Tuple[int, ...]
GridLocations = Tuple[GridDimensions, ...]


class Simulation(BaseModel):
    grid_dimensions: GridDimensions = ()
    active_cells: GridLocations = ()
    states: list = []
    state_count: dict = {}

    def set_initial_state(self, grid_locations: GridLocations) -> None:
        if isinstance(grid_locations[0], tuple):
            self.active_cells = grid_locations
        else:
            self.active_cells = (grid_locations,)
        self.states = [self.active_cells]
        self.state_count[self.active_cells] = 1

    def get_neighbors(self, x: int, y: int, current_grid: GridLocations) -> GridLocations:
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if 0 <= x + i < self.grid_dimensions[0] and 0 <= y + j < self.grid_dimensions[1]:
                    neighbor_state = (x + i, y + j) in current_grid
                    neighbors.append((x + i, y + j, neighbor_state))
        return tuple(neighbors)

    def step(self, rules: "Rules") -> None:
        current_grid = [c for c in self.active_cells]
        for i in range(self.grid_dimensions[0]):
            for j in range(self.grid_dimensions[1]):
                neighbors = self.get_neighbors(i, j, current_grid)
                alive_neighbors = len([n for n in neighbors if n[2]])
                # Check if the cell is alive
                if (i, j) in self.active_cells:
                    # Check if the cell should die
                    if (
                        alive_neighbors < rules.die_when_under
                        or alive_neighbors > rules.die_when_over
                    ):
                        self.active_cells = tuple(
                            [cell for cell in self.active_cells if cell != (i, j)]
                        )
                else:
                    # Check if the cell should be born
                    if alive_neighbors == rules.dead_reproduction:
                        self.active_cells += ((i, j),)

    def run(self, steps: int, rules: "Rules") -> None:
        for _ in range(steps):
            self.step(rules)
            if self.active_cells not in self.state_count:
                self.state_count[self.active_cells] = 1
            else:
                self.state_count[self.active_cells] += 1
            self.states.append(self.active_cells)
            # If every state has been visited twice, we have reached a stable state
            if all([count == 2 for count in self.state_count.values()]):
                break
            # If the current state is the same as the previous state, we have reached a stable state
            if len(self.states) > 2:
                if self.active_cells == self.states[-2]:
                    break

    def get_sim_as_text(self) -> str:
        # Create an array of the grid dimensions filled with spaces
        # with "|" as left/right border and "-" as upper/lower border
        grid = [
            [" " for _ in range(self.grid_dimensions[0])] for _ in range(self.grid_dimensions[1])
        ]
        for location in self.active_cells:
            grid[location[1]][location[0]] = "X"
        out = "\n".join(["|" + "".join(row) + "|" for row in grid])
        out = "-" * len(out.split("\n")[0]) + "\n" + out + "\n" + "-" * len(out.split("\n")[0])
        return out

    def render_as_text(self, all_states: bool = False) -> None:
        if not all_states:
            print(self.get_sim_as_text())
        else:
            for state in self.states:
                self.active_cells = state
                print(self.get_sim_as_text())
                print("\n")

    def render_gif(self):
        frames = []
        fig, ax = plt.subplots()

        for step_number, active_cells in enumerate(self.states):
            # Create a grid for this step
            grid = np.zeros(self.grid_dimensions)
            for cell in active_cells:
                grid[cell] = 1  # Mark the active cells

            # Plot the grid
            ax.clear()
            ax.imshow(grid.T, cmap="binary")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Step {step_number}")

            # Capture the frame
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(Image.fromarray(image))

        # Save the frames as a GIF
        frames[0].save(
            "./temp_sim.gif", save_all=True, append_images=frames[1:], duration=200, loop=0
        )


class Rules(BaseModel):
    die_when_under: int = 2
    die_when_over: int = 3
    dead_reproduction: int = 3
