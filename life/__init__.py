from pydantic import BaseModel
from typing import Tuple

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.io.wavfile import write

import imageio
import subprocess


GridDimensions = Tuple[int, ...]
GridLocations = Tuple[GridDimensions, ...]


class Simulation(BaseModel):
    grid_dimensions: GridDimensions = ()
    active_cells: GridLocations = ()
    states: list = []
    state_count: dict = {}
    sample_rate: int = 44100
    duration_per_slice: float = 0.1
    gif_file_path: str = "game_of_life_gif.gif"
    video_file_path: str = "game_of_life_video.mp4"
    audio_file_path: str = "game_of_life_audio.wav"
    output_file_path: str = "game_of_life_audio_video.mp4"

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

    def generate_sine_wave(self, frequency: float, duration: float) -> np.ndarray:
        """Generate a sine wave for a given frequency and duration."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        return np.sin(2 * np.pi * frequency * t)

    def set_audio(self) -> None:
        """Map the Game of Life grid to a sound signal and save the audio file."""

        dims = self.grid_dimensions  # (rows, columns) dimensions of the grid
        num_slices = len(self.states)  # Number of time steps (states)

        # Frequency range (200 Hz to 2000 Hz, for example)
        freq_range = np.logspace(np.log10(200), np.log10(2000), dims[0])

        time_per_column = self.duration_per_slice / dims[1]  # Time span for each column
        total_duration = self.duration_per_slice * num_slices  # Total duration of the audio
        sound = np.zeros(
            int(self.sample_rate * total_duration)
        )  # Initialize the entire sound signal

        # Loop over all the states (each time slice)
        for t, state in enumerate(self.states):  # `t` is the index for time slices
            slice_start = int(
                t * self.sample_rate * self.duration_per_slice
            )  # Start index of this time slice

            # Loop over all active cells in this state
            for row, col in state:
                frequency = freq_range[row]  # Map the row to frequency
                wave = self.generate_sine_wave(
                    frequency, time_per_column
                )  # Generate wave for each active cell

                # Determine the correct time offset within the slice
                start = slice_start + int(col * self.sample_rate * time_per_column)
                end = start + len(wave)

                # Add the wave to the sound signal (within the current time slice)
                sound[start:end] += wave

        # Rescale to int16 for audio export
        audio_signal = np.interp(sound, (-1, 1), (-32767, 32767)).astype(np.int16)

        # Save the audio signal to a WAV file
        write(self.audio_file_path, self.sample_rate, audio_signal)

        self.audio_file_path  # Return the path to the audio file

    def set_video(self) -> None:
        """Render the Game of Life states as a GIF and save it as a video."""
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

        plt.close(fig)  # Close the plot

        # Save frames as video
        imageio.mimsave(self.gif_file_path, frames, duration=self.duration_per_slice)

        # Convert the GIF to a video using imageio
        reader = imageio.get_reader(self.gif_file_path)
        fps = 1 / self.duration_per_slice
        writer = imageio.get_writer(self.video_file_path, fps=fps)

        for frame in reader:
            writer.append_data(frame)
        writer.close()

    def combine_audio_video(self) -> None:
        """Use ffmpeg to combine the audio and video into a single file."""
        # Run ffmpeg command to combine audio and video
        command = [
            "ffmpeg",
            "-y",  # Overwrite output file if exists
            "-i",
            self.video_file_path,  # Input video file
            "-i",
            self.audio_file_path,  # Input audio file
            "-c:v",
            "libx264",  # Use H.264 codec for video
            "-c:a",
            "aac",  # Use AAC codec for audio
            "-crf",
            "18",  # Lower CRF for higher quality (default is 23)
            "-g",
            "10",  # Force a keyframe every 10 frames
            "-pix_fmt",
            "yuv420p",  # Ensure compatibility with most players
            "-shortest",  # Ensure the output matches the shorter stream (audio or video)
            self.output_file_path,  # Output file
        ]

        subprocess.run(command)

    def generate_synced_video(self):
        """Generate the synced video with audio and video combined."""
        # Render the video
        self.set_video()

        # Generate the audio
        self.set_audio()

        # Combine the audio and video
        self.combine_audio_video()


class Rules(BaseModel):
    die_when_under: int = 2
    die_when_over: int = 3
    dead_reproduction: int = 3
