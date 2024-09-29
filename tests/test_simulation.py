import unittest

from life import Simulation


class TestSimulation(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_simulation(self):
        grid_dimensions = (10, 10)
        sim = Simulation(grid_dimensions=grid_dimensions)
        error_str = f"Expected grid_dimensions {grid_dimensions}; got {sim.grid_dimensions}"
        self.assertEqual(sim.grid_dimensions, grid_dimensions, error_str)

    def test_single_cell(self):
        grid_dimensions = (2, 2)
        sim = Simulation(grid_dimensions=grid_dimensions)
        sim.set_initial_state([(0, 0)])
        expected = "----\n|X |\n|  |\n----"
        self.assertEqual(sim.get_sim_as_text(), expected)
