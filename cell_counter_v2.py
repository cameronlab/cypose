import os
import numpy as np
import imageio
import time
from pathlib import Path
from scipy.ndimage import label
from concurrent.futures import ProcessPoolExecutor


class CellCounter:
    def __init__(self, directory=None):
        """Set the directory where images are"""
        self.path = Path(directory) if directory else Path(__file__).parent

    def process_tif_file(self, filename):
        """Open and process tif files."""
        tif_file = imageio.mimread(filename)

        # Convert to a numpy array
        frame_array = np.array(tif_file)

        # Label connected components
        labeled_array, num_features = label(frame_array)

        # Return the number of connected components
        return num_features

    def count_cells(self):
        """Use multiprocessing to count the cells in the tif files."""
        # Get the list of tif files
        tif_files = list(self.path.glob('*.tif'))

        # Time it
        start_time = time.time()

        # Run in paralelle
        with ProcessPoolExecutor() as executor:
            # Map the function to each file
            results = executor.map(self.process_tif_file, tif_files)

            # Sum the cell counts from all files
            total_cell_count = sum(results)
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(f'It took {elapsed_time:.2f} seconds to calculate the cell count')
            print(f'Total cell count is {total_cell_count}')

        return total_cell_count


if __name__ == "__main__":
    counter = CellCounter()
    counter.count_cells()
