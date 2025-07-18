import os
import matplotlib.pyplot as plt

class SessionReport:
    """
    Record and generate report for a single session's local map and localization.
    Generates a text summary and a histogram of pole counts per local map.
    """
    def __init__(self, session_name, root_dir):
        self.session_name = session_name
        self.root_dir = root_dir
        # store raw frames and measurement update counts
        self.total_frames = 0
        self.measurement_updates = 0
        # collect list of pole counts per local map
        self.pole_counts = []
        # ensure directory
        os.makedirs(self.root_dir, exist_ok=True)

    def log_local_maps(self, poleparams_list):
        """
        poleparams_list: list of numpy arrays of shape (N_i, 3) per local map
        """
        # record distribution of pole counts
        for params in poleparams_list:
            self.pole_counts.append(params.shape[0])

    def log_localization(self, total_frames, measurement_updates):
        """
        total_frames: total number of frames processed
        measurement_updates: number of measurement updates performed
        """
        self.total_frames = total_frames
        self.measurement_updates = measurement_updates

    def write_report(self):
        # text report
        txt_path = os.path.join(self.root_dir, f"{self.session_name}.txt")
        with open(txt_path, 'w') as f:
            f.write(f"Session: {self.session_name}\n")
            f.write(f"Total frames: {self.total_frames}\n")
            f.write(f"Measurement updates: {self.measurement_updates}\n")
            # count how many local maps have at least 5 poles
            large_pole_maps = sum(1 for c in self.pole_counts if c >= 5)
            f.write(f"Local maps with >=5 poles: {large_pole_maps}\n")
        # histogram
        png_path = os.path.join(self.root_dir, f"{self.session_name}.png")
        plt.figure()
        plt.hist(self.pole_counts, bins=range(0, max(self.pole_counts) + 2))
        plt.xlabel('Number of poles per local map')
        plt.ylabel('Count of local maps')
        plt.title(f'Pole count distribution: {self.session_name}')
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()