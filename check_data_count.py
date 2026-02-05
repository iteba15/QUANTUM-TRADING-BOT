
import pickle
from pathlib import Path

def check_data():
    data_dir = Path('training_data')
    for pkl in data_dir.glob('*_snapshots.pkl'):
        try:
            with open(pkl, 'rb') as f:
                data = pickle.load(f)
                snapshots = data.get('snapshots', [])
                labels = data.get('labels', [])
                valid_labels = [l for l in labels if l is not None]
                print(f"{pkl.name}: {len(snapshots)} snapshots, {len(valid_labels)} labeled.")
        except Exception as e:
            print(f"Error reading {pkl.name}: {e}")

if __name__ == "__main__":
    check_data()
