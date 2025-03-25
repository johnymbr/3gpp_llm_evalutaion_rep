from pathlib import Path
from source.process_dataset import ProcessDataset

if __name__ == '__main__':
    output_path_str = "../files/script"
    Path(output_path_str).mkdir(parents=True, exist_ok=True)

    process_dataset = ProcessDataset("../datasets/TeleQnA/TeleQnA.txt", output_path_str)
    process_dataset.process_teleqna()
