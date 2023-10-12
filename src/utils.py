import re
from pathlib import Path
from datetime import datetime

def find_latest_best_checkpoint(base_dir):
    # Recursively find all 'lightning_logs' directories
    all_runs = list(Path(base_dir).rglob('*'))  # retrieve all files and folders
    all_runs = [run for run in all_runs if run.is_dir() and 'lightning_logs' in str(run)]  # filter only relevant directories

    # Function to convert the directory name to a datetime object
    def get_datetime_from_dir(dir):
        dir_name = str(dir)  # convert Path object to string
        match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', dir_name)
        if match:
            time_str = match.group(0)
            # Replace '_' with space and the first two '-' with ':'
            time_str = time_str.replace('_', ' ').replace('-', ':', 2)
            # Now replace the remaining '-' with ':'
            time_str = time_str.replace('-', ':')
            try:
                date_time_obj = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
                # Debug print to show the extracted datetime
                return date_time_obj
            except ValueError as e:
                # Debug print to show directories that failed the datetime extraction
                return datetime.min  # return the smallest possible datetime for directories that don't match
        else:
            # Debug print to show directories that failed the datetime extraction
            return datetime.min  # return the smallest possible datetime for directories that don't match


    # Sort the runs based on the datetime and get the latest
    latest_run = max(all_runs, key=lambda dir: get_datetime_from_dir(dir))

    # Find the 'best.ckpt' file in the latest run directory
    best_checkpoint = next(Path(latest_run).rglob('*best.ckpt'), None)

    if best_checkpoint:
        return best_checkpoint
    else:
        print(f"No best checkpoint found in the latest run at: {latest_run}")
        return None

