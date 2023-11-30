import os
import torch
import multiprocessing
import psutil


def determine_num_workers():
    num_cores = multiprocessing.cpu_count()   # CPU cores
    num_gpus = torch.cuda.device_count()      # Number of GPUs

    # Available virtual memory (we will consider 80% of it)
    virtual_memory = psutil.virtual_memory().available * 0.8

    # Estimate the size of our dataset
    dataset_size = os.path.getsize('path/to/your/dataset')  # Substitute with your dataset path

    # The amount of memory anticipated to be needed per worker
    memory_per_worker = dataset_size / min(num_cores, num_gpus * 4)

    # Determine the memory capacity
    memory_capacity = virtual_memory / memory_per_worker

    # Disk speed factor (a constant that would be determined by disk speed, 1.0 for SSD, less for HDD)
    disk_speed_factor = 1.0 

    num_workers = min(num_cores, memory_capacity, num_gpus * 4, disk_speed_factor * num_cores)

    return int(num_workers)


num_workers = determine_num_workers()