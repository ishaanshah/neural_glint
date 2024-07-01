import torch
import subprocess
import sys
import os
from typing import List
from tqdm import tqdm

def get_num_gpus() -> int:
    return torch.cuda.device_count()

class GPUScheduler:
    def __init__(self, jobs: List[dict], stdout: List[str]=None, stderr: List[str]=None) -> None:
        self.jobs = jobs
        self.num_gpus = get_num_gpus()

        self.stdout = stdout
        self.stderr = stderr

        assert self.num_gpus, "No GPUs found"
        print(f"Found {self.num_gpus} GPUs to run experiments on")

    def run(self, debug: bool=False) -> List[int]:
        gpu = 0
        processes = []
        returncodes = []
        for i in tqdm(range(len(self.jobs))):
            job = self.jobs[i]
            env = { **os.environ }
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            if not "env" in job:
                job["env"] = env
            else:
                job["env"].update(env)

            # Open files for stderr and stdout
            if debug:
                stdout = sys.stdout
                stderr = sys.stderr
            else:
                stdout = subprocess.DEVNULL
                if self.stdout and self.stdout[i]:
                    stdout = open(self.stdout[i], "w")

                stderr = stdout
                if self.stderr and self.stderr[i]:
                    stderr = open(self.stdout[i], "w")

            process = subprocess.Popen(**job, stdout=stdout, stderr=stderr)
            if debug:
                # Serialize execution in debug mode
                process.wait()
            else:
                tqdm.write(f"Scheduled {' '.join(job['args'])} on GPU {gpu}")
                gpu += 1
                processes.append({
                    "process": process,
                    "stdout": stdout,
                    "stderr": stderr,
                })

                if gpu == self.num_gpus or i == len(self.jobs)-1:
                    for process in processes:
                        process["process"].wait()
                        if process["stdout"] != subprocess.DEVNULL:
                            process["stdout"].close()
                        if process["stderr"] != subprocess.DEVNULL:
                            process["stderr"].close()
                        returncodes.append(process["process"].returncode)
                    
                    gpu = 0
                    processes = []
            
        succ = len(list(filter(lambda x: x == 0, returncodes)))
        fail = len(self.jobs) - succ
        print(f"Completed {succ} jobs successfully, {fail} failed")
        return returncodes
    
    def __repr__(self) -> str:
        return f"GPUScheduler[{self.num_gpus},{len(self.jobs)}]"