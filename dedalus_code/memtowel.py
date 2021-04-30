"""Clean up those memory leaks."""

import os
import psutil
from mpi4py import MPI
import numpy as np


class MemoryTowel:
    """Measure and save memory used by all processes."""

    def __init__(self, comm=MPI.COMM_WORLD, filename="memory.txt"):
        self.comm = comm
        self.comm_rank = comm.rank
        self.comm_size = comm.size
        self.filename = filename
        self.process = psutil.Process(os.getpid())
        self.sendbuf = np.zeros(1, dtype=int)
        if self.comm_rank == 0:
            self.recvbuf = np.zeros([self.comm_size, 1], dtype=int)
        else:
            self.recvbuf = None

    def process_memory(self):
        """Measure memory usage by current process."""
        return self.process.memory_info()[0]

    def comm_memory(self):
        """Gather memory usage by each process."""
        self.sendbuf[0] = self.process_memory()
        self.comm.Gather(self.sendbuf, self.recvbuf, root=0)
        return self.recvbuf

    def print_comm_memory(self):
        """Print comm memory."""
        comm_memory = self.comm_memory()
        if self.comm_rank == 0:
            print(comm_memory.ravel())

    def write_comm_memory(self, reset=False):
        """Save comm memory."""
        comm_memory = self.comm_memory()
        if self.comm_rank == 0:
            if reset:
                mode = "wb"
            else:
                mode = "ab"
            with open(self.filename, mode) as file:
                np.savetxt(file, comm_memory.T, fmt='%i')


if __name__ == '__main__':
    # Test the towel
    memtowel = MemoryTowel()
    a = np.random.random((512,512,512))
    memtowel.print_comm_memory()
    memtowel.write_comm_memory(reset=True)
    b = np.random.random((512,512,512))
    memtowel.print_comm_memory()
    memtowel.write_comm_memory()
    del b
    memtowel.print_comm_memory()
    memtowel.write_comm_memory()

