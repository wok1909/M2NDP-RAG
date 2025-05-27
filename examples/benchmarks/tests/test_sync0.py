from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class Sync0(NdpKernel):
    def __init__(self):
        super().__init__()
        self.vector_size = 8 * 8 * 3
        self.max_val = 4096
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'test_sync0'
        self.bound = self.vector_size * configs.data_size
        self.input_addrs = [0]

    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += f'KERNELBODY:\n'
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'addi x3, UTHREADID, 0\n'
        template += f'li x4, 0\n' # outer loop count
        template += f'.LOOP0\n'
        template += f'li x5, 0\n' # inner loop count

        template += f'KERNELBODY:\n'
        template += f'.LOOP1\n'
        template += f'addi x5, x5, 1\n'
        template += f'TEST.v.x x5\n'
        template += f'ble x5, x3, .LOOP1\n'
        template += f'TEST.v.x x0\n'

        template += f'KERNELBODY:\n'
        template += f'addi x4, x4, 1\n'
        template += f'TEST.v.x x4\n'
        template += f'TEST.v.x NDPID\n'
        template += f'ble x4, NDPID, .LOOP0\n'

        return template

    def make_input_map(self):
      return make_memory_map([])

    def make_output_map(self):
      return make_memory_map([])
