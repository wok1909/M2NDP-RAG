from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class Sync0(NdpKernel):
    def __init__(self):
        super().__init__()
        self.vector_size = 8 * 8 * 3
        self.reg_context_addr = 0x1000000000001000
        self.count_addr = 0x800000000000
        self.context_sz = 128
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'test_sync0'
        self.bound = self.vector_size * configs.data_size
        self.smem_size = 0x10000 - 0x10

        self.input = (np.array([0, 0, 0], dtype=np.int32))
        self.output = (np.array([1, 2, 3], dtype=np.int32))

        self.input_addrs = [self.reg_context_addr, self.count_addr]

    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += f'KERNELBODY:\n'  # KERNELBODY 0
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'addi x3, UTHREADID, 0\n'
        template += f'li x4, 0\n' # outer loop count

        # Store reg context
        template += f'li x1, {configs.spad_addr}\n'
        template += f'ld x31, (x1)\n'
        template += f'muli x30, UTHREADID, {self.context_sz}\n'
        template += f'add x31, x31, x30\n'
        template += f'sw x3, (x31)\n'
        template += f'sw x4, 4(x31)\n'

        template += f'.LOOP0\n'

        # Load reg context
        template += f'li x1, {configs.spad_addr}\n'
        template += f'ld x31, (x1)\n'
        template += f'muli x30, UTHREADID, {self.context_sz}\n'
        template += f'add x31, x31, x30\n'
        template += f'lw x3, (x31)\n'
        template += f'lw x4, 4(x31)\n'
        
        template += f'li x5, 0\n' # inner loop count

        # Store reg context
        template += f'sw x3, (x31)\n'
        template += f'sw x4, 4(x31)\n'
        template += f'sw x5, 8(x31)\n'

        template += f'KERNELBODY:\n'  # KERNELBODY 1: Inner loop
        # Load reg context
        template += f'li x1, {configs.spad_addr}\n'
        template += f'ld x31, (x1)\n'
        template += f'muli x30, UTHREADID, {self.context_sz}\n'
        template += f'add x31, x31, x30\n'
        template += f'lw x3, (x31)\n'
        template += f'lw x4, 4(x31)\n'
        template += f'lw x5, 8(x31)\n'

        template += f'.LOOP1\n'
        template += f'addi x5, x5, 1\n'
        template += f'TEST.v.x x5\n'
        template += f'ble x5, x3, .LOOP1\n'

        # Store reg context
        template += f'sw x3, (x31)\n'
        template += f'sw x4, 4(x31)\n'
        template += f'sw x5, 8(x31)\n'

        template += f'KERNELBODY:\n'  # KERNELBODY 2
        # Load reg context
        template += f'li x1, {configs.spad_addr}\n'
        template += f'ld x31, (x1)\n'
        template += f'muli x30, UTHREADID, {self.context_sz}\n'
        template += f'add x31, x31, x30\n'
        template += f'lw x3, (x31)\n'
        template += f'lw x4, 4(x31)\n'
        template += f'lw x5, 8(x31)\n'

        template += f'addi x4, x4, 1\n'
        template += f'muli x10, x4, 10\n'
        template += f'TEST.v.x x10\n'

        # Store reg context
        template += f'sw x3, (x31)\n'
        template += f'sw x4, 4(x31)\n'
        template += f'sw x5, 8(x31)\n'
        
        template += f'ble x4, NDPID, .LOOP0\n'

        # Store count
        template += f'ld x6, 8(x1)\n'
        template += f'muli x7, NDPID, {data_size}\n'
        template += f'add x6, x6, x7\n'
        template += f'sw x4, (x6)\n'
        return template

    def make_input_map(self):
      return make_memory_map([(self.count_addr, self.input)])

    def make_output_map(self):
      return make_memory_map([(self.count_addr, self.output)])