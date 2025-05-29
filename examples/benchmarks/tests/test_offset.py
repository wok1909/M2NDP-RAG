from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class Offset(NdpKernel):
    def __init__(self):
        super().__init__()
        self.vector_size = 8 * 8 * 3
        self.max_val = 4096
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'test_offset'
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
        template += f'TEST.v.x x2\n'

        return template

    def make_input_map(self):
      return make_memory_map([])

    def make_output_map(self):
      return make_memory_map([])
