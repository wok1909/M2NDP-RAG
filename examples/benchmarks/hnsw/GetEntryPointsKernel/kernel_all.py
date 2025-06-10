import sys
sys.path.append("/root/workspace/M2NDP-public/examples")



from typing import Any
import numpy as np
import math
import os
from utils.utils import NdpKernel, make_memory_map, pad8, make_input_files
import configs
from ..hnsw_utils import *

DEGUG = False
LEVEL = 1
DIST_TYPE = 0
VISITED_LIST_SIZE = 8192

ENTRY_INIT_ID = -1
DISTANCE_INIT_VALUE = 999999

class GetEntryPointsKernel(NdpKernel):
    def __init__(self):
        super().__init__()
        self.INT32_SIZE = 4
        self.INT_MAX = 99999999

        # DRAM
        self.qdata_addr = 0x800000000000
        self.target_data_addr = 0x810000000000
        self.target_nodes_addr = 0x820000000000
        self.graph_addr = 0x830000000000
        self.degree_addr = 0x840000000000
        self.visited_addr = 0x850000000000
        self.visited_list_addr = 0x860000000000
        self.entries_addr = 0x870000000000
        self.acc_visited_cnt_addr = 0x880000000000
        self.graph_info_addr = 0x890000000000

        # Scratchpad
        self.qdata_sapd_addr = 0x1000000000100000
        self.visited_cnt_addr = 0x1000000000200000
        self.entry_distance_addr = 0x1000000000300000
        self.candidate_distance_addr = 0x1000000000400000
        self.partial_sum_addr = 0x1000000000500000

        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'hnsw_GetEntryPointsKernel'
        self.smem_size = 0x1000000

        # graph info
        self.graph_file = os.path.join(os.path.dirname(__file__), f'../data/graph.txt')
        self.num_query, self.num_data, self.num_dims, self.max_level, self.max_m, self.max_m0, self.enter_point, graphs, queries, data = read_graph_file(self.graph_file)
        qdata_array = self.preprocess_with_data(queries, True)
        target_data_array = self.preprocess_with_data(data)
        target_nodes, graph, degree = self.make_graph(graphs, LEVEL, self.max_m)

        self.qdata = pad8(np.array(qdata_array, dtype=np.int32))
        self.target_data = pad8(np.array(target_data_array, dtype=np.int32))
        self.target_nodes = pad8(np.array(target_nodes, dtype=np.int32))
        self.graph = pad8(np.array(graph, dtype=np.int32))
        self.degree = pad8(np.array(degree, dtype=np.int32))
        self.graph_info = pad8(np.array([self.num_query, len(target_nodes), self.num_dims, self.max_level, self.max_m, DIST_TYPE, VISITED_LIST_SIZE], dtype=np.int32))

        # input info
        self.input_file = os.path.join(os.path.dirname(__file__), f'../data/GetEntryPoints_{LEVEL}_start.txt')
        graph_level, entries, visited, visited_list, acc_visited_cnt = read_step1_file(self.input_file)
        assert(graph_level == LEVEL)

        self.input_entries = pad8(np.array(entries, dtype=np.int32))
        self.input_visited = pad8(np.array(visited, dtype=np.int32))
        self.input_visited_list = pad8(np.array(visited_list, dtype=np.int32))
        self.input_acc_visited_cnt = pad8(np.array(acc_visited_cnt, dtype=np.int32))

        # output info
        self.output_file = os.path.join(os.path.dirname(__file__), f'../data/GetEntryPoints_{LEVEL}_finish.txt')
        graph_level, entries, visited, visited_list, acc_visited_cnt = read_step1_file(self.output_file)
        assert(graph_level == LEVEL)

        self.output_entries = pad8(np.array(entries, dtype=np.int32))
        self.output_visited = pad8(np.array(visited, dtype=np.int32))
        self.output_visited_list = pad8(np.array(visited_list, dtype=np.int32))
        self.output_acc_visited_cnt = pad8(np.array(acc_visited_cnt, dtype=np.int32))

        self.bound = len(qdata_array) * configs.data_size
        self.input_addrs = [self.qdata_addr, self.target_data_addr, self.target_nodes_addr, self.graph_addr, self.degree_addr,
                            self.visited_addr, self.visited_list_addr, self.entries_addr, self.acc_visited_cnt_addr, self.graph_info_addr,
                            self.qdata_sapd_addr, self.entry_distance_addr, self.partial_sum_addr, self.candidate_distance_addr, self.visited_cnt_addr]

    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        elems_per_packet = packet_size / data_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        ################ SYNC0 #################
        template += f'KERNELBODY:\n'  # KERNELBODY0
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'li x1, {configs.spad_addr}\n'

        # Load num_query from Graph info and skip if necessary (NDP branch)
        template += f'ld x3, {self.get_arg_offset(self.graph_info_addr)}(x1)\n'
        template += f'lw x4, (x3)\n'  # Load num_query
        template += f'bge NDPID, x4, .SKIP0\n'

        # Load num_target_nodes, num_dims, max_m, visited_list_size
        template += f'lw x5, 4(x3)\n' # num_target_nodes
        template += f'lw x6, 8(x3)\n' # num_dims
        template += f'lw x7, 16(x3)\n' # max_m
        template += f'lw x8, 24(x3)\n' # visited_list_size

        # Load visited variable address
        template += f'ld x9, {self.get_arg_offset(self.visited_addr)}(x1)\n'
        template += f'ld x10, {self.get_arg_offset(self.visited_list_addr)}(x1)\n'

        # Query loop when num_query bigger than 32
        template += f'li x12, 0\n'
        template += f'.LOOP0\n'

        # Skip if query index is out of range
        template += f'muli x13, x12, {configs.ndp_units}\n'
        template += f'add x13, x13, NDPID\n'
        template += f'bge x13, x4, .SKIP0\n'

        # Store query to spad
        template += f'ld x14, {self.get_arg_offset(self.qdata_addr)}(x1)\n'
        template += f'ld x15, {self.get_arg_offset(self.qdata_sapd_addr)}(x1)\n'
        template += f'add x14, x14, x2\n' # query data address for current uthread
        template += f'vle32.v v1, (x14)\n'
        template += f'muli x16, UTHREADID, {packet_size}\n'
        template += f'add x15, x15, x16\n'  # qdata address
        template += f'vse32.v v1, (x15)\n'  # Store qdata to spad

        # Set visited variables
        template += f'mul x14, x5, x13\n'
        template += f'muli x14, x14, {data_size}\n'
        template += f'add x9, x9, x14\n'  # visited

        template += f'mul x14, x8, x13\n'
        template += f'muli x14, x14, {data_size}\n'
        template += f'add x10, x10, x14\n'  # visited_list

        # Load entryid
        template += f'ld x14, {self.get_arg_offset(self.entries_addr)}(x1)\n'
        template += f'muli x16, x13, {data_size}\n'
        template += f'add x16, x14, x16\n'
        template += f'lw x3, (x16)\n'  # entryid

        # Initialize visited_cnt to 0
        template += f'bnez UTHREADID, .SKIP1\n'
        template += f'li x11, 0\n'
        template += f'.SKIP1\n'

        # Get entry data address
        template += f'ld x16, {self.get_arg_offset(self.target_nodes_addr)}(x1)\n'
        template += f'ld x17, {self.get_arg_offset(self.target_data_addr)}(x1)\n'
        template += f'muli x18, x3, {data_size}\n'
        template += f'add x18, x16, x18\n'
        template += f'lw x14, (x18)\n'  # targete_nodes[entryid]
        template += f'mul x19, x6, x14\n'
        template += f'muli x19, x19, {data_size}\n'
        template += f'add x18, x17, x19\n'  # dest_vec address
        template += f'muli x19, UTHREADID, {packet_size}\n'
        template += f'add x18, x18, x19\n'

        # Local distance calculation
        template += f'ld x19, {self.get_arg_offset(self.partial_sum_addr)}(x1)\n'
        template += f'vle32.v v2, (x18)\n'
        template += f'vsub.vv v3, v2, v1\n'
        template += f'vmul.vv v4, v3, v3\n'
        template += f'vmv.v.x v0, x0\n'
        template += f'vredsum.vs v5, v4, v0\n'
        template += f'vmv.x.s x20, v5\n'  # partial distance output
        template += f'muli x21, UTHREADID, {data_size}\n'
        template += f'add x19, x19, x21\n'  # partial sum address
        template += f'sw x20, (x19)\n'  # store partial distance output

        ################ SYNC1 #################
        template += f'KERNELBODY:\n'  # KERNELBODY1
        template += f'bgt UTHREADID, x0, .SKIP2\n'

        # Global distance calculation
        template += f'li x20, 0\n'  # accumulator
        template += f'addi x21, UTHREADSZ, 0\n'  # counter
        template += f'vid.v v2\n' # v2 = [0, 1, 2, 3, 4, 5, 6, 7]

        template += f'addi x18, x19, 0\n'
        template += f'.LOOP1\n'
        template += f'vle32.v v3, (x18)\n'
        template += f'vmslt.vx v4, v2, x21\n'
        template += f'vredsum.vs v5, v3, v0, v4\n'
        template += f'vmv.x.s x22, v5\n'
        template += f'add x20, x20, x22\n'
        template += f'addi x21, x21, -{packet_size / data_size}\n'
        template += f'addi x18, x18, {packet_size}\n'
        template += f'bgt x21, x0, .LOOP1\n'

        # Store entry distance to scratchpad
        template += f'ld x21, {self.get_arg_offset(self.entry_distance_addr)}(x1)\n'
        template += f'sw x20, (x21)\n'

        ################ SYNC2 #################
        template += f'KERNELBODY:\n'  # KERNELBODY2
        template += f'.SKIP2\n'

        # Get entry distance address
        template += f'ld x18, {self.get_arg_offset(self.entry_distance_addr)}(x1)\n'
        template += f'lw x20, (x18)\n'  # entry distance

        # Get degree and graph address
        template += f'ld x21, {self.get_arg_offset(self.degree_addr)}(x1)\n'
        template += f'ld x22, {self.get_arg_offset(self.graph_addr)}(x1)\n'

        # Loop for finding entry point
        template += f'.LOOP2\n'
        template += f'muli x23, x3, {data_size}\n'
        template += f'add x23, x21, x23\n'
        template += f'lw x24, (x23)\n'  # degree of entry
        template += f'mul x23, x7, x3\n' # starting idx = max_m * entry_id
        template += f'add x24, x23, x24\n'  # end idx = starting idx + degree

        # Initialize updated to true
        template += f'li x25, 0\n'

        # neighbor search
        template += f'.LOOP3\n'
        template += f'bge x23, x24, .SKIP6\n'
        template += f'muli x26, x23, {data_size}\n' # starting index
        template += f'add x26, x22, x26\n'
        template += f'lw x27, (x26)\n'  # candid

        template += f'muli x26, x27, {data_size}\n'
        template += f'add x26, x9, x26\n'
        template += f'lw x28, (x26)\n'  # visited
        template += f'addi x23, x23, 1\n' # increase idx
        template += f'bnez x28, .LOOP3\n'  # branch if visited

        ################ SYNC3 #################
        template += 'KERNELBODY:\n' # KERNELBODY3

        # set visited and visited_list
        template += f'bnez UTHREADID, .SKIP3\n'
        template += f'bge x11, x8, .SKIP3\n'  # branch if visited_cnt >= visited_list_size
        template += f'li x28, 1\n'
        template += f'sw x28, (x26)\n'  # store visited
        template += f'muli x29, x11, {data_size}\n'
        template += f'add x29, x10, x29\n'
        template += f'sw x27, (x29)\n'  # visited_list[visited_cnt] = cnadid
        template += f'addi x11, x11, 1\n' # increase visited_cnt
        template += f'.SKIP3\n'

        # Get candidate node address
        template += f'muli x28, x27, {data_size}\n'
        template += f'add x28, x16, x28\n'
        template += f'lw x29, (x28)\n'  # target_nodes[cacndid]
        template += f'mul x30, x6, x29\n'
        template += f'muli x30, x30, {data_size}\n'
        template += f'add x30, x17, x30\n'  # dest_vec address

        # Distance calculation with query
        template += f'muli x31, UTHREADID, {packet_size}\n'
        template += f'add x30, x30, x31\n'
        template += f'vle32.v v2, (x30)\n'
        template += f'vsub.vv v3, v2, v1\n'
        template += f'vmul.vv v4, v3, v3\n'
        template += f'vmv.v.x v0, x0\n'
        template += f'vredsum.vs v5, v4, v0\n'
        template += f'vmv.x.s x18, v5\n'  # partial distance output
        template += f'sw x18, (x19)\n'  # store partial distance output

        ################ SYNC4 #################
        template += 'KERNELBODY:\n' # KERNELBODY4
        template += f'bgt UTHREADID, x0, .SKIP4\n'

        # Global distance calculation
        template += f'li x5, 0\n'  # accumulator (dist)
        template += f'addi x26, UTHREADSZ, 0\n' # counter
        template += f'vid.v v2\n' # v2 = [0, 1, 2, 3, 4, 5, 6, 7]

        template += f'addi x15, x19, 0\n'
        template += f'.LOOP4\n'
        template += f'vle32.v v3, (x15)\n'
        template += f'vmslt.vx v4, v2, x26\n'
        template += f'vredsum.vs v5, v3, v0, v4\n'
        template += f'vmv.x.s x28, v5\n'
        template += f'add x5, x5, x28\n'
        template += f'addi x26, x26, -{packet_size / data_size}\n'
        template += f'addi x15, x15, {packet_size}\n'
        template += f'bgt x26, x0, .LOOP4\n'

        # Store global distance of candidate
        template += f'ld x26, {self.get_arg_offset(self.candidate_distance_addr)}(x1)\n'
        template += f'sw x5, (x26)\n'

        template += f'.SKIP4\n'
        ################ SYNC5 #################
        template += 'KERNELBODY:\n' # KERNELBODY5

        template += f'ld x26, {self.get_arg_offset(self.candidate_distance_addr)}(x1)\n'
        template += f'lw x5, (x26)\n'

        # Update entry_dist, entryid and updated if dist < entry_dist
        template += f'bge x5, x20, .SKIP5\n'
        template += f'addi x20, x5, 0\n'  # entry_dist = dist
        template += f'addi x3, x27, 0\n' # entryid = candid
        template += f'li x25, 1\n'  # updated =  ture
        template += f'.SKIP5\n'

        ################ SYNC6 #################
        template += 'KERNELBODY:\n' # KERNELBODY6
        # branch to .LOOP3 if entry has more neighbors
        template += f'blt x23, x24, .LOOP3\n'

        template += f'.SKIP6\n'
        # Store entry
        template += f'bgt UTHREADID, x0, .SKIP7\n'
        template += f'ld x23, {self.get_arg_offset(self.entries_addr)}(x1)\n'
        template += f'muli x27, x13, {data_size}\n'
        template += f'add x27, x23, x27\n'
        template += f'sw x3, (x27)\n'
        template += f'.SKIP7\n'

        # Branch to .LOOP2 if updated
        template += f'bnez x25, .LOOP2\n'

        # Store visited_cnt to spad and acc_visited_cnt update
        template += f'ld x21, {self.get_arg_offset(self.visited_cnt_addr)}(x1)\n'
        template += f'bgt UTHREADID, x0, .SKIP8\n'
        template += f'ld x25, {self.get_arg_offset(self.acc_visited_cnt_addr)}(x1)\n'
        template += f'muli x26, x13, {data_size}\n'
        template += f'add x25, x25, x26\n'
        template += f'lw x26, (x25)\n'
        template += f'add x27, x26, x11\n'
        template += f'sw x11, (x21)\n'
        template += f'sw x27, (x25)\n'
        template += f'.SKIP8\n'

        ################ SYNC7 #################
        template += 'KERNELBODY:\n' # KERNELBODY7

        # visited update
        template += f'ld x21, {self.get_arg_offset(self.visited_cnt_addr)}(x1)\n'
        template += f'lw x11, (x21)\n'  # load visited_cnt
        template += f'addi x28, UTHREADID, 0\n'
        template += f'.LOOP5\n'
        template += f'bge x28, x11, .SKIP9\n'
        template += f'muli x29, x28, {data_size}\n'
        template += f'add x29, x10, x29\n'
        template += f'lw x30, (x29)\n' # visited_list[UTHREADID]
        template += f'muli x31, x30, {data_size}\n'
        template += f'add x31, x9, x31\n'
        template += f'sw x0, (x31)\n'
        template += f'add x28, x28, UTHREADSZ\n'
        template += f'blt x28, x11, .LOOP5\n'
        template += f'.SKIP9\n'

        # Increment query idx and jump back
        template += f'addi x12, x12, 1\n'
        template += f'j .LOOP0\n'

        ################ SYNC8 #################
        template += 'KERNELBODY:\n' # KERNELBODY8
        template += f'.SKIP0\n'

        return template

    def make_input_map(self):
      return make_memory_map([(self.qdata_addr, self.qdata),
                              (self.target_data_addr, self.target_data),
                              (self.target_nodes_addr, self.target_nodes),
                              (self.graph_addr, self.graph),
                              (self.degree_addr, self.degree),
                              (self.visited_addr, self.input_visited),
                              (self.visited_list_addr, self.input_visited_list),
                              (self.entries_addr, self.input_entries),
                              (self.acc_visited_cnt_addr, self.input_acc_visited_cnt),
                              (self.graph_info_addr, self.graph_info)])

    def make_output_map(self):
      return make_memory_map([(self.qdata_addr, self.qdata),
                              (self.target_data_addr, self.target_data),
                              (self.target_nodes_addr, self.target_nodes),
                              (self.graph_addr, self.graph),
                              (self.degree_addr, self.degree),
                              (self.visited_addr, self.output_visited),
                              (self.visited_list_addr, self.output_visited_list),
                              (self.entries_addr, self.output_entries),
                              (self.acc_visited_cnt_addr, self.output_acc_visited_cnt),
                              (self.graph_info_addr, self.graph_info)])

    def preprocess_with_data(self, data, is_query=False):
      stride = configs.stride
      # stride = configs.packet_size
      elem_stride = int(stride / 4)
      n_data = len(data)
      if is_query and n_data < configs.ndp_units:
        n_data = configs.ndp_units
      dim = len(data[0])
      pad_dim = (dim + elem_stride - 1) // elem_stride * elem_stride
      data_size = n_data * pad_dim
      processed_data = [0] * data_size

      if is_query:
        stride_stride = elem_stride * n_data
        for query_idx in range(configs.ndp_units):
          for o_loop in range(int(pad_dim / elem_stride)):
            for i_loop in range(elem_stride):
              # print("idx: ", o_loop * elem_stride + i_loop)
              if o_loop * elem_stride + i_loop < dim and query_idx < len(data):
                val = data[query_idx][o_loop * elem_stride + i_loop]

                # print(val)
                processed_data[o_loop * stride_stride + query_idx * elem_stride + i_loop] = val
      else:
        for idx in range(n_data):
          for dim_idx in range(pad_dim):
            processed_data[idx * pad_dim + dim_idx] = data[idx][dim_idx]

      return processed_data

    def make_graph(self, graphs, level, max_m):
      if DEGUG:
        print(f"Make graph level {level}")

      graph = graphs[level]
      upper_size = len(graph)
      upper_node = []
      neighbors = [0] * (upper_size * max_m)
      degree = [0] * upper_size

      if DEGUG:
        print("Graph len: ", len(graph))
        print("neighbors len: ", len(neighbors))
        # print("Graph 0: ", graph)

      for idx, (src, dst_info) in enumerate(graph.items()):
        upper_node.append(src)
        degree[idx] = len(dst_info)

      for idx, (src, dst_info) in enumerate(graph.items()):
        offset = idx * max_m
        # print(idx, src, max_m, offset, len(dst_info))
        for n_idx in range(len(dst_info)):
          try:
            neighbors[offset + n_idx] = upper_node.index(dst_info[n_idx][0])
          except ValueError:
            print("Node id not found")


        # if DEGUG:
        #   print(neighbors)
        #   print(degree)
      return upper_node, neighbors, degree

    def get_arg_offset(self, addr):
      try:
          return self.input_addrs.index(addr) * 8
      except ValueError:
          raise ValueError(f"Address {addr} not found in input_addrs list.")


if __name__ == "__main__":

  kernel = GetEntryPointsKernel()
  input_map = kernel.make_input_map()
  output_map = kernel.make_output_map()
  make_input_files("test_kernel_name", "test_code", input_map, input_map, [], "/root/workspace/M2NDP-public/examples/benchmarks/hnsw/GetEntryPointsKernel/test_input.txt")
  # print(kernel.make_kernel())
