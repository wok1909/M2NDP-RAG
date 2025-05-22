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
LEVEL = 0
ITER = 0
visited_list_size = 14
TOPK = 5
EFSEARCH = 300

INFINITY = 9999999999

ENTRY_INIT_ID = -1
DISTANCE_INIT_VALUE = 999999

class SearchGraphKernel0(NdpKernel):
    def __init__(self):
        super().__init__()
        self.INT32_SIZE = 4
        self.INT_MAX = 99999999
        address_list = get_address_list()

        self.qdata_addr = address_list[0]
        self.target_data_addr = address_list[1]
        self.target_nodes_addr = address_list[2]
        self.graph_addr = address_list[3]
        self.degree_addr = address_list[4]
        self.visited_addr = address_list[5]
        self.visited_list_addr = address_list[6]
        self.visited_table_addr = address_list[7]
        self.entries_addr = address_list[8]
        self.acc_visited_cnt_addr = address_list[9]
        self.neighbors_addr = address_list[10]
        self.global_cand_nodes_addr = address_list[11]
        self.global_cand_distances_addr = address_list[12]

        self.graph_info_addr = address_list[13]
        self.update_info_addr = address_list[14]
        self.finish_info_addr = address_list[15]
        self.iter_info_addr = address_list[16]
        self.queue_size_addr = address_list[17]

        self.calculated_distance_addr = address_list[18]
        self.entry_distance_addr = address_list[19]

        # self.partial_sum_addr = address_list[20]
        # self.tmp_entries_addr = address_list[21]

        self.spad_addr    = 0x1000000000000000
        self.base_addr = self.qdata_addr

        self.smem_size = 0x1000000

        self.partial_sum_addr = 0x1000000000100000
        self.local_candidate_addr = 0x1000000000200000

        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'hnsw_SearchGraph0'
        self.graph_file = os.path.join(os.path.dirname(__file__), f'../data/graph.txt')
        self.iter_file = os.path.join(os.path.dirname(__file__), f'../data/SearchGraph_{ITER}.txt')

        # input memorymap
        self.num_query, self.num_data, self.num_dims, self.max_level, self.max_m, self.max_m0, self.enter_point, graphs, queries, data = read_graph_file(self.graph_file)
        entries_array, neighbors_array = read_step2_iter_file(self.iter_file)
        qdata_array = self.preprocess_with_data(queries, True)
        target_data_array = self.preprocess_with_data(data)
        visited_table_array = None
        visited_list_array = None
        # entries_array = None
        acc_visited_cnt_array = None

        target_nodes, graph, degree = self.make_graph(graphs, LEVEL, self.max_m0)

        visited_table_array = [0] *  configs.ndp_units * len(target_nodes)
        visited_list_array = [-1] * visited_list_size * len(target_nodes)
        # entries_array = [self.enter_point] * self.num_query
        acc_visited_cnt_array = [0] * len(target_nodes)
        global_cand_nodes_array = [0] * EFSEARCH * self.num_query
        global_cand_distances_array = [0] * EFSEARCH * self.num_query

        self.qdata = pad8(np.array(qdata_array, dtype=np.int32))
        self.target_data = pad8(np.array(target_data_array, dtype=np.int32))
        # self.target_nodes = pad8(np.array(target_nodes, dtype=np.int32))
        self.graph = pad8(np.array(graph, dtype=np.int32))
        self.degree = pad8(np.array(degree, dtype=np.int32))
        self.visited_table = pad8(np.array(visited_table_array, dtype=np.int32))
        self.visited_table_init = pad8(np.array(visited_table_array, dtype=np.int32))
        self.visited_list = pad8(np.array(visited_list_array, dtype=np.int32))
        self.visited_list_init = pad8(np.array(visited_list_array, dtype=np.int32))
        self.entries = pad8(np.array(entries_array, dtype=np.int32))
        self.acc_visited_cnt = pad8(np.array(acc_visited_cnt_array, dtype=np.int32))
        self.acc_visited_cnt_init = pad8(np.array(acc_visited_cnt_array, dtype=np.int32))
        self.graph_info = pad8(np.array([self.num_query, self.num_data, self.num_dims, self.max_level, self.max_m, self.max_m0, TOPK, EFSEARCH], dtype=np.int32)) # num_query, num_data, num_dims, max_m, topk, ef_search
        self.partial_sum = pad8(np.array([0]*64*self.num_query, dtype=np.int32)) # FIXME: store to spad
        self.distance = pad8(np.array([0] * self.num_query, dtype=np.int32))

        self.tmp_entries = pad8(np.array([ENTRY_INIT_ID] * self.num_query, dtype=np.int32))
        # self.current_entry_distances = pad8(np.array(current_entry_distances * self.num_query, dtype=np.int32))

        self.neighbors = pad8(np.array(neighbors_array, dtype=np.int32))
        self.neighbors_init = pad8(np.array([0] * len(neighbors_array), dtype=np.int32))
        self.global_cand_nodes = pad8(np.array(global_cand_nodes_array, dtype=np.int32))
        self.global_cand_nodes_init = pad8(np.array(global_cand_nodes_array, dtype=np.int32))
        self.global_cand_distances = pad8(np.array(global_cand_distances_array, dtype=np.int32))
        self.global_cand_distances_init = pad8(np.array(global_cand_distances_array, dtype=np.int32))

        self.query_iteration = pad8(np.array([0] * self.num_query, dtype=np.int32))
        self.query_iteration_init = pad8(np.array([0] * self.num_query, dtype=np.int32))
        self.queue_size = pad8(np.array([1] * self.num_query, dtype=np.int32))
        self.queue_size_init = pad8(np.array([0] * self.num_query, dtype=np.int32))
        self.bound = len(qdata_array) * configs.data_size
        self.input_addrs = address_list

        print(f"BOUND = ", self.bound)

    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        elems_per_packet = packet_size / data_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += f'KERNELBODY:\n'
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'li x1, {configs.spad_addr}\n'

        # Kernel0: PushNodeToSearch for entry and GetCand
        # Step1: Push entry node to search pq

        # Get qdata address
        template += f'ld x3, {get_arg_offset(self.qdata_addr)}(x1)]\n' # qdata_addr
        template += f'add x3, x3, x2\n'  # qdata_addr + offset

        # Get node data address
        template += f'ld x4, {get_arg_offset(self.target_data_addr)}(x1)\n'  # target_data_addr

        # Get entry id
        template += f'ld x5, {get_arg_offset(self.entries_addr)}(x1)\n'  # entries_addr
        template += f'muli x6, NDPID, {data_size}\n'
        template += f'add x5, x5, x6\n' # entry_id
      
        # Get queue address
        template += f'ld x6, {get_arg_offset(self.neighbors_addr)}(x1)\n'  # neighbors_addr

        # Get infos for push (ef_search, num_dims)
        template += f'ld x7, {get_arg_offset(self.graph_info_addr)}(x1)\n'  # graph_info_addr
        template += f'lw x8, (x7)\n'  # num_query
        template += f'lw x9, 8(x7)\n'  # num_dims
        template += f'lw x10, 20(x7)\n'  # max_m
        template += f'lw x11, 24(x7)\n'  # topk
        template += f'lw x12, 28(x7)\n'  # ef_search

        template += f'bge NDPID, x8, .SKIP0\n'
        
        # Get node address
        template += f'lw x14, (x5)\n'  # entry_id
        template += f'mul x13, x14, x9\n' # entry_id * num_dims
        template += f'muli x13, x13, {data_size}\n'
        template += f'add x14, x4, x13\n'

        # Add offset in data address
        template += f'li x15, {configs.stride * configs.ndp_units}\n'
        template += f'div x16, x2, x15\n'
        template += f'muli x16, x16, {configs.stride}\n'
        template += f'add x14, x14, x16\n'
        template += f'rem x16, x2, x15\n'
        template += f'add x14, x14, x16\n'
        template += f'muli x16, NDPID, {configs.stride}\n'
        template += f'sub x14, x14, x16\n'

        # Get queue starting address (neighbors)
        template += f'muli x15, x12, {data_size}\n'
        template += f'mul x15, NDPID, x15\n' # ef_search * queryIdx * 4
        template += f'add x6, x6, x15\n'  # neighbors address

        # Calculate distance
        template += f'li x16, 0\n'  # Accumulator
        template += f'.LOOP0\n'
        template += f'vle32.v v1, (x3)\n' # load query
        template += f'vle32.v v2, (x14)\n'  # load entry

        template += f'vsub.vv v3, v1, v2\n'
        template += f'vmul.vv v4, v3, v3\n'

        # Reduce
        template += f'vmv.v.x v0, x0\n'
        template += f'vredsum.vs v5, v4, v0\n'
        template += f'vmv.x.s x13, v5\n'
        template += f'add x16, x16, x13\n'

        # Calculate store address and store partial sum in spad
        template += f'ld x17, {get_arg_offset(self.partial_sum_addr)}(x1)\n'  # partial_sum_addr
        template += f'li x18, {configs.stride * configs.ndp_units}\n'
        template += f'div x19, x2, x18\n'
        template += f'muli x19, x19, {configs.packet_size}\n'
        template += f'add x17, x17, x19\n'

        template += f'rem x19, x2, x18\n'
        template += f'muli x20, NDPID, {configs.stride}\n'
        template += f'sub x19, x19, x20\n'
        template += f'li x20, {packet_size / data_size}\n'
        template += f'div x19, x19, x20\n'
        template += f'add x17, x17, x19\n'
        template += f'sw x16, (x17)\n'
        template += f'.SKIP0\n'

        template += f'KERNELBODY:\n'

        # Reduce distance
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'li x1, {configs.spad_addr}\n'

        # Get qdata address
        template += f'ld x3, {get_arg_offset(self.qdata_addr)}(x1)]\n' # qdata_addr
        template += f'add x3, x3, x2\n'  # qdata_addr + offset

        # Get infos for push (ef_search, num_dims)
        template += f'ld x4, {get_arg_offset(self.graph_info_addr)}(x1)\n'  # graph_info_addr
        template += f'lw x5, (x4)\n'  # num_query
        template += f'lw x6, 8(x4)\n'  # num_dims
        template += f'lw x7, 20(x4)\n'  # max_m
        template += f'lw x8, 24(x4)\n'  # topk
        template += f'lw x9, 28(x4)\n'  # ef_search

        template += f'bge NDPID, x5, .SKIP1\n'  # Skip query out of bound
        
        template += f'muli x10, NDPID, {configs.stride}\n'
        template += f'bne x2, x10, .SKIP1\n'    # Skip micro threads except 0

        # Check already exist
        # If Checked, skip to .SKIP1

        # Calculate store address and store partial sum in spad
        template += f'ld x10, {get_arg_offset(self.partial_sum_addr)}(x1)\n'  # partial_sum_addr
        template += f'li x11, {configs.stride * configs.ndp_units}\n'
        template += f'div x12, x2, x11\n'
        template += f'add x10, x10, x12\n'

        template += f'rem x12, x2, x11\n'
        template += f'muli x20, NDPID, {configs.stride}\n'
        template += f'sub x12, x12, x20\n'
        template += f'li x20, {packet_size / data_size}\n'
        template += f'div x12, x12, x20\n'
        template += f'add x10, x10, x12\n'

        template += f'li x13, 0\n'  # offset
        template += f'li x14, 0\n'  # accumulator
        template += f'.LOOP1\n'
        template += f'add x10, x10, x13\n'
        template += f'vle32.v v1, (x10)\n'
        
        template += f'vmv.v.x v0, x0\n'
        template += f'vredsum.vs v2, v1, v0\n'
        template += f'vmv.x.s x15, v2\n'
        template += f'add x14, x14, x15\n'  # accumulate

        template += f'li x15, {packet_size / (data_size  * data_size)}\n'  # 2
        template += f'div x15, x6, x15\n' # 128 / 2 = 64
        template += f'addi x13, x13, {packet_size}\n'
        template += f'blt x13, x15, .LOOP1\n'

        # Push to queue
        # if size is smaller than ef_search skip to .SKIP2
        template += f'ld x16, {get_arg_offset(self.queue_size_addr)}(x1)\n'  # queue_size_addr
        template += f'muli x17, NDPID, {data_size}\n'
        template += f'add x16, x16, x17\n'
        template += f'ld x18, {get_arg_offset(self.neighbors_addr)}(x1)\n' # neighbors_addr
        template += f'li x17, 0\n' # init size 0
        template += f'bge x17, x9, .SKIP2\n'  # Just push if size bigger than ef_search
        
        # PqPop()
        # Need to implement PqPop()

        template += f'.SKIP2\n'

        # PqPush()
        template += f'addi x19, x17, 0\n' # idx

        template += f'.LOOP2\n'
        template += f'ble x19, x0, .SKIP3\n'  # while (idx > 0)
        template += f'addi x20, x19, 1\n' # nidx = idx+1
        template += f'li 13, 2\n'
        template += f'div x20, x20, x13\n'  # nidx = (idx+1)/2
        template += f'addi x20, x20, -1\n'  # nidx = (idx+1)/2-1

        template += f'muli x21, x20, {3 * data_size}\n'  # nodeid, distance, checked
        template += f'add x21, x18, x21\n'  # nidx node

        # Need to implement loading nidx node and @@@

        template += f'j .LOOP2\n'
        template += f'.SKIP3\n'
        
        template += f'muli x21, x19, {3 * data_size}\n'  # nodeid, distance, checked
        template += f'muli x20, x9, {3 * data_size}\n'  # ef_search * data_size
        template += f'mul x20, x20, NDPID\n'
        template += f'add x21, x18, x21\n'
        template += f'add x21, x21, x20\n'  # address to push
        

        # Get entry id
        template += f'ld x22, {get_arg_offset(self.entries_addr)}(x1)\n'  # entries_addr
        template += f'muli x23, NDPID, {data_size}\n'
        template += f'add x22, x22, x23\n' # entry_id
        template += f'lw x23, (x22)\n'  # entry_id

        template += f'sw x14, (x21)\n' # Store distance
        template += f'sw x23, 4(x21)\n'  # Store nodeid
        template += f'sw x0, 8(x21)\n'  # Store check
        template += f'li x22, 1\n'  # save size 1
        template += f'sw x22, (x16)\n'

        template += f'TEST.v.x x21\n'
        template += f'TEST.v.x x14\n'        
        template += f'TEST.v.x x23\n'

        template += f'.SKIP1\n'

        # CheckVisited implement

        # GetCand
        template += f'KERNELBODY:\n'
        
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'li x1, {configs.spad_addr}\n'

        # Load graph_info, neighbors ...
        # Get infos for push (ef_search, num_dims)
        template += f'ld x3, {get_arg_offset(self.graph_info_addr)}(x1)\n'  # graph_info_addr
        template += f'lw x4, (x3)\n'  # num_query
        template += f'lw x5, 8(x3)\n'  # num_dims
        template += f'lw x6, 20(x3)\n'  # max_m
        template += f'lw x7, 24(x3)\n'  # topk
        template += f'lw x8, 28(x3)\n'  # ef_search

        template += f'bge NDPID, x4, .SKIP4\n'

        # Get queue address
        template += f'ld x9, {get_arg_offset(self.neighbors_addr)}(x1)\n'  # neighbors_addr

        # Get size
        template += f'ld x10, {get_arg_offset(self.queue_size_addr)}(x1)\n'  # queue_size_addr
        template += f'muli x11, NDPID, {data_size}\n'
        template += f'add x12, x10, x11\n'
        template += f'lw x13, (x12)\n'  # load size

        # Calculate micro thread idx
        template += f'li x14, 0\n'  # micro thread idx
        template += f'li x15, {configs.stride * configs.ndp_units}\n'
        template += f'div x16, x2, x15\n'
        template += f'muli x16, x16, {8}\n'
        template += f'add x14, x14, x16\n'

        template += f'rem x16, x2, x15\n'
        template += f'muli x17, NDPID, {configs.stride}\n'
        template += f'sub x16, x16, x17\n'
        template += f'li x17, {packet_size}\n'
        template += f'div x16, x16, x17\n'
        template += f'add x14, x14, x16\n'

        template += f'bge x14, x13, .SKIP4\n' # jump if micro thread idx is bigger than size

        # Find smallest distance in queue
        template += f'li x18, {-INFINITY}\n'
        template += f'li x19, 0\n'  # loop idx (No need? nust add 12 bytes everytime)
        template += f'li x20, {-1}\n' # candidate
        
        template += f'.LOOP3\n'
        template += f'muli x21, x19, {3 * data_size}\n'
        template += f'add x9, x9, x21\n' # load neighbor info of idx
        template += f'lw x22, 8(x9)\n'  # load check

        # template += f'' # compare distance

        template += f'.SKIP6\n' # prepare for jump back to LOOP3
        template += f'li x23, {self.bound}\n'
        template += f'div x24, x23, x15\n'  # bound / 8192
        template += f'rem x25, x23, x15\n'  # bound % 8192
        template += f'muli x26, NDPID, {configs.stride}\n'
        template += f'sub x26, x25, x26\n'  # range 256 byte
        template += f'blt x26, x0, .SKIP5\n'
        template += f'div x27, x26, x17\n'  # range / 32
        template += f'addi x28, x27, 0\n'
        template += f'.SKIP5\n'
        template += f'muli x28, x24, {8}\n' # micro thread launched in the NDP

        template += f'addi x29, x19, 0\n' # copy idx

        template += f'add x19, x19, x28\n'  # add micro thread launched
        template += f'blt x19, x13, .LOOP3\n'

        # Store local candidate
        template += f'ld x30, {get_arg_offset(self.local_candidate_addr)}(x1)\n'
        template += f'muli x31, x14, {data_size}\n'
        template += f'add x30, x30, x31\n'
        template += f'sw x29, (x30)\n'

        template += f'.SKIP4\n'



        # template += f'KERNELBODY:\n'
        # Reduce local candidate and store candidate 

        # HERE I WILL JUST STORE IDX 0 (Too complicated for now)

        # Find global min distance


        
        return template

    def make_input_map(self):
      # if ITER == 0:
        # from ..GetEntryPointsKernel.kernel1 import GetEntryPointsKernel1
        # kernel = GetEntryPointsKernel1(1, 8) # last iter
        # return kernel.make_output_map()
      return make_memory_map([(self.qdata_addr, self.qdata),
                              (self.target_data_addr, self.target_data),
                              (self.graph_addr, self.graph),
                              (self.degree_addr, self.degree),
                              (self.visited_table_addr, self.visited_table_init),
                              (self.visited_list_addr, self.visited_list_init),
                              (self.entries_addr, self.entries),
                              (self.acc_visited_cnt_addr, self.acc_visited_cnt_init),
                              (self.neighbors_addr, self.neighbors_init),
                              (self.global_cand_nodes_addr, self.global_cand_nodes_init),
                              (self.global_cand_distances_addr, self.global_cand_distances_init),
                              (self.graph_info_addr, self.graph_info),
                              (self.iter_info_addr, self.query_iteration_init),
                              (self.queue_size_addr, self.queue_size_init)])

    def make_output_map(self):
      return make_memory_map([(self.qdata_addr, self.qdata),
                              (self.target_data_addr, self.target_data),
                              (self.graph_addr, self.graph),
                              (self.degree_addr, self.degree),
                              (self.visited_table_addr, self.visited_table),
                              (self.visited_list_addr, self.visited_list),
                              (self.entries_addr, self.entries),
                              (self.acc_visited_cnt_addr, self.acc_visited_cnt),
                              (self.neighbors_addr, self.neighbors),
                              (self.global_cand_nodes_addr, self.global_cand_nodes),
                              (self.global_cand_distances_addr, self.global_cand_distances),
                              (self.graph_info_addr, self.graph_info),
                              (self.iter_info_addr, self.query_iteration),
                              (self.queue_size_addr, self.queue_size)])

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



if __name__ == "__main__":

  kernel = GetEntryPointsKernel()
  input_map = kernel.make_input_map()
  output_map = kernel.make_output_map()