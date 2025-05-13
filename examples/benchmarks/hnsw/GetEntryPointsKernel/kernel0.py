import sys
sys.path.append("/root/workspace/M2NDP-public/examples")



from typing import Any
import numpy as np
import math
import os
from utils.utils import NdpKernel, make_memory_map, pad8, make_input_files
import configs

DEGUG = False
ITER = 0
LEVEL = 3
visited_list_size = 14

ENTRY_INIT_ID = -1
DISTANCE_INIT_VALUE = 999999

class GetEntryPointsKernel0(NdpKernel):
    def __init__(self):
        super().__init__()
        self.INT32_SIZE = 4
        self.INT_MAX = 99999999
        self.qdata_addr    = 0x800000000000
        # self.qnodes_addr = 0x810000000000
        self.target_data_addr = 0x820000000000
        self.target_nodes_addr = 0x830000000000
        self.graph_addr = 0x840000000000
        self.degree_addr = 0x850000000000
        self.visited_addr = 0x860000000000
        self.visited_list_addr = 0x870000000000
        self.entries_addr = 0x880000000000
        self.acc_visited_cnt_addr = 0x890000000000
        self.spad_addr    = 0x1000000000000000
        self.base_addr = self.qdata_addr

        # Additional info
        self.graph_info_addr = 0x8a0000000000
        self.iteration_info_addr = 0x8b0000000000
        self.partial_sum_addr = 0x8c0000000000
        self.calculated_distance_addr = 0x8d0000000000

        self.tmp_entries_addr = 0x8e0000000000
        self.current_entry_distance_addr = 0x8f0000000000
        self.query_iteration_addr = 0x810000000000

        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'hnsw_GetEntryPointsKernel'
        self.graph_file = os.path.join(os.path.dirname(__file__), f'../data/graph.txt')
        self.iter_file = os.path.join(os.path.dirname(__file__), f'../data/{LEVEL}_{ITER}.txt')

        # input memorymap
        self.num_query, self.num_data, self.num_dims, self.max_level, self.max_m, self.max_m0, self.enter_point, graphs, queries, data = self.read_file()
        entry_id, future_entry_ids, candidate_ids, partial_sums_result, calculated_distances, current_entry_distances = self.read_iter_file(self.max_level)
        qdata_array = self.preprocess_with_data(queries, True)
        target_data_array = self.preprocess_with_data(data)
        visited_array = None
        visiteid_list_array = None
        entries_array = None
        acc_visited_cnt_array = None

        target_nodes, graph, degree = self.make_graph(graphs, LEVEL, self.max_m)

        visited_array = [0] *  configs.ndp_units * len(target_nodes)
        visited_list_array = [-1] * visited_list_size * len(target_nodes)
        entries_array = [self.enter_point] * self.num_query
        acc_visited_cnt_array = [0] * len(target_nodes)

        self.qdata = pad8(np.array(qdata_array, dtype=np.int32))
        self.target_data = pad8(np.array(target_data_array, dtype=np.int32))
        self.target_nodes = pad8(np.array(target_nodes, dtype=np.int32))
        self.graph = pad8(np.array(graph, dtype=np.int32))
        self.degree = pad8(np.array(degree, dtype=np.int32))
        self.visited = pad8(np.array(visited_array, dtype=np.int32))
        self.visited_list = pad8(np.array(visited_list_array, dtype=np.int32))
        self.entries = pad8(np.array(entries_array, dtype=np.int32))
        self.acc_visited_cnt = pad8(np.array(acc_visited_cnt_array, dtype=np.int32))
        self.graph_info = pad8(np.array([self.num_query, self.num_data, self.num_dims, self.max_level, self.max_m, self.max_m0], dtype=np.int32)) # num_query, num_data, num_dims, max_level, max_m, max_m0, ITER
        self.partial_sum = pad8(np.array([0]*64*self.num_query, dtype=np.int32)) # FIXME: store to spad
        self.distance = pad8(np.array([0] * self.num_query, dtype=np.int32))

        self.partial_sums_result = pad8(np.array(partial_sums_result, dtype=np.int32))
        self.calculated_distances = pad8(np.array(calculated_distances, dtype=np.int32))
        
        self.tmp_entries = pad8(np.array([ENTRY_INIT_ID] * self.num_query, dtype=np.int32))
        self.current_entry_distances = pad8(np.array(current_entry_distances * self.num_query, dtype=np.int32))

        self.iteration_info = pad8(np.array([ITER, 0, 0], dtype=np.int32))  # Current iter, update, finish
        self.query_iteration = pad8(np.array([0] * self.num_query, dtype=np.int32))
        self.bound = len(qdata_array) * configs.data_size
        self.input_addrs = [self.qdata_addr, self.target_data_addr, self.target_nodes_addr, self.graph_addr, self.degree_addr,
                            self.visited_addr, self.visited_list_addr, self.entries_addr, self.acc_visited_cnt_addr,
                            self.graph_info_addr, self.iteration_info_addr, self.partial_sum_addr, self.calculated_distance_addr,
                            self.tmp_entries_addr, self.current_entry_distance_addr, self.query_iteration_addr]

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

        # Query data address
        template += f'ld x3, (x1)\n' # qdata_addr
        template += f'add x4, x3, x2\n' # qdata_addr + offset (256 Byte, 64 elements)

        # Get num query
        template += f'ld x26, 72(x1)\n' # graph_info_addr
        template += f'lw x27, (x26)\n'
        template += f'bge NDPID, x27, .SKIP0\n'

        # Get entry id
        template += f'ld x5, 56(x1)\n' # entries_addr
        template += f'muli x6, NDPID, {data_size}\n'
        template += f'add x7, x5, x6\n' # entry address
        template += f'lw x8, (x7)\n'  # load entry id

        # Get entry id
        template += f'ld x9, 16(x1)\n' # target_nodes_addr
        template += f'muli x10, x8, {data_size}\n' # entry node offset
        template += f'add x11, x9, x10\n' # entry id
        template += f'lw x11, (x11)\n'

        # Get entry data address
        template += f'ld x27, 8(x1)\n' # target_data_addr
        template += f'muli x18, x11, {data_size}\n'
        template += f'muli x18, x18, {self.num_dims}\n'
        template += f'add x28, x27, x18\n'
        # Add offset in data address
        template += f'li x30, {configs.stride * configs.ndp_units}\n' # 8192
        template += f'div x31, x2, x30\n'
        template += f'muli x31, x31, {configs.stride}\n'
        template += f'add x28, x28, x31\n'
        template += f'rem x31, x2, x30\n'
        template += f'add x28, x28, x31\n'
        template += f'muli x31, NDPID, {configs.stride}\n'
        template += f'sub x28, x28, x31\n'

        # Prepare loop: 64 elements (256 Byte)
        template += f'li x12, 0\n'  # address offset
        template += f'li x30, 0\n'  # accumulator

        template += f'.LOOP0\n' # micro thread reduce loop
        # Get query data
        template += f'add x14, x4, x12\n' # qdata address
        template += f'vle32.v v1, (x14)\n'

        # Get entry node data
        template += f'add x15, x28, x12\n'  # entry data address
        template += f'vle32.v v2, (x15)\n'

        # Vector compute (l2: sum((l2-l1)^2))
        template += f'vsub.vv v3, v2, v1\n' # l2 - l1
        template += f'vmul.vv v4, v3, v3\n' # square

        # Reduce
        template += f'vmv.v.x v0, x0\n'
        template += f'vredsum.vs v5, v4, v0\n'  # vector reduce
        template += f'vmv.x.s x15, v5\n'
        template += f'add x30, x30, x15\n'  # accumulate

        # Add 32(packet size) to offset and loop if small than stride
        template += f'addi x12, x12, {packet_size}\n'
        template += f'li x13, {packet_size}\n'
        template += f'blt x12, x13, .LOOP0\n'

        # Calculate store address and store partial sum
        template += f'ld x16, 88(x1)\n' # partial_sum_addr
        template += f'li x19, {configs.stride * configs.ndp_units}\n'
        template += f'div x17, x2, x19\n' # 몇번째 stride 에 있는지
        template += f'muli x17, x17, {packet_size}\n'
        template += f'add x16, x16, x17\n'

        template += f'rem x17, x2, x19\n' # stride 안에서 의 offset
        template += f'muli x18, NDPID, {configs.stride}\n'
        template += f'sub x17, x17, x18\n'
        template += f'li x18, {packet_size / data_size}\n'
        template += f'div x17, x17, x18\n'
        template += f'add x16, x16, x17\n'

        template += f'muli x31, NDPID, {64 * data_size}\n' # Because currently storing to DRAM
        template += f'add x19, x16, x31\n'
        template += f'sw x30, (x19)\n'

        template += f'.SKIP0\n'

# ------------------ SYNC ------------------
        template += f'KERNELBODY:\n'  # Sync
        template += f'li x1, {configs.spad_addr}\n'
        template += f'ld x13, 72(x1)\n' # graph_info_addr
        template += f'lw x13, (x13)\n'  # num query

        template += f'bge NDPID, x13, .SKIP1\n'

        template += f'ld x16, 88(x1)\n' # partial_sum_addr

        template += f'li x17, {64 * data_size}\n'
        template += f'mul x18, NDPID, x17\n'
        template += f'add x16, x16, x18\n'

        template += f'muli x19, NDPID, {configs.stride}\n'
        template += f'bne x2, x19, .SKIP1\n'  # skip if not micro thread 0

        # Reduce among micro threads
        template += f'li x20, 0\n'  # reduce address offset
        template += f'li x21, 0\n'  # accumulator
        template += f'.LOOP1\n' # global reduce loop
        # Load partial sums
        template += f'add x22, x16, x20\n'  # load address
        template += f'vle32.v v6, (x22)\n'

        # Reduce
        template += f'vmv.v.x v0, x0\n'
        template += f'vredsum.vs v7, v6, v0\n' # vector reduce
        template += f'vmv.x.s x23, v7\n'
        template += f'add x21, x21, x23\n'  # accumulate

        # Add 32(packet size) to offset and loop if small than 64 (upper bound)
        template += f'addi x20, x20, {packet_size}\n'
        template += f'li x24, {data_size * 64}\n' # max upper bound 64 micro thread
        template += f'blt x20, x24, .LOOP1\n'

        # Store global reduce sum
        template += f'ld x25, 96(x1)\n' # calculated_distance_addr
        template += f'muli x26, NDPID, {data_size}\n'
        template += f'add x26, x25, x26\n'
        template += f'sw x21, (x26)\n'  # store distance

        template += f'ld x25, 112(x1)\n' # current_entry_distance_addr
        template += f'muli x26, NDPID, {data_size}\n'
        template += f'add x26, x25, x26\n'
        template += f'sw x21, (x26)\n'  # store distance

        template += f'.SKIP1\n'


        return template

    def make_input_map(self):
      return make_memory_map([(self.qdata_addr, self.qdata),
                              # (self.qnodes_addr, self.qnodes),
                              (self.target_data_addr, self.target_data),
                              (self.target_nodes_addr, self.target_nodes),
                              (self.graph_addr, self.graph),
                              (self.degree_addr, self.degree),
                              (self.visited_addr, self.visited),
                              (self.visited_list_addr, self.visited_list),
                              (self.entries_addr, self.entries),
                              (self.acc_visited_cnt_addr, self.acc_visited_cnt),
                              (self.graph_info_addr, self.graph_info),
                              (self.iteration_info_addr, self.iteration_info),
                              (self.partial_sum_addr, self.partial_sum),
                              (self.calculated_distance_addr, self.distance),
                              (self.tmp_entries_addr, self.entries),
                              (self.current_entry_distance_addr, self.distance),
                              (self.query_iteration_addr, self.query_iteration)])

    def make_output_map(self):
      return make_memory_map([(self.qdata_addr, self.qdata),
                              # (self.qnodes_addr, self.qnodes),
                              (self.target_data_addr, self.target_data),
                              (self.target_nodes_addr, self.target_nodes),
                              (self.graph_addr, self.graph),
                              (self.degree_addr, self.degree),
                              (self.visited_addr, self.visited),
                              (self.visited_list_addr, self.visited_list),
                              (self.entries_addr, self.entries),
                              (self.acc_visited_cnt_addr, self.acc_visited_cnt),
                              (self.graph_info_addr, self.graph_info),
                              (self.iteration_info_addr, self.iteration_info),
                              (self.partial_sum_addr, self.partial_sums_result),
                              (self.calculated_distance_addr, self.calculated_distances),
                              (self.tmp_entries_addr, self.entries),
                              (self.current_entry_distance_addr, self.calculated_distances),
                              (self.query_iteration_addr, self.query_iteration)])

    def read_file(self):
      graph = {}
      num_data = num_dims = max_level = max_m = max_m0 = enter_point = None
      qdata = []
      data = []

      with open(self.graph_file, 'r') as f:
        mode = 0 # 0: normal, 1: query, 2: data

        for line in f:
          line = line.strip()
          if not line or line.startswith("#"):
            if line == "# Query ID, Data":
              mode = 1
              continue
            elif line == "# Data ID, Data":
              mode = 2
              continue
            elif line.startswith("# num_query:"):
              num_query = int(line.split(":")[1].strip())
            elif line.startswith("# num_data:"):
              num_data = int(line.split(":")[1].strip())
            elif line.startswith("# num_dims:"):
              num_dims = int(line.split(":")[1].strip())
            elif line.startswith("# max_level:"):
              max_level = int(line.split(":")[1].strip())
            elif line.startswith("# max_m:"):
              parts = line.split(":")[1].split(",")
              max_m = int(line.split(":")[1].split(",")[0].strip())
              max_m0 = int(line.split(":")[-1].strip())
            elif line.startswith("# enter_point:"):
              enter_point = int(line.split(":")[1].strip())
            mode = 0
            continue

          # Query
          if mode == 1:
            q_idx = int(line.split(":")[0])
            q_data = list(map(int, line.split(":")[1].strip().split()))
            qdata.append(q_data)
          elif mode == 2:
            idx = int(line.split(":")[0])
            _data = list(map(int, line.split(":")[1].strip().split()))
            data.append(_data)

          # Data line: level src dst dist
          parts = line.split()
          if len(parts) != 4:
            continue  # skip invalid lines
          level, src, dst, dist = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])

          if level not in graph:
            graph[level] = {}
          if src not in graph[level]:
            graph[level][src] = []
          graph[level][src].append((dst, dist))
      return num_query, num_data, num_dims, max_level, max_m, max_m0, enter_point, graph, qdata, data

    def read_iter_file(self, max_level):
      entry_ids = []
      future_entry_ids = []
      candidate_ids = []
      partial_sums  = []
      calculated_distances  = []
      current_entry_distances  = []

      with open(self.iter_file, 'r') as f:
        mode = 0
        for line in f:
          line = line.strip()
          if not line or line.startswith("#"):
            if line.startswith("# Entry id:"):
              mode = 0
              continue
            if line.startswith("# Future entry id:"):
              mode = 1
              continue
            elif line.startswith("# Candidate id:"):
              mode = 2
              continue
            elif line.startswith("# Entry Partial Sum:"):
              mode = 3
              continue
            elif line.startswith("# Calculated Distance:"):
              mode = 4
              continue
            elif line.startswith("# Current Entry Distance:"):
              mode = 5
              continue
            continue

          if mode == 0:
            entry_id = int(line.split(":")[1].strip())
            entry_ids.append(entry_id)
          elif mode == 1:
            future_entry_id = int(line.split(":")[1].strip())
            future_entry_ids.append(future_entry_id)
          elif mode == 2:
            candidate_id = int(line.split(":")[1].strip())
            candidate_ids.append(candidate_id)
          elif mode == 3:
            partial_sum = list(map(int, line.split(":")[1].strip().split()))
            if len(partial_sum) < 64:
              partial_sum.extend([0] * (64 - len(partial_sum)))
            partial_sums.extend(partial_sum)
          elif mode == 4:
            calculated_distance = int(line.split(":")[1].strip())
            calculated_distances.append(calculated_distance)
          elif mode == 5:
            current_entry_distance = int(line.split(":")[1].strip())
            current_entry_distances.append(current_entry_distance)

      return entry_ids, future_entry_ids, candidate_ids, partial_sums, calculated_distances, current_entry_distances

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
  make_input_files("test_kernel_name", "test_code", input_map, input_map, [], "/root/workspace/M2NDP-public/examples/benchmarks/hnsw/GetEntryPointsKernel/test_input.txt")
  # print(kernel.make_kernel())
