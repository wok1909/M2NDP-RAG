import sys
sys.path.append("/root/workspace/M2NDP-public/examples")



from typing import Any
import numpy as np
import math
import os
from utils.utils import NdpKernel, make_memory_map, pad8, make_input_files
import configs

DEGUG = False
ITER = 1
LEVEL = 3
visited_list_size = 14

class GetEntryPointsKernel1(NdpKernel):
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
        # query-entry distance(initialize), current neighbor idx
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
        num_query, num_data, num_dims, max_level, max_m, max_m0, enter_point, graphs, queries, data = self.read_file()
        entry_id, future_entry_ids, candidate_ids, partial_sums_result, calculated_distances, current_entry_distances = self.read_iter_file(max_level)
        print("calculated_distance: ", calculated_distances)
        print("current_entry_distances: ", current_entry_distances)
        qdata_array = self.preprocess_with_data(queries, True)
        target_data_array = self.preprocess_with_data(data)
        visited_array = None
        visiteid_list_array = None
        entries_array = None
        acc_visited_cnt_array = None

        target_nodes, graph, degree = self.make_graph(graphs, LEVEL, max_m)

        print(f"graph: {graph}")

        visited_array = [0] *  configs.ndp_units * len(target_nodes)
        visited_list_array = [-1] * visited_list_size * len(target_nodes)
        entries_array = [enter_point] * num_query
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
        self.graph_info = pad8(np.array([num_query, num_data, num_dims, max_level, max_m, max_m0], dtype=np.int32)) # num_query, num_data, num_dims, max_level, max_m, max_m0, ITER
        self.partial_sum = pad8(np.array([0]*64*num_query, dtype=np.int32)) # FIXME: store to spad
        self.distance = pad8(np.array([0]*num_query, dtype=np.int32))

        self.partial_sums_result = pad8(np.array(partial_sums_result, dtype=np.int32))
        self.calculated_distances = pad8(np.array(calculated_distances, dtype=np.int32))
        print(f"future: {future_entry_ids}")
        self.tmp_entries = pad8(np.array(future_entry_ids, dtype=np.int32))
        self.current_entry_distance = pad8(np.array(current_entry_distances, dtype=np.int32))

        self.iteration_info = pad8(np.array([0, 0], dtype=np.int32))  # Current update, finish
        self.query_iteration = pad8(np.array([1] * num_query, dtype=np.int32))
        self.bound = len(qdata_array) * configs.data_size
        self.input_addrs = [self.qdata_addr, self.target_data_addr, self.target_nodes_addr, self.graph_addr, self.degree_addr,
                            self.visited_addr, self.visited_list_addr, self.entries_addr, self.acc_visited_cnt_addr,
                            self.graph_info_addr, self.iteration_info_addr, self.partial_sum_addr, self.calculated_distance_addr,
                            self.tmp_entries_addr, self.current_entry_distance_addr, self.query_iteration_addr]

        self.num_dims = num_dims

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

        # Get query data address
        template += f'ld x3, (x1)\n'  # qdata_addr
        template += f'add x4, x3, x2\n' # qdata_addr + offset (256 Byte, 64 elements)

        # Get num query
        template += f'ld x5, 72(x1)\n'  # graph_info_addr
        template += f'lw x6, (x5)\n'
        template += f'bge NDPID, x6, .SKIP0\n'

        # Get entry id
        template += f'ld x7, 56(x1)\n'  # entries_addr
        template += f'muli x8, NDPID, {data_size}\n'
        template += f'add x9, x7, x8\n' # entry address
        template += f'lw x10, (x9)\n' # entry_id

        template += f'ld x11, 16(x1)\n' # target_nodes_addr
        template += f'muli x12, x10, {data_size}\n'
        template += f'add x13, x11, x12\n'
        template += f'lw x14, (x13)\n'  # target_nodes[entry_id]

        # Get graph info current neighbor iteration
        template += f'ld x15, 120(x1)\n' # iteration_info_addr
        template += f'add x15, x15, x8\n'
        template += f'lw x16, (x15)\n'  # ITER

        # Get graph info max_m
        template += f'lw x17, 16(x5)\n' # max_m

        # Get neighbor address
        template += f'ld x18, 24(x1)\n' # graph_addr

        # Get degree
        template += f'ld x19, 32(x1)\n' # degree_addr
        template += f'muli x20, x10, {data_size}\n' # entry_id * data_size
        template += f'add x21, x19, x20\n'
        template += f'lw x22, (x21)\n'  # degree of entry_id

        # Get current candidate id
        template += f'mul x22, x10, x17\n'  # begin = max_m * entry_id
        template += f'add x23, x22, x16\n'  # begin + ITER
        template += f'muli x23, x23, {data_size}\n'
        template += f'add x23, x18, x23\n'
        template += f'lw x23, (x23)\n'  # candid = graph[begin + ITER]

        ###### SKIP IF VISITED ######

        template += f'muli x24, x23, {data_size}\n'
        template += f'add x25, x11, x24\n'
        template += f'lw x26, (x25)\n' # target_node[candid]

        # Get candidate data address
        template += f'ld x27, 8(x1)\n'  # target_data_addr
        template += f'lw x17, 8(x5)\n'  # num_dim
        template += f'muli x17, x17, {data_size}\n'
        template += f'mul x28, x17, x26\n'
        template += f'add x29, x27, x28\n'  # candidate data starting address

        # Add offset in data offset
        template += f'li x30, {configs.stride * configs.ndp_units}\n' # 8192
        template += f'div x31, x2, x30\n'
        template += f'muli x31, x31, {configs.stride}\n'
        template += f'add x29, x29, x31\n'
        template += f'rem x31, x2, x30\n'
        template += f'add x29, x29, x31\n'
        template += f'muli x31, NDPID, {configs.stride}\n'
        template += f'sub x29, x29, x31\n'  # candidate data address

        # Calculate and reduce
        template += f'li x30, 0\n'  # accumulator

        template += f'.LOOP0\n'
        # Get query and candidate data
        template += f'vle32.v v1, (x4)\n' # qdata
        template += f'vle32.v v2, (x29)\n'  # candidate data

        template += f'vsub.vv v3, v1, v2\n'
        template += f'vmul.vv v4, v3, v3\n'

        # Reduce
        template += f'vmv.v.x v0, x0\n'
        template += f'vredsum.vs v5, v4, v0\n'
        template += f'vmv.x.s x8, v5\n'
        template += f'add x30, x30, x8\n' # accumulate

        # Calculate store address and store parital sum
        template += f'ld x8, 88(x1)\n'  # partial_sum_addr
        template += f'li x9, {configs.stride * configs.ndp_units}\n'
        template += f'div x12, x2, x9\n'
        template += f'muli x12, x12, {packet_size}\n'
        template += f'add x8, x8, x12\n'

        template += f'rem x12, x2, x9\n'
        template += f'muli x13, NDPID, {configs.stride}\n'
        template += f'sub x12, x12, x13\n'
        template += f'li x13, {packet_size / data_size}\n'
        template += f'div x12, x12, x13\n'
        template += f'add x8, x8, x12\n'

        template += f'muli x20, NDPID, {64 * data_size}\n'  # Because currently storing to DRAM
        template += f'add x9, x8, x20\n'
        template += f'sw x30, (x9)\n'

        template += f'.SKIP0\n'
        # ------------------ SYNC ------------------
        template += f'KERNELBODY:\n'

        # Load info
        template += f'li x1, {configs.spad_addr}\n'
        template += f'ld x3, 72(x1)\n' # graph_info_addr
        template += f'lw x4, (x3)\n'  # num query

        template += f'bge NDPID, x4, .SKIP1\n'

        # Load partial sums
        template += f'ld x5, 88(x1)\n' # partial_sum_addr

        template += f'li x6, {64 * data_size}\n'
        template += f'mul x7, NDPID, x6\n'
        template += f'add x5, x5, x7\n'

        template += f'muli x8, NDPID, {configs.stride}\n'
        template += f'bne x2, x8, .SKIP1\n'  # skip if not micro thread 0

        # Reduce among micro threads
        template += f'li x9, 0\n'  # reduce address offset
        template += f'li x10, 0\n'  # accumulator
        template += f'.LOOP1\n' # global reduce loop

        # Load partial sum
        template += f'add x11, x5, x9\n'  # load address
        template += f'vle32.v v1, (x11)\n'

        # Reduce
        template += f'vmv.v.x v0, x0\n'
        template += f'vredsum.vs v2, v1, v0\n' # vector reduce
        template += f'vmv.x.s x12, v2\n'
        template += f'add x10, x10, x12\n'  # accumulate

        # Add 32(packet size) to offset and loop if small than 64 (upper bound)
        template += f'addi x9, x9, {packet_size}\n'
        template += f'li x13, {data_size * 64}\n' # max upper bound 64 micro thread
        template += f'blt x9, x13, .LOOP1\n'

        # Store distancee (Is not manditory)
        template += f'ld x9, 96(x1)\n' # distance_addr
        template += f'muli x13, NDPID, {data_size}\n'
        template += f'add x9, x9, x13\n'
        template += f'sw x10, (x9)\n' # store distance

        # Load query entry distance
        template += f'ld x14, 112(x1)\n' # current_entry_distance_addr
        template += f'muli x15, NDPID, {data_size}\n'
        template += f'add x15, x14, x15\n'
        template += f'lw x16, (x15)\n'

        # Get entry id
        template += f'ld x17, 56(x1)\n'  # entries_addr
        template += f'muli x18, NDPID, {data_size}\n'
        template += f'add x19, x17, x18\n' # entry address
        template += f'lw x20, (x19)\n' # entry_id

        # Get graph info ITER and max_m
        # Get graph info current neighbor iteration
        template += f'ld x19, 120(x1)\n' # iteration_info_addr
        template += f'add x19, x19, x18\n'
        template += f'lw x21, (x19)\n'  # ITER
        template += f'lw x22, 16(x3)\n' # max_m

        # Compare distance
        template += f'bge x10, x16, .SKIP3\n'

        template += f'ld x23, 24(x1)\n' # graph_addr
        # Get current candidate id
        template += f'mul x24, x20, x22\n'  # begin = max_m * entry_id
        template += f'add x25, x24, x21\n'  # begin + ITER
        template += f'muli x26, x25, {data_size}\n'
        template += f'add x27, x23, x26\n'
        template += f'lw x28, (x27)\n'  # candid = graph[begin + ITER]

        # if distance is smaller, store it to tmp entryid and tmp distance
        template += f'ld x29, 104(x1)\n'  # tmp_entries_addr
        template += f'muli x30, NDPID, {data_size}\n'
        template += f'add x30, x29, x30\n'
        template += f'sw x28, (x30)\n'  # store tmp entry_id
        template += f'sw x10, (x15)\n'  # store tmp distance

        template += f'.SKIP3\n'

        # if this is the last degree jump to update
        template += f'ld x31, 32(x1)\n' # degree_addr
        template += f'muli x7, x20, {data_size}\n' # entry_id * data_size
        template += f'add x7, x31, x7\n'
        template += f'lw x8, (x7)\n'  # degree of entry_id

        template += f'addi x21, x21, 1\n' # increase ITER
        template += f'TEST.v.x x21\n'
        template += f'TEST.v.x x8\n'
        template += f'blt x21, x8, .SKIP2\n'  # branch less than degree

        template += f'li x21, 0\n'  # initialize ITER to 0
        template += f'sw x21, (x19)\n'
        template += f'j .SKIP1\n'

        template += f'.SKIP2\n' 
        template += f'sw x21, (x19)\n'

        # Update if update is True then change entry id and iteration info to 0
        # if update is false, then change end sign

        template += f'.SKIP1\n'

        return template

    def make_input_map(self):
      if ITER == 1:
        from .kernel0 import GetEntryPointsKernel0
        kernel0 = GetEntryPointsKernel0()
        return kernel0.make_output_map()
      else:
        None
      # return make_memory_map([(self.qdata_addr, self.qdata),
      #                         # (self.qnodes_addr, self.qnodes),
      #                         (self.target_data_addr, self.target_data),
      #                         (self.target_nodes_addr, self.target_nodes),
      #                         (self.degree_addr, self.degree),
      #                         (self.visited_addr, self.visited),
      #                         (self.visited_list_addr, self.visited_list),
      #                         (self.entries_addr, self.entries),
      #                         (self.acc_visited_cnt_addr, self.acc_visited_cnt),
      #                         (self.graph_info_addr, self.graph_info),
      #                         (self.iteration_info_addr, self.iteration_info),
      #                         (self.partial_sum_addr, self.partial_sum),
      #                         (self.distance_addr, self.distance)])

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
                              (self.tmp_entries_addr, self.tmp_entries),
                              (self.current_entry_distance_addr, self.current_entry_distance),
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
      print(f"is_query: {is_query}")
      print(f"n_data: {n_data}")
      print(f"pad_dim: {pad_dim}")
      print(f"data_size : {data_size}")
      # exit(0)
      processed_data = [0] * data_size
      # if is_query:
      #   stride_stride = elem_stride * n_data
      #   for query_idx in range(n_data):
      #     for o_loop in range(int(pad_dim / elem_stride)):
      #       for i_loop in range(elem_stride):
      #         # print("idx: ", o_loop * elem_stride + i_loop)
      #         if o_loop * elem_stride + i_loop < dim:
      #           val = data[query_idx][o_loop * elem_stride + i_loop]

      #         # print(val)
      #         processed_data[o_loop * stride_stride + query_idx * elem_stride + i_loop] = val

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

  kernel = GetEntryPointsKernel1()
  # input_map = kernel.make_input_map()
  # output_map = kernel.make_output_map()
  # make_input_files("test_kernel_name", "test_code", input_map, input_map, [], "/root/workspace/M2NDP-public/examples/benchmarks/hnsw/GetEntryPointsKernel/test_input.txt")
  # print(kernel.make_kernel())
