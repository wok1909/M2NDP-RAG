import sys
sys.path.append("/root/workspace/M2NDP-public/examples")



from typing import Any
import numpy as np
import math
import os
from utils.utils import NdpKernel, make_memory_map, pad8, make_input_files
import configs

DEGUG = False
LEVEL = 3
visited_list_size = 14

class GetEntryPointsKernel(NdpKernel):
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
        self.iteration_info_addr = 0x8a0000000000




        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'hnsw_GetEntryPointsKernel'
        self.input_file = os.path.join(os.path.dirname(__file__), f'../text.txt')
        # self.input_file = os.path.join(os.path.dirname(__file__), f'../../data/USA-road-d.NY.gr')

        # input memorymap
        num_data, num_dims, max_level, max_m, max_m0, enter_point, graph, queries, data = self.read_file()
        print(queries)
        qdata_array = self.preprocess_with_data(queries, True)
        target_data_array = self.preprocess_with_data(data)
        target_nodes = []
        graphs = []
        degrees = []
        visited_array = None
        visiteid_list_array = None
        entries_array = None
        acc_visited_cnt_array = None

        for level in range(max_level+1):
          upper_node, neighbors, degree = self.make_graph(graph, level, max_m)
          target_nodes.append(upper_node)
          graphs.append(neighbors)
          degrees.append(degree)

        visited_array = [0] *  configs.ndp_units * len(target_nodes[LEVEL])
        visited_list_array = [-1] * visited_list_size * len(target_nodes[LEVEL])
        entries_array = [enter_point] * len(queries)
        acc_visited_cnt_array = [0] * len(target_nodes[LEVEL])

        self.qdata = pad8(np.array(qdata_array, dtype=np.int32))
        self.target_data = pad8(np.array(target_data_array, dtype=np.int32))
        self.target_nodes = pad8(np.array(target_nodes[LEVEL], dtype=np.int32))
        self.graph = pad8(np.array(graphs[LEVEL], dtype=np.int32))
        self.degree = pad8(np.array(degrees[LEVEL], dtype=np.int32))
        self.visited = pad8(np.array(visited_array, dtype=np.int32))
        self.visited_list = pad8(np.array(visited_list_array, dtype=np.int32))
        self.entries = pad8(np.array(entries_array, dtype=np.int32))
        self.acc_visited_cnt = pad8(np.array(acc_visited_cnt_array, dtype=np.int32))
        self.iteration_info = pad8(np.array([0, 0], dtype=np.int32))
        self.query_entry_distance = pad8(np.array([0]), dtype=np.int32)
        self.current_neighbor_idx = pad8(np.array([0], dtype=np.int32))

        self.input_addrs = [self.qdata_addr, self.target_data_addr, self.target_nodes_addr, self.graph_addr, self.degree_addr,
                            self.visited_addr, self.visited_list_addr, self.entries_addr, self.acc_visited_cnt_addr, self.iteration_info_addr,
                            num_data, num_dims, max_level, max_m, max_m0, enter_point]

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
        
        # Query의 data를 가져오기 (NDPID)
        template += f'ld x13, (x1)' # qdata_addr
        template += f'li x3, 256\n'
        template += f'div x3, x2, x3\n' # micro thread idx
        template += f'muli x4, x3, {packet_size}\n' # Offset for this micro thread
        template += f'add x5, x13, x4\n' # query data to load

        # 현재의 entry node의 id 가져오기
        template += f'ld x14, 56(x1)\n' # entries_addr
        template += f'muli x6, NDPID, 4\n'  # entries array offset
        template += f'add x7, x14, x6\n'  # entry id to load
        template += f'ld x8, (x7)\n' # entry id

        # Entry id를 이용하여 node id를 찾기 (query와의 distance 계산을 위해)
        template += f'ld x15, 16(x1)\n' # target_node_addr
        template += f'muli x9, x8, {data_size}\n'
        template += f'add x10, x15, x9\n' # node id addr
        template += f'ld x11, (x10)\n' # node id
        
        # Entry id로 degree 찾기
        template += f'ld x16, 32(x1)\n' # degree_addr
        template += f'add x17, x16, x9\n'
        template += f'ld x18, (x17)\n' # entry node degree

        # 

        


        # 현재 몇번째 neighbor node를 처리하는지 확인 후 해당 node data 가져오기
        # 가져온 node data와 query node data를 vector unit을 이용하여 distance 계산하기
        # vector 0번째 index에 reduce sum
        # reduce sum 값을 store 하기
        template += f'KERNELBODY:\n'
        # 위에서 저장한 reduce sum을 다 load 하여 reduce sum 하여 최종 distance 계산하기
        # 구한 distance 와 entry node의 distance를 비교하여 만약 distance가 더 작으면 temporal entry node 정보 update
        # entry node neighbor 전체에 대하여 모든 loop을 돌았으면 현재 entryid를 수정
        # 만약 entries[i]와 entryid의 값이 동일하면 (update가 필요 없다면) 종료 process 밟기


        return template

    def make_input_map(self):
      return make_memory_map([(self.qdata_addr, self.qdata),
                              # (self.qnodes_addr, self.qnodes),
                              (self.target_data_addr, self.target_data),
                              (self.target_nodes_addr, self.target_nodes),
                              (self.degree_addr, self.degree),
                              (self.visited_addr, self.visited),
                              (self.visited_list_addr, self.visited_list),
                              (self.entries_addr, self.entries),
                              (self.acc_visited_cnt_addr, self.acc_visited_cnt)])

    def make_output_map(self):
      return make_memory_map([(self.info_addr, self.o_graph_info),
                              (self.row_addr, self.o_row),
                              (self.col_addr, self.o_col),
                              (self.data_addr, self.o_data),
                              (self.vector1_addr, self.o_vector1),
                              (self.vector2_addr, self.o_vector2)])

    def read_file(self):
      graph = {}
      num_data = num_dims = max_level = max_m = max_m0 = enter_point = None
      qdata = []
      data = []

      with open(self.input_file, 'r') as f:
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

      return num_data, num_dims, max_level, max_m, max_m0, enter_point, graph, qdata, data

    def preprocess_with_data(self, data, is_query=False):
      stride = configs.stride
      elem_stride = int(stride / 4)
      n_data = len(data)
      dim = len(data[0])
      print(dim, elem_stride)
      pad_dim = (dim + elem_stride - 1) // elem_stride * elem_stride
      data_size = n_data * pad_dim
      processed_data = [0] * data_size
      if is_query:
        stride_stride = elem_stride * n_data
        for query_idx in range(n_data):
          for o_loop in range(int(pad_dim / elem_stride)):
            for i_loop in range(elem_stride):
              # print("idx: ", o_loop * elem_stride + i_loop)
              if o_loop * elem_stride + i_loop < dim:
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
        print("neighbors len: ", len(neighbors))
        if level == 3:
          print(graph)

      if level ==3 :
        for idx, (src, dst_info) in enumerate(graph.items()):
          upper_node.append(src)
          degree[idx] = len(dst_info)
          offset = idx * max_m
          for n_idx in range(len(dst_info)):
            neighbors[offset + n_idx] = dst_info[n_idx][0]

        if DEGUG:
          print(neighbors)
          print(degree)

      return upper_node, neighbors, degree



if __name__ == "__main__":

  kernel = GetEntryPointsKernel()
  input_map = kernel.make_input_map()
  make_input_files("test_kernel_name", "test_code", input_map, input_map, [], "/root/workspace/M2NDP-public/examples/benchmarks/hnsw/GetEntryPointsKernel/test_input.txt")
  print(kernel.make_kernel())
