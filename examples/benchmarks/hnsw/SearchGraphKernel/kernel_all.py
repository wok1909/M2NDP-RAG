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
INFINITY = 9999999

ENTRY_INIT_ID = -1
DISTANCE_INIT_VALUE = 999999

class SearchGraphKernel(NdpKernel):
    def __init__(self):
        super().__init__()
        self.INT32_SIZE = 4
        self.INT_MAX = 99999999

        # DRAM
        self.qdata_addr = 0x800000000000
        self.data_addr = 0x810000000000
        self.entries_addr = 0x820000000000
        self.graph_addr = 0x830000000000
        self.degree_addr = 0x840000000000
        self.nns_addr = 0x850000000000
        self.distances_addr = 0x860000000000
        self.found_cnt_addr = 0x870000000000
        self.visited_table_addr = 0x880000000000
        self.visited_list_addr =  0x890000000000
        self.acc_visited_cnt_addr = 0x8a0000000000
        self.neighbors_addr = 0x8b0000000000
        self.global_cand_nodes_addr = 0x8c0000000000
        self.global_cand_distances_addr = 0x8d0000000000
        self.graph_info_addr = 0x8e0000000000

        # Scratchpad
        self.qdata_sapd_addr = 0x1000000000100000
        self.pq_size = 0x1000000000200000
        self.visited_cnt = 0x1000000000300000
        self.check_exist = 0x1000000000400000
        self.partial_cand = 0x1000000000500000
        self.partial_sum_addr = 0x1000000000700000

        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'hnsw_SearchGraphKernel'
        self.smem_size = 0x1000000

        # graph info
        self.graph_file = os.path.join(os.path.dirname(__file__), f'../data/graph.txt')
        self.num_query, self.num_data, self.num_dims, self.max_level, self.max_m, self.max_m0, self.enter_point, graphs, queries, data = read_graph_file(self.graph_file)
        qdata_array = self.preprocess_with_data(queries, True)
        target_data_array = self.preprocess_with_data(data)
        target_nodes, graph, degree = self.make_graph(graphs, 0, self.max_m0)

        self.qdata = pad8(np.array(qdata_array, dtype=np.int32))
        self.data = pad8(np.array(target_data_array, dtype=np.int32))
        self.graph = pad8(np.array(graph, dtype=np.int32))
        self.degree = pad8(np.array(degree, dtype=np.int32))

        # input info
        self.input_file = os.path.join(os.path.dirname(__file__), f'../data/SearchGraph_start.txt')
        topk, visited_table_size, visited_list_size, ef_search, entries, input_nns, input_distances, input_found_cnt, input_visited_table, input_visited_list, input_acc_visited_cnt, input_neighbors, input_global_cand_nodes, input_global_cand_distances = read_step2_file(self.input_file)

        self.graph_info = pad8(np.array([self.num_query, len(target_nodes), self.num_dims, self.max_level, self.max_m0, DIST_TYPE, topk, visited_table_size, visited_list_size, ef_search], dtype=np.int32))

        self.entries = pad8(np.array(entries, dtype=np.int32))
        self.input_nns = pad8(np.array(input_nns, dtype=np.int32))
        self.input_distances = pad8(np.array(input_distances, dtype=np.int32))
        self.input_found_cnt = pad8(np.array(input_found_cnt, dtype=np.int32))
        self.input_visited_table = pad8(np.array(input_visited_table, dtype=np.int32))
        self.input_visited_list = pad8(np.array(input_visited_list, dtype=np.int32))
        self.input_acc_visited_cnt = pad8(np.array(input_acc_visited_cnt, dtype=np.int32))
        self.input_neighbors = pad8(np.array(input_neighbors, dtype=np.int32))
        self.input_global_cand_nodes = pad8(np.array(input_global_cand_nodes, dtype=np.int32))
        self.input_global_cand_distances = pad8(np.array(input_global_cand_distances, dtype=np.int32))

        # output info
        self.output_file = os.path.join(os.path.dirname(__file__), f'../data/SearchGraph_finish.txt')
        _, _, _, _, _, output_nns, output_distances, output_found_cnt, output_visited_table, output_visited_list, output_acc_visited_cnt, output_neighbors, output_global_cand_nodes, output_global_cand_distances = read_step2_file(self.output_file)

        self.output_nns = pad8(np.array(output_nns, dtype=np.int32))
        self.output_distances = pad8(np.array(output_distances, dtype=np.int32))
        self.output_found_cnt = pad8(np.array(output_found_cnt, dtype=np.int32))
        self.output_visited_table = pad8(np.array(output_visited_table, dtype=np.int32))
        self.output_visited_list = pad8(np.array(output_visited_list, dtype=np.int32))
        self.output_acc_visited_cnt = pad8(np.array(output_acc_visited_cnt, dtype=np.int32))
        self.output_neighbors = pad8(np.array(output_neighbors, dtype=np.int32))
        self.output_global_cand_nodes = pad8(np.array(output_global_cand_nodes, dtype=np.int32))
        self.output_global_cand_distances = pad8(np.array(output_global_cand_distances, dtype=np.int32))

        self.bound = len(qdata_array) * configs.data_size
        self.input_addrs = [self.qdata_addr, self.data_addr, self.entries_addr, self.graph_addr, self.degree_addr, self.nns_addr, self.distances_addr,
                            self.found_cnt_addr, self.visited_table_addr, self.visited_list_addr, self.acc_visited_cnt_addr, self.neighbors_addr, self.global_cand_nodes_addr, self.global_cand_distances_addr, self.graph_info_addr,
                            self.qdata_sapd_addr, self.pq_size, self.visited_cnt, self.check_exist, self.partial_cand, self.partial_sum_addr]



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

        # Get graph info
        template += f'lw x5, 8(x3)\n'   # num_dims
        template += f'lw x6, 16(x3)\n'  # max_m
        template += f'lw x7, 36(x3)\n'  # ef_search
        template += f'lw x8, 28(x3)\n'  # visited_table_size
        template += f'lw x9, 32(x3)\n' # visited_list_size

        # Set pq_size, queue, cand_nodes and cand_distances address
        # template += f'ld x10, {self.get_arg_offset(self.pq_size)}(x1)\n'
        template += f'ld x11, {self.get_arg_offset(self.neighbors_addr)}(x1)\n'

        # Query loop
        template += f'li x14, 0\n'  # loop idx
        template += f'.LOOP0\n'
        template += f'muli x15, x14, {configs.ndp_units}\n'
        template += f'add x15, x15, NDPID\n'  # query idx
        template += f'bge x15, x4, .SKIP0\n'

        # Set pq, cand_nodes, cand_distances
        template += f'ld x16, {self.get_arg_offset(self.neighbors_addr)}(x1)\n'
        # template += f'ld x17, {self.get_arg_offset(self.global_cand_nodes_addr)}(x1)\n'
        # template += f'ld x18, {self.get_arg_offset(self.global_cand_distances_addr)}(x1)\n'
        template += f'mul x21, x7, x15\n'
        template += f'muli x21, x21, {data_size}\n'
        template += f'add x16, x16, x21\n'  # ef_search_pq
        # template += f'add x17, x17, x21\n'  # cand_nodes
        # template += f'add x18, x18, x21\n'  # cand_distances

        # Set visited_table, visited_list address
        template += f'ld x19, {self.get_arg_offset(self.visited_table_addr)}(x1)\n'
        template += f'ld x20, {self.get_arg_offset(self.visited_list_addr)}(x1)\n'
        template += f'mul x21, x8, x15\n'
        template += f'muli x21, x21, {data_size}\n'
        template += f'add x19, x19, x21\n'  # _visited_table
        template += f'mul x21, x9, x15\n'
        template += f'muli x21, x21, {data_size}\n'
        template += f'add x20, x20, x21\n'  # _visited_list

        # Initialize size, visited_cnt
        template += f'ld x21, {self.get_arg_offset(self.pq_size)}(x1)\n'
        # template += f'ld x22, {self.get_arg_offset(self.visited_cnt)}(x1)\n'
        # template += f'li x21, 0\n'  # size
        template += f'sw x0, (x21)\n' # size = 0
        template += f'li x22, 0\n'  # visited_cnt
        template += f'li x31, 1\n'

        # Query
        template += f'ld x23, {self.get_arg_offset(self.qdata_addr)}(x1)\n'
        template += f'ld x24, {self.get_arg_offset(self.entries_addr)}(x1)\n'
        template += f'add x23, x23, x2\n'
        template += f'vle32.v v1, (x23)\n'  # qdata
        template += f'muli x25, x15, {data_size}\n'
        template += f'add x25, x24, x25\n'
        template += f'lw x24, (x25)\n'  # entries[qidx]

        ### PushNodeToSearchPq Start ###
        template += f'.LOOP20\n'
        # CheckAlreadyExists
        template += f'ld x3, {self.get_arg_offset(self.check_exist)}(x1)\n'
        template += f'ld x30, {self.get_arg_offset(self.pq_size)}(x1)\n'
        template += f'lw x21, (x30)\n'  # size
        template += f'muli x25, UTHREADID, {data_size}\n'
        template += f'add x18, x3, x25\n'
        template += f'li x25, 0\n'  # exists = false
        template += f'addi x28, UTHREADID, 0\n'  # pq idx
        template += f'.LOOP1\n'
        template += f'bge x28, x21, .SKIP1\n'
        template += f'muli x29, x28, {3 * data_size}\n'
        template += f'add x29, x16, x29\n'
        template += f'lw x23, 4(x29)\n'  # nodeid
        template += f'sub x30, x24, x23\n'
        template += f'seqz x29, x30\n'
        template += f'add x28, x28, UTHREADSZ\n'
        template += f'or x25, x29, x25\n'
        template += f'j .LOOP1\n'
        template += f'.SKIP1\n'
        template += f'sw x25, (x18)\n' # store exists

        template += f'KERNELBODY:\n'  # KERNELBODY1
        # Reduce exists
        template += f'bgt UTHREADID, x0, .SKIP2\n'
        template += f'li x25, 0\n'  # exists = false
        template += f'addi x28, UTHREADSZ, 0\n' # copy size
        template += f'vid.v v2\n' # v2 = [0, 1, 2, 3, 4, 5, 6, 7]
        template += f'vmv.v.x v0, x0\n'
        template += f'.LOOP2\n'
        template += f'blez x28, .SKIP2\n'
        template += f'vle32.v v3, (x18)\n'
        template += f'vmslt.vx v4, v2, x28\n'
        template += f'vredor.vs v5, v3, v0, v4\n'
        template += f'vmv.x.s x25, v5\n'
        template += f'addi x18, x18, {packet_size}\n'
        template += f'addi x28, x28, -{packet_size / data_size}\n'
        template += f'beqz x25, .LOOP2\n'

        # Store exists to spad and load
        template += f'sw x25, (x3)\n'
        template += f'.SKIP2\n'
        

        template += f'KERNELBODY:\n'  # KERNELBODY2
        template += f'lw x18, (x3)\n' # exists

        # When x31 is 1 -> go to CheckVisited LOOP21
        # When x31 is 0 -> go to SKIP20
        template += f'beqz x18, .SKIP70\n'
        template += f'beqz x31, .SKIP20\n'  # from while loop
        template += f'j .LOOP21\n'

        # Distance calculation
        template += f'.SKIP70\n'
        template += f'ld x18, {self.get_arg_offset(self.partial_sum_addr)}(x1)\n'
        template += f'ld x25, {self.get_arg_offset(self.data_addr)}(x1)\n'
        template += f'mul x28, x5, x24\n'
        template += f'muli x28, x28, {data_size}\n'
        template += f'add x25, x25, x28\n'
        template += f'muli x28, UTHREADID, {packet_size}\n'
        template += f'add x25, x25, x28\n'  # data_addr

        template += f'vle32.v v2, (x25)\n'
        template += f'vsub.vv v3, v2, v1\n'
        template += f'vmul.vv v4, v3, v3\n'
        template += f'vmv.v.x v0, x0\n'
        template += f'vredsum.vs v5, v4, v0\n'
        template += f'vmv.x.s x25, v5\n'
        template += f'muli x28, UTHREADID, {data_size}\n'
        template += f'add x18, x18, x28\n'
        template += f'sw x25, (x18)\n'

        template += f'KERNELBODY:\n'  # KERNELBODY3
        template += f'bgt UTHREADID, x0, .SKIP3\n'

        # Reduce distance
        template += f'li x25, 0\n'  # accumulator
        template += f'addi x28, UTHREADSZ, 0\n'
        template += f'vid.v v2\n' # v2 = [0, 1, 2, 3, 4, 5, 6, 7]

        template += f'addi x30, x18, 0\n' # copy parital_sum_addr
        template += f'.LOOP3\n'
        template += f'vle32.v v3, (x30)\n'
        template += f'vmslt.vx v4, v2, x28\n'
        template += f'vredsum.vs v5, v3, v0, v4\n'
        template += f'vmv.x.s x29, v5\n'
        template += f'add x25, x25, x29\n'
        template += f'addi x28, x28, -{packet_size / data_size}\n'
        template += f'addi x30, x30, {packet_size}\n'
        template += f'bgt x28, x0, .LOOP3\n'

        # Push to pq
        # size: x21, ef_search: x7
        template += f'ld x30, {self.get_arg_offset(self.pq_size)}(x1)\n'
        template += f'lw x21, (x30)\n'
        template += f'blt x21, x7, .SKIP4\n'
        
        template += f'lw x28, (x16)\n'  # pq[0].distance
        template += f'ble x28, x25, .SKIP3\n'

        # PqPop()
        # if (*size == 0) return;
        template += f'beqz x21, .SKIP4\n'
        # (*size)--
        template += f'addi x21, x21, -1\n'
        template += f'sw x21, (x30)\n'
        # if (*size == 0) return;
        template += f'beqz x21, .SKIP4\n'
        # cuda_scalar tail_dist = pq[*size].distance;
        template += f'muli x3, x21, {3 * data_size}\n'
        template += f'add x28, x16, x3\n'
        template += f'lw x3, (x28)\n'  # pq[*size].distance
        # int p = 0, r = 1;
        template += f'li x10, 0\n'  # p = 0
        template += f'li x13, 1\n'  # r = 1
        # while (r < *size) {
        template += f'.LOOP4\n'
        template += f'bge x13, x21, .SKIP5\n'
        # if (r < (*size) - 1 and gt(pq[r + 1].distance, pq[r].distance))
        template += f'addi x28, x21, -1\n'  # (*size) - 1
        template += f'bge x13, x28, .SKIP6\n'
        template += f'muli x28, x13, {3 * data_size}\n'
        template += f'add x28, x16, x28\n'
        template += f'lw x29, (x28)\n'  # pq[r].distance
        template += f'lw x30, 12(x28)\n' # pq[r+1].distance
        template += f'bge x29, x30, .SKIP6\n'
        # r++;
        template += f'addi x13, x13, 1\n'
        template += f'.SKIP6\n'
        # if (ge(tail_dist, pq[r].distance)) break;
        template += f'muli x28, x13, {3 * data_size}\n'
        template += f'add x28, x16, x28\n'
        template += f'lw x29, (x28)\n'  # pq[r].distance
        template += f'bge x3, x29, .SKIP5\n'

        # pq[p] = pq[r];
        template += f'muli x30, x10, {3 * data_size}\n'
        template += f'add x30, x16, x30\n'
        template += f'sw x29, (x30)\n'
        template += f'lw x29, 4(x28)\n'
        template += f'lw x18, 8(x28)\n'
        template += f'addi x10, x13, 0\n'
        template += f'muli x13, x10, 2\n'
        template += f'addi x13, x13, 1\n'
        template += f'sw x29, 4(x30)\n'
        template += f'sw x18, 8(x30)\n'
        template += f'j .LOOP4\n'
        template += f'.SKIP5\n'
        # pq[p] = pq[*size];
        template += f'muli x3, x21, {3 * data_size}\n'
        template += f'add x28, x16, x3\n'
        template += f'lw x29, (x28)\n'
        template += f'lw x30, 4(x28)\n'
        template += f'lw x18, 8(x28)\n'
        template += f'muli x13, x10, {3 * data_size}\n'
        template += f'add x13, x16, x13\n'  # pq[p]
        template += f'sw x29, (x13)\n'
        template += f'sw x30, 4(x13)\n'
        template += f'sw x18, 8(x13)\n'
        # template += f'TEST.v.x x10\n'
        # template += f'TEST.v.x x30\n'

        template += f'.SKIP4\n'
        # PqPush()
        template += f'addi x3, x21, 0\n' # idx = *size
        # template += f'TEST.v.x x25\n'
        # template += f'TEST.v.x x3\n'
        template += f'.LOOP5\n'
        template += f'ble x3, x0, .SKIP7\n'
        template += f'addi x10, x3, 1\n'
        template += f'srli x10, x10, 1\n'
        template += f'addi x10, x10, -1\n'  # nidx = (idx + 1) / 2 - 1

        template += f'muli x13, x10, {3 * data_size}\n'
        template += f'add x13, x16, x13\n'
        template += f'lw x28, (x13)\n' # pq[nidx].distance
        template += f'bge x28, x25, .SKIP7\n'
        template += f'muli x29, x3, {3 * data_size}\n'
        template += f'add x29, x16, x29\n'  # pq[idx]
        
        template += f'lw x30, 4(x13)\n'
        template += f'lw x18, 8(x13)\n'
        template += f'addi x3, x10, 0\n'  # idx = nidx
        template += f'sw x28, (x29)\n'
        template += f'sw x30, 4(x29)\n'
        template += f'sw x18, 8(x29)\n'

        template += f'j .LOOP5\n'
        template += f'.SKIP7\n'
        template += f'muli x10, x3, {3 * data_size}\n'
        template += f'add x10, x16, x10\n'
        template += f'sw x25, (x10)\n'
        template += f'sw x24, 4(x10)\n'
        template += f'sw x0, 8(x10)\n'
        # template += f'TEST.v.x x3\n'
        # template += f'TEST.v.x x24\n'
        template += f'addi x21, x21, 1\n'
        template += f'ld x3, {self.get_arg_offset(self.pq_size)}(x1)\n'
        template += f'sw x21, (x3)\n'

        # # FOR DEBUG
        # template += f'TEST.v.x x0\n'
        # template += f'TEST.v.x x0\n'
        # template += f'TEST.v.x x0\n'
        # template += f'li x3, 0\n'
        # template += f'.LOOP99\n'
        # template += f'bge x3, x21, .SKIP99\n'
        # template += f'muli x10, x3, {3 * data_size}\n'
        # template += f'add x10, x16, x10\n'
        # template += f'lw x30, 4(x10)\n'
        # template += f'TEST.v.x x30\n'
        # template += f'lw x30, (x10)\n'
        # template += f'TEST.v.x x30\n'
        # template += f'lw x30, 8(x10)\n'
        # template += f'TEST.v.x x30\n'
        # template += f'addi x3, x3, 1\n'
        # template += f'j .LOOP99\n'
        # template += f'.SKIP99\n'

        template += f'.SKIP3\n'

        template += f'beqz x31, .SKIP20\n'  # from while loop

        ### PushNodeToSearchPq End ###

        
        template += f'KERNELBODY:\n'  # KERNELBODY4
        
        ### CheckVisited Start ###
        template += f'.LOOP21\n'
        template += f'li x23, 0\n'  # ret = false
        template += f'bge x22, x9, .SKIP8\n'
        template += f'rem x25, x24, x8\n' # idx = entryid % visited_table_size;
        template += f'muli x29, x25, {data_size}\n'
        template += f'add x29, x19, x29\n'
        template += f'lw x30, (x29)\n'  # visited_table[idx]
        
        template += f'KERNELBODY:\n'  # KERNELBODY5
        template += f'beq x30, x24, .SKIP9\n' # visited_table[idx] == target
        template += f'bgt UTHREADID, x0, .SKIP8\n'
        template += f'li x28, -1\n'
        template += f'bne x30, x28, .SKIP8\n'
        template += f'sw x24, (x29)\n'  # visited_table[idx] = target
        template += f'muli x28, x22, {data_size}\n'
        template += f'add x28, x20, x28\n'
        template += f'sw x25, (x28)\n'  # visite_list[visited_cnt] = idx
        template += f'addi x22, x22, 1\n'
        template += f'j .SKIP8\n'

        template += f'.SKIP9\n'
        template += f'li x23, 1\n'
        template += f'.SKIP8\n'
        template += f'beqz x31, .SKIP21\n'  # from while loop
        template += f'bnez x23, .SKIP10\n'

        ### CheckVisited End ###

        template += f'KERNELBODY:\n'  # KERNELBODY6
        ### GetCand Start ###
        template += f'.LOOP22\n'
        template += f'ld x29, {self.get_arg_offset(self.partial_cand)}(x1)\n'
        template += f'ld x30, {self.get_arg_offset(self.pq_size)}(x1)\n'
        template += f'lw x21, (x30)\n'  # size
        template += f'li x23, {INFINITY}\n' # dist
        template += f'li x25, -1\n' # cand
        template += f'addi x26, UTHREADID, 0\n'
        template += f'.LOOP6\n'
        template += f'bge x26, x21, .SKIP12\n'
        template += f'muli x27, x26, {3 * data_size}\n'
        template += f'add x27, x16, x27\n'
        template += f'lw x28, 8(x27)\n'  # pq[i].checked
        template += f'bnez x28, .SKIP11\n'
        template += f'lw x28, (x27)\n'  # pq[i].distance
        template += f'bge x28, x23, .SKIP11\n'
        template += f'addi x25, x26, 0\n' # cand = i;
        template += f'addi x23, x28, 0\n' # dist = pq[i].distance
        template += f'.SKIP11\n'
        template += f'add x26, x26, UTHREADSZ\n'
        template += f'j .LOOP6\n'

        template += f'.SKIP12\n'
        template += f'muli x27, UTHREADID, {data_size}\n'
        template += f'add x30, x29, x27\n'
        template += f'sw x25, (x30)\n'
        
        template += f'KERNELBODY:\n'  # KERNELBODY7
        # Global cand
        template += f'bgt UTHREADID, x0, .SKIP13\n'
        template += f'addi x18, UTHREADSZ, 0\n' # counter
        template += f'li x23, {INFINITY}\n'
        template += f'li x25, -1\n' # NEED THIS??
        template += f'vid.v v2\n' # v2 = [0, 1, 2, 3, 4, 5, 6, 7]

        template += f'.LOOP7\n'
        template += f'vle32.v v3, (x30)\n'  # load local i
        template += f'vmslt.vx v4, v2, x18\n'
        # template += f'vmv.x.s x26, v3\n'
        # template += f'blt x26, x0, .SKIP14\n'
        # Find max and if max is -1 then continue
        template += f'li x28, -{INFINITY}\n'
        template += f'vmv.v.x v0, x28\n'
        template += f'vredmax.vs v6, v3, v0, v4\n'
        template += f'vmv.x.s x26, v6\n'
        template += f'blt x26, x0, .SKIP15\n'
        template += f'vmseq.vi v6, v3, -1\n'
        template += f'vmnot.m v6, v6\n'
        template += f'vmand.mm v4, v4, v6\n'
        template += f'vmul.vi v7, v3, {3 * data_size}\n'
        template += f'vluxei32.v v5, (x16), v7, v4\n' # index load distance
        template += f'li x28, {INFINITY}\n'
        template += f'vmv.v.x v0, x28\n'
        template += f'vredmin.vs v6, v5, v0, v4\n'
        template += f'vmv.x.s x26, v6\n'
        template += f'vmv.v.x v8, x26\n' # [min, min, ...]
        template += f'vmseq.vv v9, v5, v8\n'  # v9[i] = (v5[i] == min) ? 1 : 0
        template += f'vfirst.m x13, v9\n'
        template += f'csrw vstart, x13\n'
        template += f'vmv.x.s x13, v3\n' # vector1[count]
        template += f'ble x23, x26, .SKIP15\n'
        template += f'addi x23, x26, 0\n'
        template += f'addi x25, x13, 0\n'
        template += f'.SKIP15\n'
        template += f'addi x18, x18, -{packet_size / data_size}\n'
        template += f'addi x30, x30, {packet_size}\n'
        template += f'bgt x18, x0, .LOOP7\n'

        template += f'.SKIP14\n'
        template += f'sw x25, (x29)\n'
        template += f'.SKIP13\n'
        template += f'beqz x31, .SKIP22\n'  # from while loop

        ### GetCand End ###
        template += f'KERNELBODY:\n'  # KERNELBODY8
        template += f'li x31, 0\n'
        template += f'lw x25, (x29)\n' # idx
        template += f'.LOOP8\n'
        template += f'blt x25, x0, .SKIP18\n'
        template += f'TEST.v.x x25\n'
        template += f'muli x23, x25, {3 * data_size}\n'
        template += f'add x23, x16, x23\n'
        template += f'bgt UTHREADID, x0, .SKIP16\n'
        template += f'li x26, 1\n'
        template += f'sw x26, 8(x23)\n' # ef_serach_pq[idx].checked = true
        template += f'.SKIP16\n'
        template += f'lw x26, 4(x23)\n' # entry = ef_search_pq[idx].nodeid
        template += f'TEST.v.x x26\n'

        template += f'KERNELBODY:\n'  # KERNELBODY9
        template += f'ld x23, {self.get_arg_offset(self.degree_addr)}(x1)\n'
        template += f'muli x27, x26, {data_size}\n'
        template += f'add x27, x23, x27\n'
        template += f'lw x12, (x27)\n'  # deg[entry_id]
        
        template += f'ld x17, {self.get_arg_offset(self.graph_addr)}(x1)\n'
        template += f'mul x26, x6, x26\n'  # j = max_m * entry
        template += f'add x27, x26, x12\n'  # max_m * entry + degree[entry]
        template += f'.LOOP9\n'
        template += f'bge x26, x27, .SKIP17\n'
        template += f'muli x28, x26, {data_size}\n'
        template += f'add x28, x17, x28\n'
        template += f'lw x24, (x28)\n'  # graph[j]

        # Check Visited
        template += f'j .LOOP21\n'  # CheckVisited()
        template += f'.SKIP21\n'
        template += f'addi x26, x26, 1\n' # j++
        template += f'bnez x23, .LOOP9\n' # CheckVisited() != false
        
        template += f'KERNELBODY:\n'  # KERNELBODY10
        template += f'TEST.v.x x24\n'
        template += f'j .LOOP20\n'  # PushNodeToSearchPq()
        template += f'.SKIP20\n'

        template += f'KERNELBODY:\n'  # KERNELBODY11

        template += f'j .LOOP9\n'
        # go back to .LOOP9

        template += f'.SKIP17\n'
        
        # 필요한  reg: x26, x27, x31, x17
        # 필요없는 reg: x23, x25, x28, x29, x30, x10, x13, , x18
        template += f'j .LOOP22\n'  # GetCand()
        template += f'.SKIP22\n'
        template += f'lw x25, (x29)\n' # idx

        template += f'j .LOOP8\n'

        template += f'.SKIP18\n'

        template += f'KERNELBODY:\n'  # KERNELBODY12

        # acc_visited_cnt[blockIdx.x] += visited_cnt;

        # for (int j = threadIdx.x; j < visited_cnt; j += blockDim.x) {
        #   _visited_table[_visited_list[j]] = -1;
        # }

        # Get sorted neighbors
        template += f'bgt UTHREADID, x0, .SKIP10\n'
        template += f'ld x30, {self.get_arg_offset(self.pq_size)}(x1)\n'
        template += f'lw x21, (x30)\n'  # size

        template += f'TEST.v.x x21\n'

        
        template += f'KERNELBODY:\n'  # KERNELBODY13
        ############### END #####################
        template += f'.SKIP10\n'
        template += f'addi x14, x14, 1\n'
        template += f'TEST.v.x x14\n'
        template += f'j .LOOP0\n'

        template += f'.SKIP0\n'


        return template

    def make_input_map(self):
      return make_memory_map([(self.qdata_addr, self.qdata),
                              (self.data_addr, self.data),
                              (self.entries_addr, self.entries),
                              (self.graph_addr, self.graph),
                              (self.degree_addr, self.degree),
                              (self.nns_addr, self.input_nns),
                              (self.distances_addr, self.input_distances),
                              (self.found_cnt_addr, self.input_found_cnt),
                              (self.visited_table_addr, self.input_visited_table),
                              (self.visited_list_addr, self.input_visited_list),
                              (self.acc_visited_cnt_addr, self.input_acc_visited_cnt),
                              (self.neighbors_addr, self.input_neighbors),
                              (self.global_cand_nodes_addr, self.input_global_cand_nodes),
                              (self.global_cand_distances_addr, self.input_global_cand_distances),
                              (self.graph_info_addr, self.graph_info)])

    def make_output_map(self):
      return make_memory_map([(self.qdata_addr, self.qdata),
                              (self.data_addr, self.data),
                              (self.entries_addr, self.entries),
                              (self.graph_addr, self.graph),
                              (self.degree_addr, self.degree),
                              (self.nns_addr, self.output_nns),
                              (self.distances_addr, self.output_distances),
                              (self.found_cnt_addr, self.output_found_cnt),
                              (self.visited_table_addr, self.output_visited_table),
                              (self.visited_list_addr, self.output_visited_list),
                              (self.acc_visited_cnt_addr, self.output_acc_visited_cnt),
                              (self.neighbors_addr, self.output_neighbors),
                              (self.global_cand_nodes_addr, self.output_global_cand_nodes),
                              (self.global_cand_distances_addr, self.output_global_cand_distances),
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