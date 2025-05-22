
# Address
qdata_addr = 0x800000000000
target_data_addr = 0x810000000000
target_nodes_addr = 0x820000000000
graph_addr = 0x830000000000
degree_addr = 0x840000000000
visited_addr = 0x850000000000
visited_list_addr = 0x860000000000
visited_table_addr = 0x870000000000
entries_addr = 0x880000000000
acc_visited_cnt_addr = 0x890000000000
neighbors_addr = 0x8a0000000000
global_cand_nodes_addr = 0x8b0000000000
global_cand_distances_addr = 0x8c0000000000

graph_info_addr = 0x8d0000000000
update_info_addr = 0x8d1000000000
finish_info_addr = 0x8d2000000000
iter_info_addr = 0x8d3000000000

calculated_distance_addr = 0x8d4000000000
entry_distance_addr = 0x8d5000000000
queue_size_addr = 0x8d4000000000

partial_sum_addr = 0x1000000000100000
local_candidate_addr = 0x1000000000200000
tmp_entries_addr = 0x8d7000000000

address_list = [qdata_addr, target_data_addr, target_nodes_addr, graph_addr, degree_addr,
            visited_addr, visited_list_addr, visited_table_addr, entries_addr, acc_visited_cnt_addr,
            neighbors_addr, global_cand_nodes_addr, global_cand_distances_addr,
            graph_info_addr, update_info_addr, finish_info_addr, iter_info_addr,
            calculated_distance_addr, entry_distance_addr, queue_size_addr,
            partial_sum_addr, local_candidate_addr, tmp_entries_addr]

def get_address_list():
    return address_list

def get_arg_offset(addr):
    # print("ADDR: ", hex(addr))
    try:
        # print("index: ", address_list.index(addr) * 8)
        return address_list.index(addr) * 8
    except ValueError:
        raise ValueError(f"Address {addr} not found in input_addrs list.")

def read_graph_file(filename):
    graph = {}
    num_data = num_dims = max_level = max_m = max_m0 = enter_point = None
    qdata = []
    data = []

    with open(filename, 'r') as f:
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

def read_step1_iter_file(filename):
    entry_ids = []
    future_entry_ids = []
    candidate_ids = []
    partial_sums  = []
    calculated_distances  = []
    current_entry_distances  = []

    with open(filename, 'r') as f:
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

def read_step2_iter_file(filename):
    entry_ids = []
    neighbors = []

    with open(filename, 'r') as f:
        mode = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if line.startswith("# Entry id:"):
                    mode = 1
                    continue
                elif line.startswith("# Neighbors (Distance, NodeId, Checked):"):
                    mode = 2
                    continue

                mode = 0
                continue
            
            if mode == 1:
                entry_id = int(line.split(":")[1].strip())
                entry_ids.append(entry_id)
            elif mode == 2:
                neighbor_info = line.split(":")[1].strip()
                distance = int(neighbor_info.split(',')[0].strip())
                nodeid = int(neighbor_info.split(',')[1].strip())
                checked = int(neighbor_info.split(',')[2].strip())
                neighbors.extend([distance, nodeid, checked])
    return entry_ids, neighbors
