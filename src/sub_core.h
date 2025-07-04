#ifndef FUNCSIM_SUB_CORE_H_
#define FUNCSIM_SUB_CORE_H_

#include "common.h"
#include "memory_map.h"
#include "m2ndp_config.h"
#include "register_unit.h"
#ifdef TIMING_SIMULATION
#include "delayqueue.h"
#include "execution_unit.h"
#include "instruction_buffer.h"
#include "instruction_queue.h"
#include "mem_fetch.h"
#include "ndp_stats.h"
#include "tlb.h"
#include "uthread_generator.h"
#endif
namespace NDPSim {

class SubCore {
 public:
  SubCore() {}
  SubCore(M2NDPConfig* config, MemoryMap* memory_map, int id, int sub_core_id);
  void SubCoreInitialize(int num_kernel_bodies, int uthread_sz);
  void LoadContext(RequestInfo* info, int uthread_id);
  void StoreContext(RequestInfo* info, int uthread_id);
  void InitializeRegister(RequestInfo* info, int uthread_id);
  void ExecuteInitializer(MemoryMap* spad_map, RequestInfo* info);
  int ExecuteKernelBody(MemoryMap* spad_map, RequestInfo* info,
                         int kenrel_body_id, int uthread_sz, int uthread_id);
  void ExecuteFinalizer(MemoryMap* spad_map, RequestInfo* info);
  void set_ndp_kernel(NdpKernel* ndp_kernel) {m_ndp_kernel = ndp_kernel;}

#ifdef TIMING_SIMULATION
  SubCore(M2NDPConfig* config, MemoryMap* memory_map, NdpStats* stats, int id, int sub_core_id,
          fifo_pipeline<InstColumn>* inst_column_q, fifo_pipeline<mem_fetch>* to_icache, fifo_pipeline<mem_fetch>* from_icache,
          fifo_pipeline<std::pair<NdpInstruction, Context>> *to_ldst_unit,
          fifo_pipeline<std::pair<NdpInstruction, Context>> *to_spad_unit,
          fifo_pipeline<std::pair<NdpInstruction, Context>> *to_v_ldst_unit,
          fifo_pipeline<std::pair<NdpInstruction, Context>> *to_v_spad_unit,
          std::queue<Context> *finished_contexts, std::queue<Context> *finished_uthreads);

  void cycle();
  void execute_instruction();
  void l0_inst_cache_cycle();
  void instruction_queue_allocate();
  void instruction_register_allocate();

  void set_rf_ready(int64_t reg) {m_register_unit->SetReady((int)reg);}
  bool is_vreg(int64_t reg) {return m_register_unit->IsVreg((int)reg);}
  bool is_xreg(int64_t reg) {return m_register_unit->IsXreg((int)reg);}
  bool is_freg(int64_t reg) {return m_register_unit->IsFreg((int)reg);}
  void free_rf(int packet_id) {m_register_unit->FreeRegs(packet_id);}
  void free_inst_column(int col_id) {m_instruction_queue->finish_inst_column(col_id);}
  bool is_inst_queue_empty() { return m_instruction_queue->all_empty(); }

  bool is_active();
  RegisterStats get_register_stats();
  CacheStats get_l0_icache_stats();
  void print_sub_core_stats();
#endif

 private:
  M2NDPConfig* m_config;
  int m_num_ndp;
  int m_id;
  int m_sub_core_id;
  int m_num_xreg;
  int m_num_freg;
  int m_num_vreg;
  NdpKernel* m_ndp_kernel;
  MemoryMap* m_memory_map;
  RegisterUnit* m_register_unit;

  std::vector<std::vector<std::deque<NdpInstruction>>> insts_list;
  std::vector<int> branch_idx;

#ifdef TIMING_SIMULATION
  InstructionQueue* m_instruction_queue;
  ExecutionUnit* m_execution_unit;
  CacheConfig m_icache_config;
  Cache* m_l0_icache;
  // request queues
  fifo_pipeline<InstColumn>* m_inst_column_q;
  fifo_pipeline<mem_fetch>* m_to_icache;
  fifo_pipeline<mem_fetch>* m_from_icache;
  fifo_pipeline<std::pair<NdpInstruction, Context>> *m_to_ldst_unit;
  fifo_pipeline<std::pair<NdpInstruction, Context>> *m_to_spad_unit;
  fifo_pipeline<std::pair<NdpInstruction, Context>> *m_to_v_ldst_unit;
  fifo_pipeline<std::pair<NdpInstruction, Context>> *m_to_v_spad_unit;

  std::queue<Context> *m_finished_contexts;
  std::queue<Context> *m_finished_uthreads;
  NdpStats* m_stats;
  int m_active_queque_count;
  uint64_t m_deadlock_count;
  uint64_t m_issue_fail_count;
#endif
  void ExecuteInsts(MemoryMap* spad_map, std::deque<NdpInstruction> insts,
                    RequestInfo* info);
  int ExecuteInsts_Array(MemoryMap* spad_map, std::vector<std::deque<NdpInstruction>> insts_list,
                          RequestInfo* info,
                          const std::vector<std::map<int, int>>& loop_map,
                          int kernel_body_id, int uthread_sz, int uthread_id = 0);
};
}  // namespace NDPSim
#endif  // FUNCSIM_SUB_CORE_H_
