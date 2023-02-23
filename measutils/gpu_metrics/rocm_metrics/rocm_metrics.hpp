
#include "hip/hip_runtime.h"
#include <hsa/hsa.h>
#include <iostream>
#include <rocprofiler.h>
#include <unistd.h>
#include <vector>

#include <dlfcn.h>
#include <hsa/hsa.h>
#include <string.h>
#include <unistd.h>

#include <atomic>
#include <iostream>
#include <sstream>
#include <vector>

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

// Dispatch callbacks and context handlers synchronization
pthread_mutex_t mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
// Tool is unloaded
volatile bool is_loaded = false;
// Profiling features
// rocprofiler_feature_t* features = NULL;
// unsigned feature_count = 0;

// Error handler
void fatal(const std::string msg) {
  fflush(stdout);
  fprintf(stderr, "%s\n\n", msg.c_str());
  fflush(stderr);
  abort();
}

// Check returned HSA API status
void check_status(hsa_status_t status) {
  if (status != HSA_STATUS_SUCCESS) {
    const char *error_string = NULL;
    rocprofiler_error_string(&error_string);
    fprintf(stderr, "ERROR: %s\n", error_string);
    abort();
  }
}

// Context stored entry type
struct context_entry_t {
  bool valid;
  hsa_agent_t agent;
  rocprofiler_group_t group;
  rocprofiler_callback_data_t data;
};

// Context callback arg
struct callbacks_arg_t {
  rocprofiler_pool_t **pools;
};

// Handler callback arg
struct handler_arg_t {
  rocprofiler_feature_t *features;
  unsigned feature_count;
};

hsa_agent_t agent_info_arr[16];
unsigned agent_info_arr_len;

static hsa_status_t _count_devices(hsa_agent_t agent, void *data) {
  unsigned *count = (unsigned *)data;
  hsa_device_type_t type;
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  assert(status == HSA_STATUS_SUCCESS);
  if (type == HSA_DEVICE_TYPE_GPU) {
    agent_info_arr[(*count)++] = agent;
  }
  return status;
}

static unsigned _get_device_count(void) {
  unsigned count = 0;
  hsa_status_t status = hsa_iterate_agents(&_count_devices, &count);
  assert(status == HSA_STATUS_SUCCESS);
  return count;
}

static hsa_agent_t _get_agent(unsigned gpu_id) {
  return agent_info_arr[gpu_id];
}

// Dump stored context entry
void dump_context_entry(context_entry_t *entry, rocprofiler_feature_t *features,
                        unsigned feature_count) {
  volatile std::atomic<bool> *valid =
      reinterpret_cast<std::atomic<bool> *>(&entry->valid);
  while (valid->load() == false)
    sched_yield();

  const std::string kernel_name = entry->data.kernel_name;
  const rocprofiler_dispatch_record_t *record = entry->data.record;

  fflush(stdout);
  fprintf(stdout,
          "kernel symbol(0x%lx) name(\"%s\") tid(%u) queue-id(%u) gpu-id(%u) ",
          entry->data.kernel_object, kernel_name.c_str(), entry->data.thread_id,
          entry->data.queue_id, 0);
  if (record)
    fprintf(stdout, "time(%lu,%lu,%lu,%lu)", record->dispatch, record->begin,
            record->end, record->complete);
  fprintf(stdout, "\n");
  fflush(stdout);

  rocprofiler_group_t &group = entry->group;
  if (group.context == NULL) {
    fatal("context is NULL\n");
  }
  if (feature_count > 0) {
    hsa_status_t status = rocprofiler_group_get_data(&group);
    check_status(status);
    status = rocprofiler_get_metrics(group.context);
    check_status(status);
  }

  for (unsigned i = 0; i < feature_count; ++i) {
    const rocprofiler_feature_t *p = &features[i];
    fprintf(stdout, ">  %s ", p->name);
    switch (p->data.kind) {
    // Output metrics results
    case ROCPROFILER_DATA_KIND_INT64:
      fprintf(stdout, "= (%lu)\n", p->data.result_int64);
      break;
    case ROCPROFILER_DATA_KIND_DOUBLE:
      fprintf(stdout, "= (%lf)\n", p->data.result_double);
      break;
    default:
      fprintf(stderr, "Undefined data kind(%u)\n", p->data.kind);
      abort();
    }
  }
}

// Profiling completion handler
// Dump and delete the context entry
// Return true if the context was dumped successfully
bool context_handler(const rocprofiler_pool_entry_t *entry, void *arg) {
  std::cout << "context_handler\n";
  // Context entry
  context_entry_t *ctx_entry =
      reinterpret_cast<context_entry_t *>(entry->payload);
  handler_arg_t *handler_arg = reinterpret_cast<handler_arg_t *>(arg);

  dump_context_entry(ctx_entry, handler_arg->features,
                     handler_arg->feature_count);

  return false;
}

// Kernel disoatch callback
hsa_status_t dispatch_callback(const rocprofiler_callback_data_t *callback_data,
                               void *arg, rocprofiler_group_t *group) {
  std::cout << "dispatch callback\n";
  // Passed tool data
  hsa_agent_t agent = _get_agent(0);
  // HSA status
  hsa_status_t status = HSA_STATUS_ERROR;

  // Open profiling context
  const unsigned gpu_id = 0;
  callbacks_arg_t *callbacks_arg = reinterpret_cast<callbacks_arg_t *>(arg);
  rocprofiler_pool_t *pool = callbacks_arg->pools[gpu_id];
  rocprofiler_pool_entry_t pool_entry{};
  status = rocprofiler_pool_fetch(pool, &pool_entry);
  check_status(status);

  // Profiling context entry
  rocprofiler_t *context = pool_entry.context;
  context_entry_t *entry =
      reinterpret_cast<context_entry_t *>(pool_entry.payload);
  // Get group[0]
  status = rocprofiler_get_group(context, 0, group);
  check_status(status);

  // Fill profiling context entry
  entry->agent = agent;
  entry->group = *group;
  entry->data = *callback_data;
  entry->data.kernel_name = strdup(callback_data->kernel_name);
  reinterpret_cast<std::atomic<bool> *>(&entry->valid)->store(true);

  return HSA_STATUS_SUCCESS;
}

unsigned metrics_input(rocprofiler_feature_t **ret) {
  // Profiling feature objects
  const unsigned feature_count = 10;
  rocprofiler_feature_t *features = new rocprofiler_feature_t[feature_count];
  memset(features, 0, feature_count * sizeof(rocprofiler_feature_t));

  // PMC events
  features[0].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[0].name = "GRBM_COUNT";
  features[1].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[1].name = "GRBM_GUI_ACTIVE";
  features[2].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[2].name = "GPUBusy";
  features[3].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[3].name = "SQ_WAVES";
  features[4].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[4].name = "SQ_INSTS_VALU";
  features[5].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[5].name = "VALUInsts";
  features[6].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[6].name = "TCC_HIT_sum";
  features[7].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[7].name = "TCC_MISS_sum";
  features[8].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[8].name = "WRITE_SIZE";
  features[9].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[9].name = "FETCH_SIZE";

  *ret = features;
  return feature_count;
}

void initMeasureMetric() {

  hsa_init();
  unsigned gpu_count = _get_device_count();
  agent_info_arr_len = gpu_count;

  for (unsigned gpu_id = 0; gpu_id < gpu_count; ++gpu_id) {
    hsa_agent_t agent = _get_agent(gpu_id);
    std::cout << "Agent " << gpu_id << "\n";
  }

  hsa_agent_t agent = _get_agent(0);
  // Available GPU agents

  // Getting profiling features
  rocprofiler_feature_t *features = NULL;
  unsigned feature_count = metrics_input(&features);

  // Handler arg
  handler_arg_t *handler_arg = new handler_arg_t{};
  handler_arg->features = features;
  handler_arg->feature_count = 2;

  // Context properties
  rocprofiler_pool_properties_t properties{};
  properties.num_entries = 100;
  properties.payload_bytes = sizeof(context_entry_t);
  properties.handler = context_handler;
  properties.handler_arg = handler_arg;

  // Adding dispatch observer
  callbacks_arg_t *callbacks_arg = new callbacks_arg_t{};
  callbacks_arg->pools = new rocprofiler_pool_t *[gpu_count];

  rocprofiler_pool_t *pool = NULL;
  hsa_status_t status =
      rocprofiler_pool_open(agent, features, 2, &pool,
                            0 /*ROCPROFILER_MODE_SINGLEGROUP*/, &properties);
  check_status(status);
  rocprofiler_pool_t *pool2 = NULL;
  status =
      rocprofiler_pool_open(agent, features + 2, 2, &pool2,
                            0 /*ROCPROFILER_MODE_SINGLEGROUP*/, &properties);
  check_status(status);
  callbacks_arg->pools[0] = pool2;

  rocprofiler_queue_callbacks_t callbacks_ptrs{};
  callbacks_ptrs.dispatch = dispatch_callback;
  std::cout << "set queue callbacks\n";
  rocprofiler_set_queue_callbacks(callbacks_ptrs, callbacks_arg);
}

void cleanup() {
  // Unregister dispatch callback
  rocprofiler_remove_queue_callbacks();
}


void measureBandwidthStart() {
  static bool initialized = false;
  if (!initialized) {
    initMeasureMetric();
    initialized = true;
  }
  rocprofiler_start_queue_callbacks();
}

std::vector<double> measureMetricStop() {

  rocprofiler_stop_queue_callbacks();
  return std::vector<double>(2);
}
