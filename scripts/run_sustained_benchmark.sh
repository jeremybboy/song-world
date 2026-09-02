#!/bin/zsh
set -euo pipefail

if (( $# != 5 )); then
  print -u2 "usage: $0 PHASE0_BINARY MODEL_MLXFN RESOURCES_DIR REPORT_JSON GPU_CSV"
  exit 64
fi

phase0_binary="$1"
model_mlxfn="$2"
resources_dir="$3"
report_json="$4"
gpu_csv="$5"

mkdir -p "${report_json:h}" "${gpu_csv:h}"
print 'sample,device_utilization_percent,renderer_utilization_percent,in_use_system_memory_bytes' > "$gpu_csv"

env TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 \
  "$phase0_binary" \
  --engine mrt2 \
  --model "$model_mlxfn" \
  --resources "$resources_dir" \
  --duration 180 \
  --device-buffer 512 \
  --ring-buffer 4096 \
  --control-period 8 \
  --tempo-ratio 1.25 \
  --report "$report_json" &
harness_pid=$!

sample_index=0
while kill -0 "$harness_pid" 2>/dev/null; do
  device_utilization="$(ioreg -r -c AGXAccelerator -d 1 -a | plutil -extract '0.PerformanceStatistics.Device Utilization %' raw -o - -)"
  renderer_utilization="$(ioreg -r -c AGXAccelerator -d 1 -a | plutil -extract '0.PerformanceStatistics.Renderer Utilization %' raw -o - -)"
  in_use_memory="$(ioreg -r -c AGXAccelerator -d 1 -a | plutil -extract '0.PerformanceStatistics.In use system memory' raw -o - -)"
  print "$sample_index,$device_utilization,$renderer_utilization,$in_use_memory" >> "$gpu_csv"
  (( sample_index += 1 ))
  sleep 1
done

set +e
wait "$harness_pid"
harness_status=$?
set -e
exit "$harness_status"
