#!/usr/bin/env bash
# A/B the ATOMesh CI's 1P1D run of one Kimi-K3 case on the ROCm 7.2.4 and
# 10.1 images: prefill, router and aiperf on g08 (rank 0), decode on g07
# (rank 1), each in the container pd_slurm_job.sh: run_container_rank would
# start, then itl.py's per-image mean ITL and the gap.
#
# usage: pd_ab.sh [--dry-run] --issue <n> <case> <out-dir>
#   <case>     cases/<case>.env, the CI's docker-rank-0.env without
#              DOCKER_IMAGE and ATOMESH_RUN_TOKEN
#   <out-dir>  absolute, on storage both nodes mount; one <image-tag>-r<k>/
#              per run, the CI's RUN_DIR. A run whose results exist is kept.
#   --dry-run  print each rank's docker run arguments and env file, start nothing
#
# Differences from the CI: no docker pull (each image's ID is pinned at the
# start), containers named kimi-tpot-<n>-{prefill,decode}, logs under <out-dir>.
set -euo pipefail
shopt -s inherit_errexit

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
CI="${REPO_ROOT}/.github/scripts/atomesh"
IMAGES=(rocm/atom-dev:nightly_202611021458 rocm/atom-dev:ROCm_10.1.0)
RUNS=3
HOSTS=("${PD_AB_PREFILL_HOST:-pit2-p03-g08}" "${PD_AB_DECODE_HOST:-pit2-p03-g07}")
ROLES=(prefill decode)
WATCH_S=30
OTHER_CONTAINER_WAIT_S=3600

die() {
  echo "pd_ab: $*" >&2
  exit 1
}

dry_run=0
issue=""
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --issue) issue="${2:?--issue needs a number}"; shift ;;
    *) die "unknown option $1" ;;
  esac
  shift
done
[[ "${issue}" =~ ^[0-9]+$ && $# -eq 2 ]] \
  || die "usage: pd_ab.sh [--dry-run] --issue <n> <case> <out-dir>"
CASE_ENV="${HERE}/cases/$1.env"
OUT="$2"
[[ -f "${CASE_ENV}" ]] || die "no ${CASE_ENV}"
[[ "${OUT}" == /* ]] || die "<out-dir> must be absolute: both nodes mount it"

case_var() {
  sed -n "s/^$1=//p" "${CASE_ENV}"
}

# Runs stdin as a bash script on host $1 (locally for localhost). The braces
# make bash read the whole script before running it, so no command in it can
# consume the rest of the script as its input. A failed ssh connection (255)
# is retried, so a network blip does not end a two-hour run.
on() {
  local script rc try
  script="$(printf '{\n%s\n}\n' "$(cat)")"
  if [[ "$1" == localhost ]]; then
    bash -s <<<"${script}"
    return
  fi
  for try in 1 2 3; do
    rc=0
    ssh -o BatchMode=yes -o ConnectTimeout=30 "$1" bash -s <<<"${script}" || rc=$?
    ((rc == 255)) || return "${rc}"
    echo "[pd_ab] ssh $1 failed (try ${try})" >&2
    sleep 10
  done
  return 255
}

ci_function() {
  sed -n "/^$1() {/,/^}/p" "${CI}/pd_slurm_job.sh"
}

# The docker run arguments of pd_slurm_job.sh: run_container_rank for rank $1,
# phase "combined", NUL-separated. Runs on the rank's host, which decides the
# group IDs, the ionic library mounts and the socket interface.
rank_args() {
  local rank="$1" env_file="$2" container="$3" flydsl_aot_dir="$4" mesh_binary="$5"
  local video_gid render_gid host_ionic socket_ifname
  local nccl_socket_ifname="${NCCL_SOCKET_IFNAME:-}"
  local mori_socket_ifname="${MORI_SOCKET_IFNAME:-}"
  local node_ip="${IPS[rank]}"
  video_gid="$(getent group video 2>/dev/null | cut -d: -f3 || true)"
  render_gid="$(getent group render 2>/dev/null | cut -d: -f3 || true)"
  host_ionic="$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null || true)"
  if [[ -z "${nccl_socket_ifname}" || -z "${mori_socket_ifname}" ]]; then
    if ! socket_ifname="$(ip -o -4 addr show | awk -v node_ip="${node_ip}" '
      { split($4, address, "/") }
      address[1] == node_ip { sub(/@.*/, "", $2); print $2; exit }
    ')" || [[ -z "${socket_ifname}" ]]; then
      echo "ERROR: no local IPv4 interface for ${node_ip}" >&2
      return 2
    fi
    nccl_socket_ifname="${nccl_socket_ifname:-=${socket_ifname}}"
    mori_socket_ifname="${mori_socket_ifname:-${socket_ifname}}"
  fi
  local args=(
    run --name "${container}"
    --user "$(id -u):$(id -g)"
    --network host --ipc host
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband
    --cap-add=IPC_LOCK --cap-add=NET_ADMIN --cap-add=SYS_NICE
    --ulimit memlock=-1:-1 --ulimit stack=67108864 --ulimit nofile=65536:524288
    --shm-size=128G
    --env-file "${env_file}"
    -e ATOMESH_EXECUTION_PHASE=combined
    -e ATOMESH_SERVICE_PORT_OFFSET=0
    -e ATOMESH_MESH_BINARY="${mesh_binary}"
    -e SLURM_JOB_ID="${JOB_ID}"
    -e SPUR_JOB_ID="${JOB_ID}"
    -e NODE_RANK="${rank}"
    -e NODE0_ADDR="${NODE0_ADDR}"
    -e IPADDRS="${IPADDRS}"
    -e xP="${PREFILL_WORKERS}"
    -e yD="${DECODE_WORKERS}"
    -e PREFILL_TP_SIZE="${PREFILL_TP}"
    -e DECODE_TP_SIZE="${DECODE_TP}"
    -e RUN_DIR="/run_logs/slurm_job-${JOB_ID}"
    -e USER="${CURRENT_USER}"
    -e LOGNAME="${CURRENT_USER}"
    -e HOME="/tmp/atomesh-home-${JOB_ID}-${rank}"
    -e XDG_CACHE_HOME="/tmp/atomesh-cache-${JOB_ID}-${rank}"
    -e TORCHINDUCTOR_CACHE_DIR="/tmp/atomesh-cache-${JOB_ID}-${rank}/torchinductor"
    -e AITER_CACHE_DIR="/tmp/atomesh-cache-${JOB_ID}-${rank}/aiter"
    -e AITER_JIT_DIR="/tmp/atomesh-cache-${JOB_ID}-${rank}/aiter/jit"
    -e FLYDSL_RUNTIME_CACHE_DIR="/tmp/atomesh-cache-${JOB_ID}-${rank}/flydsl"
    -e NCCL_NET_PLUGIN=none
    -e NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    -e MORI_RDMA_DEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    -e MORI_IB_GID_INDEX="${NCCL_IB_GID_INDEX}"
    -e NCCL_CROSS_NIC=0
    -e NCCL_PXN_DISABLE=0
    -e NCCL_NET_DISABLE_INTRA=1
    -e NCCL_IB_TC=104
    -e NCCL_IB_FIFO_TC=192
    -e NCCL_IB_QPS_PER_CONNECTION=1
    -e NCCL_IB_TIMEOUT=22
    -e NCCL_IB_RETRY_CNT=12
    -e NCCL_DEBUG=WARN
    -v "${REPO_ROOT}":/workspace/ATOM:ro
    -v "${RUN_DIR}":/run_logs/slurm_job-"${JOB_ID}"
    -v /mnt:/mnt
    -v /data:/data
  )
  [[ -d /shared_nfs ]] && args+=(-v /shared_nfs:/shared_nfs)
  [[ -d /share_nfs/models ]] && args+=(-v /share_nfs/models:/share_nfs/models)
  [[ -n "${video_gid}" ]] && args+=(--group-add "${video_gid}")
  [[ -n "${render_gid}" ]] && args+=(--group-add "${render_gid}")
  args+=(-e NCCL_SOCKET_IFNAME="${nccl_socket_ifname}")
  args+=(-e MORI_SOCKET_IFNAME="${mori_socket_ifname}")
  [[ -n "${host_ionic}" && -e "${host_ionic}" ]] && args+=(-v "${host_ionic}:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro")
  [[ -e /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so ]] && args+=(-v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro)
  [[ -e /etc/libibverbs.d/ionic.driver ]] && args+=(-v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro)
  [[ -d /it-share ]] && args+=(-v /it-share:/it-share)
  [[ -d /shared_nfs ]] && args+=(-v /shared_nfs:/shared_nfs)
  if [[ -n "${flydsl_aot_dir}" ]]; then
    args+=(
      -v "${flydsl_aot_dir}:/opt/atomesh/flydsl-aot"
      -e ATOMESH_FLYDSL_AOT_DIR=/opt/atomesh/flydsl-aot
    )
  fi
  args+=(
    "${DOCKER_IMAGE}"
    bash -lc "export PATH=/run_logs/slurm_job-${JOB_ID}/bin:\${PATH}; cd /workspace/ATOM && bash .github/scripts/atomesh/pd_server_atom.sh"
  )
  printf '%s\0' "${args[@]}"
}

# Starts rank $1's container in the background on its host, or with --dry-run
# prints its docker command line. The FlyDSL AOT cache and, on rank 0,
# the router build are run_container_rank's own steps; a dry run prints the
# paths they give once done.
start_rank() {
  local rank="$1"
  local container="kimi-tpot-${issue}-${ROLES[rank]}"
  {
    ci_function bounded_docker_rm
    ci_function prepare_flydsl_aot_cache
    declare -f rank_args
    declare -p REPO_ROOT CI RUN_DIR JOB_ID DOCKER_IMAGE IPS NODE0_ADDR IPADDRS \
      CURRENT_USER PREFILL_WORKERS DECODE_WORKERS PREFILL_TP DECODE_TP \
      NCCL_IB_GID_INDEX ATOMESH_BUILD_MESH dry_run
    cat <<EOF
set -euo pipefail
rank=${rank}
container=${container}
env_file="\${RUN_DIR}/docker.env"
EOF
    cat <<'EOF'
mesh_binary=/app/ATOM/atom/mesh/target/release/atomesh
if [[ "${dry_run}" == 1 ]]; then
  image_id="$(docker image inspect --format '{{.Id}}' "${DOCKER_IMAGE}")"
  image_id="${image_id#sha256:}"
  flydsl_aot_dir="${ATOMESH_FLYDSL_AOT_ROOT:-${TMPDIR:-/tmp}/atomesh-flydsl-aot-$(id -u)}/${image_id:0:16}"
  if [[ "${rank}" -eq 0 && "${ATOMESH_BUILD_MESH}" == true ]]; then
    mesh_binary="/run_logs/slurm_job-${JOB_ID}/mesh-build/atomesh"
  fi
else
  bounded_docker_rm "${container}"
  mkdir -p "${RUN_DIR}/rank-${rank}" "${RUN_DIR}/bin"
  cat >"${RUN_DIR}/bin/nvcc" <<'SHIM'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  exec hipcc --version
fi
echo "nvcc shim is only available for --version on ROCm CI" >&2
exit 127
SHIM
  chmod +x "${RUN_DIR}/bin/nvcc"
  flydsl_aot_dir="$(prepare_flydsl_aot_cache)"
  if [[ "${rank}" -eq 0 ]]; then
    unset ATOMESH_MESH_BINARY
    export ATOMESH_BUILD_MESH
    mesh_binary="$(bash "${CI}/setup_mesh.sh" "${REPO_ROOT}" "${RUN_DIR}" "${DOCKER_IMAGE}" "${env_file}" "${JOB_ID}")"
  fi
fi
args_file="$(mktemp)"
rank_args "${rank}" "${env_file}" "${container}" "${flydsl_aot_dir}" "${mesh_binary}" >"${args_file}"
mapfile -d '' args <"${args_file}"
rm -f "${args_file}"
if [[ "${dry_run}" == 1 ]]; then
  printf '# rank %s on %s\ndocker' "${rank}" "$(hostname)"
  printf ' %q' "${args[@]}"
  printf '\n'
  exit 0
fi
# Detached from this shell, so a dropped ssh session does not stop the run.
setsid nohup bash -c 'log="$1" name="$2"; shift 2; docker "$@" 2>&1 | python3 "$0" "$log" 8589934592 "$name"' \
  "${CI}/cap_log.py" "${RUN_DIR}/rank-${rank}/container.log" "${container}" "${args[@]}" \
  </dev/null >/dev/null 2>&1 &
echo "[pd_ab] rank ${rank}: ${container} on $(hostname), flydsl aot ${flydsl_aot_dir:-none}"
EOF
  } | on "${HOSTS[rank]}"
}

# "<image> <id>" for each image on host $1.
image_ids() {
  local image
  for image in "${IMAGES[@]}"; do
    printf '%s %s\n' "${image}" \
      "$(on "$1" <<<"docker image inspect --format '{{.Id}}' ${image}")"
  done
}

remove_own_containers() {
  local rank
  for rank in 0 1; do
    { ci_function bounded_docker_rm; echo "bounded_docker_rm kimi-tpot-${issue}-${ROLES[rank]}"; } \
      | on "${HOSTS[rank]}" || true
  done
}

# Waits until the decode host runs no container but ours, up to
# OTHER_CONTAINER_WAIT_S, then for both hosts' GPUs to be idle.
wait_for_idle_nodes() {
  local deadline=$((SECONDS + OTHER_CONTAINER_WAIT_S)) others rank
  while others="$(other_containers)"; [[ -n "${others}" ]]; do
    ((SECONDS < deadline)) || die "${HOSTS[1]} still runs ${others}"
    echo "[pd_ab] waiting for ${HOSTS[1]} to stop running ${others}"
    sleep 60
  done
  for rank in 0 1; do
    {
      ci_function busiest_gpu_over_limit
      ci_function assert_gpus_idle
      declare -p RUN_DIR
      echo assert_gpus_idle
    } | on "${HOSTS[rank]}" || die "${HOSTS[rank]}: GPUs busy"
  done
}

other_containers() {
  on "${HOSTS[1]}" <<<"docker ps --format '{{.Names}}'" \
    | grep -vx "kimi-tpot-${issue}-decode" | paste -sd, - || true
}

# "<status> <exit code>" of rank $1's container.
container_state() {
  on "${HOSTS[$1]}" <<<"docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' kimi-tpot-${issue}-${ROLES[$1]} 2>/dev/null || echo missing 1"
}

# One run in RUN_DIR: prints done, void (another container on the decode
# host) or failed.
run_once() {
  local others state decode_state waited
  mkdir -p "${RUN_DIR}"
  {
    cat "${CASE_ENV}"
    echo "DOCKER_IMAGE=${DOCKER_IMAGE}"
    echo "ATOMESH_RUN_TOKEN=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"
  } | sort >"${RUN_DIR}/docker.env"
  start_rank 0 >&2
  start_rank 1 >&2
  while sleep "${WATCH_S}"; do
    others="$(other_containers)"
    if [[ -n "${others}" ]]; then
      echo "[pd_ab] ${HOSTS[1]} runs ${others}: run void" >&2
      remove_own_containers
      echo void
      return
    fi
    state="$(container_state 0)"
    decode_state="$(container_state 1)"
    if [[ "${decode_state}" != running* && "${decode_state}" != "exited 0" ]]; then
      echo "[pd_ab] decode container: ${decode_state}" >&2
      remove_own_containers
      echo failed
      return
    fi
    [[ "${state}" == running* ]] && continue
    # Rank 0 is done; decode stops once the router is gone.
    for ((waited = 0; waited < 600; waited += WATCH_S)); do
      [[ "$(container_state 1)" == running* ]] || break
      sleep "${WATCH_S}"
    done
    remove_own_containers
    if [[ "${state}" == "exited 0" ]]; then echo done; else
      echo "[pd_ab] prefill container: ${state}" >&2
      echo failed
    fi
    return
  done
}

# Rank 0 is the prefill host, rank 1 the decode host, as in the CI's job.
CURRENT_USER="$(id -un 2>/dev/null || id -u)"
for var in PREFILL_WORKERS DECODE_WORKERS PREFILL_TP DECODE_TP NCCL_IB_GID_INDEX ATOMESH_BUILD_MESH; do
  printf -v "${var}" '%s' "$(case_var "${var}")"
done
IPS=()
for host in "${HOSTS[@]}"; do
  IPS+=("$(getent ahostsv4 "${host}" | awk 'NR == 1 { print $1 }')")
  [[ -n "${IPS[-1]}" ]] || die "cannot resolve ${host}"
done
NODE0_ADDR="${IPS[0]}"
IPADDRS="$(IFS=,; echo "${IPS[*]}")"

set_run() {
  DOCKER_IMAGE="$1"
  RUN_DIR="${OUT}/${1#*:}-r$2"
  JOB_ID="kimi-tpot-${issue}-${1#*:}-r$2"
}

if [[ "${dry_run}" == 1 ]]; then
  for image in "${IMAGES[@]}"; do
    set_run "${image}" 1
    echo "# ${RUN_DIR}/docker.env"
    {
      cat "${CASE_ENV}"
      echo "DOCKER_IMAGE=${DOCKER_IMAGE}"
      echo "ATOMESH_RUN_TOKEN=<random per run>"
    } | sort
    start_rank 0
    start_rank 1
  done
  exit 0
fi

mkdir -p "${OUT}"
pinned=()
for rank in 0 1; do
  pinned+=("$(image_ids "${HOSTS[rank]}")")
  printf '[pd_ab] %s:\n%s\n' "${HOSTS[rank]}" "${pinned[rank]}"
done
trap remove_own_containers EXIT
trap 'exit 129' HUP INT TERM
for ((k = 1; k <= RUNS; k++)); do
  for image in "${IMAGES[@]}"; do
    set_run "${image}" "${k}"
    if compgen -G "${RUN_DIR}/benchmark_results/pd-atom-*.json" >/dev/null; then
      echo "[pd_ab] ${RUN_DIR}: results exist, kept"
      continue
    fi
    voids=0
    while true; do
      for rank in 0 1; do
        [[ "$(image_ids "${HOSTS[rank]}")" == "${pinned[rank]}" ]] \
          || die "an image tag on ${HOSTS[rank]} now names another image"
      done
      [[ ! -e "${RUN_DIR}" ]] || mv "${RUN_DIR}" "${RUN_DIR}.old-$(date +%s)"
      wait_for_idle_nodes
      echo "[pd_ab] $(date -u +%FT%TZ) start ${RUN_DIR}"
      result="$(run_once)"
      echo "[pd_ab] $(date -u +%FT%TZ) ${result} ${RUN_DIR}"
      case "${result}" in
        done) break ;;
        void)
          mv "${RUN_DIR}" "${RUN_DIR}.void-$(date +%s)"
          ((++voids < 2)) || die "second void in a row; series stopped"
          ;;
        *) die "${RUN_DIR} failed: see its rank-*/container.log" ;;
      esac
    done
  done
done
python3 "${HERE}/itl.py" "${OUT}" --json "${OUT}/itl.json"
