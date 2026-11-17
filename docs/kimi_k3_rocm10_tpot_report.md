# Kimi-K3 decode TPOT on ROCm 10.1: results

Results of the investigation designed in
[`kimi_k3_rocm10_tpot.md`](kimi_k3_rocm10_tpot.md), for junyyang-amd/ATOM#62:
the decode gap between `rocm/atom-dev:ROCm_10.1.0` (10.1) and
`rocm/atom-dev:nightly_202611021458` (7.2.4) in the ATOMesh CI's 1P1D
topology.

## Reproduce

From the repository root, on a host that reaches pit2-p03-g07 and g08 over
ssh, with both nodes' GPUs idle and g07 running no other container, once per
case:

```bash
bash scripts/kimi_k3_tpot/pd_ab.sh --issue <n> dspark4-c48 /share_nfs/junyan_tmp_r120/kimi-k3-tpot/<n>/dspark4-c48
bash scripts/kimi_k3_tpot/pd_ab.sh --issue <n> dspark3-c8 /share_nfs/junyan_tmp_r120/kimi-k3-tpot/<n>/dspark3-c8
```

Each runs 3 launches per image, alternating 7.2.4 and 10.1, prefill, router
and aiperf on g08, decode on g07, and ends with `itl.py`'s per-image mean
ITL (mean and min-max over the runs) and the gap. Add `--dry-run` to print
each rank's `docker run` command and env file without starting anything.
The settings are the CI's: `scripts/kimi_k3_tpot/cases/<case>.env` is the
`docker-rank-0.env` of CI Slurm 11899 (dspark4 c48) and 12189 (dspark3 c8)
without `DOCKER_IMAGE` and `ATOMESH_RUN_TOKEN`.

Per-run figures: `kimi_k3_rocm10_tpot/<case>.itl.json`.

### Differences from the CI

- No `docker pull`: each image's ID on each node is recorded at the start
  of a series, and the series stops if a tag later names another image.
- Containers are named `kimi-tpot-<n>-prefill` and `kimi-tpot-<n>-decode`;
  the job ID in container paths and `SLURM_JOB_ID` is
  `kimi-tpot-<n>-<image-tag>-r<k>`, and each run's logs and results are
  under `<out-dir>/<image-tag>-r<k>/`.
- Prefill runs on g08 and decode on g07; CI 11899 and 24916 placed decode
  on g08.
- No other container is stopped. A container other than our own on g07
  during a run voids the run, which is repeated; a second void in a row
  stops the series. A run starts only when no GPU on either node holds more
  than 8 GB, as `pd_slurm_job.sh: assert_gpus_idle` checks.
