# Experimental Maya on RX 7900 XT / XTX, R9700 / RX 9070 and Strix Halo / Gorgon Halo

This branch adds a Linux HIP build and installer path for Maya's GLM-5.3-Flash
engine on `gfx1100`, `gfx1201` and `gfx1151` (Strix Halo, Gorgon Halo). It uses one GPU, or two supported GPUs that split the
layers (see [Two GPUs](#two-gpus)), and serves text and, on Linux, pictures: the setup builds the
[CPU image encoder](#cpu-image-encoder-on-linux) and fetches the vision files by default (`--no-vision`: text only).
Windows: see [Windows](#windows).

Use a system ROCm 7 installation with its HIP compiler and hipBLAS, Python
3.10+, CMake 3.24+, and a C++20 compiler. `ROCM_PATH` selects an installation
outside `/opt/rocm`. Everything Maya downloads goes into the project or its
model-data folder; setup does not install system packages.

```sh
./maya.sh --backend hip --gpu 0 --check
./maya.sh --backend hip --gpu 0 --context 8192 --no-vision --yes \
  --download-model --no-start --port 8099 --env MAYA_GLM_RAM_GB=60
./maya.sh --backend hip --port 8099
```

The first setup downloads about 96.5 GB of weights and verifies the SHA-256
hash of each shard. `--gguf-dir DIR` uses existing files instead. GPU numbers
are the kernel KFD topology order shown by `--check`; on the test machine the
7900 XT is GPU 0, an R9700 is GPU 1, and the integrated GPU is GPU 2. This
installer accepts `gfx1100`, `gfx1201` and `gfx1151` and compiles the binary for the
architectures of the chosen cards only (`gfx1100;gfx1201` for that 7900 XT + R9700).

RDNA 2 (`gfx1030`: RX 6800 / 6800 XT / 6900 XT / 6950 XT, Radeon PRO V620 / W6800) is accepted too, as experimental:
it has been run on a Radeon PRO V620 with ROCm 7 - please report how it goes on other RDNA 2 cards in an issue. These
cards have no matrix cores (WMMA), so the engine uses its portable kernels there (the prompt attention on the F32
kernel), and a build for RDNA 2 cards alone needs no rocWMMA headers (ROCm 7 no longer ships them). The ROCm
installation needs its library kernels (hipBLAS) for `gfx1030`. Four V620s on a two-socket host, in a container:
[RDNA 2: four Radeon PRO V620s](#rdna-2-four-radeon-pro-v620s).

Configs are named `maya-<quant>-hip.json` and
select the AMD device through `HIP_VISIBLE_DEVICES`.

`--bench` and `--report` work on HIP: `./maya.sh --backend hip --bench` uses
`build-hip/maya` and the config's GPU order and environment; run it with Maya
stopped. `./maya.sh --backend hip --report` includes AMD GPU details from
`rocm-smi` or `amd-smi` when available (KFD topology otherwise), the ROCm path
and version, and the engine's speed lines. Add `--config /path/to/maya-hip.json`
to either command to select a particular installed config.

The discrete-GPU HIP config starts with an 8K context when requested above, 3 GiB of GPU
headroom, and 16 GiB of system-RAM headroom. The engine sizes the prompt chunk
from its prompt-memory budget. The example additionally caps the pinned expert
cache at 60 GiB. Change those settings with `--env KEY=VALUE` during setup.
Available RAM, rather than installed RAM, determines how much can be cached.
The engine reduces its RAM-tier allocation if ROCm cannot pin the requested
amount.

## CPU image encoder on Linux

The setup reads pictures by default on Linux. It builds `maya-vision`, llama.cpp's image library at the pinned commit
with GLM-5.3-Flash's vision tower added (`tools/vision/glm5next_patch.py`, the same patch the CUDA encoder uses), on
ggml's CPU backend, into `build-vision-cpu/` - a folder of its own, so a CUDA encoder in `build-vision/` stays as it
is. It fetches the vision files (1.14 GB, sha256-checked) into the model's `vision/` folder, shown and asked first as
the model is (`--yes` and `--download-model` take them). The config gets a `vision` member with `"gpu": false`, and the
server keeps the encoder running beside the HIP engine, in system RAM. `--no-vision` sets up text only; Windows stays
text only for now. A config set up before reads text only until it is set up again (`--setup`; the model is not
downloaded again).

The encoder takes 4 CPU threads, llama.cpp's default; `"threads": N` in the config's `vision` member sets another
count. There is no AMD GPU image encoder: none has been validated. Check that `/health` reports `"images": true`, then
send an actual picture through `/v1/chat/completions`.

Verified on Linux, Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`), 128 GB unified
RAM, ROCm 7.14.1, stock Maya v1.1.5 (`5403c222`), Maya-S24, and pinned llama.cpp
`3cf03257`: a real image request correctly identified a green circle on a white
background. Two checks took 3.38 and 3.12 seconds, including generation.
These are small functional checks, not broad vision-quality or performance
claims. CPU image encoding uses system RAM and adds CPU work; other AMD devices,
Windows, large-image limits, and multiple-image workloads were not tested here.

## Two GPUs

`--gpus 0,1` splits the layers across two supported cards, as on NVIDIA: each card caches the experts of its own
half, and the second half also runs the model's MTP block, which drafts the next token so both cards stay busy. Setup
puts the card with more VRAM first (it takes the bigger first half), leaves the split point to the engine, and writes
one hipBLASLt table per architecture (`MAYA_HIPBLASLT_TUNING` takes a `:`-separated list; each card uses the table
for its own architecture). A discrete GPU paired with a unified-memory APU puts the discrete GPU first and loads the
parts sequentially (`MAYA_GLM_PARALLEL_LOAD=0`), so the APU sizes its pool after the first part pins its RAM tier.
The two cards may be different architectures. The split passes one small host-memory hop
per token, so no peer-to-peer access between the cards is needed.

Measured on an AI PRO R9700 (32 GB, first) + RX 7900 XT (20 GB), 192 GB RAM, Maya-S, 8K context, 90 GB RAM tier,
v1.0.11, greedy, 256-token answers (median of 10 requests):

| | one RX 7900 XT | R9700 + RX 7900 XT |
|---|---|---|
| answer speed | ~15 tok/s | ~33 tok/s (drafts accepted ~76%) |
| prompt speed, 4K-token prompt | ~414 tok/s | ~490 tok/s |

39 back-to-back requests (1.8-4K-token prompts) ran without an error. The default split (the midpoint + 2) measured
best: moving two or four more layers to the first card did not speed up answers.

### Experimental expert helper

`MAYA_GLM_EXPERT_HELPER=1` is an opt-in alternative to a layer split: visible GPU 0 keeps every trunk layer,
attention, state and MTP; visible GPU 1 owns a complementary expert cache. A unified-memory helper reserves slots
in the existing portable pinned RAM tier (no second weight copy); a discrete helper uses its own VRAM, automatically
sized by aligned expert bytes with 512 MiB left free. Primary admissions, prefetch and background promotions exclude
helper-owned experts. At idle boundaries, colder helper experts can be replaced by hotter unowned experts of the same
layer.
The helper quantizes activations and reduces weighted expert outputs on its GPU before returning one vector per token.
CPU quantization runs only for actual CPU fallback; uncovered experts retain the existing CPU/PCIe/disk paths.
Batched prefill retains its primary-GPU staging and CPU pool, and may read helper-owned UMA slots without evicting them.
All of this is part of the single helper switch; there is no separate optimization flag.
Native GGUF experts only; EXL3 is not supported.

For physical device 0 = 8060S and 1 = R9700, run the engine with `HIP_VISIBLE_DEVICES=1,0`,
`MAYA_GLM_EXPERT_HELPER=1` and `MAYA_GLM_SPLIT=0`. The indices are process-local; the system order is unchanged.
Keep the normal pack, context, headroom and MTP settings. Unset `MAYA_GLM_EXPERT_HELPER` to restore the existing
behavior. This experimental path does not claim general multi-GPU support or a speedup; measure it on the target
hardware.

#### The helper beside a layer split (two cards and the APU)

With `MAYA_GLM_SPLIT=auto` (or pinned boundaries) the helper's device is left out of the split and every part of the
split opens its own helper there over its own layers: a part's misses go to the APU, which computes them inside that
part's pinned RAM tier at the APU's memory speed instead of the part's CPU lane. For physical device 0 = 8060S and
1, 2 = two discrete cards, run the engine with `HIP_VISIBLE_DEVICES=1,2,0`, `MAYA_GLM_EXPERT_HELPER=2` and
`MAYA_GLM_SPLIT=auto` (the split's devices are then 0 and 1; `MAYA_GLM_DEVS=0,1` says so explicitly). The helper's
device may not be one of the split's. The engine logs one `glm helper:` line per part. Measured on this layout: see the
speed table; the CPU lane still takes the misses the helper does not own.

## RDNA 2: four Radeon PRO V620s

Four V620s (32 GB each, passively cooled) split the layers like any four cards: `--gpus 0,1,2,3`. The setup writes
the discrete-card defaults (3 GiB of VRAM headroom a card, a 1024-token prompt sub-batch and a 4096 MiB prompt
budget); no hipBLASLt table exists for `gfx1030`, so the prompt's projections stay on plain hipBLAS. The split passes
one small host-memory hop per token, so the cards need no peer-to-peer path between them. Not measured on the cards yet.

**What is checked without the cards.** Maya at v1.1.28 compiles for `gfx1030` alone with ROCm 7.14.0 (TheRock's
`rocm[libraries,devel,device-gfx1030]` wheels, clang 23): both engines and all 170 HIP targets of the build, the GPU
checks of [Validation](#validation) among them; and with ROCm 7.2.0 (AMD's packages, clang 22): both engines and the
GPU checks of [Validation](#validation). Also a static check that every kernel that needs WMMA or `sudot4`
keeps a host-side guard RDNA 2 does not pass (the prompt attention's F32 path, the fused prompt experts on gfx1100 /
gfx1151 only, DeepSeek's `tc_ok()`). On the cards, run the checks of [Validation](#validation) on each one first
(`HIP_VISIBLE_DEVICES=0`, then 1, 2, 3).

**Two sockets.** With the cards on one socket's PCIe lanes, keep Maya's threads on that socket's CPUs
(`numactl --cpunodebind=<the cards' node>`; `cat /sys/bus/pci/devices/<card>/numa_node` says which). The engine
reads the CPUs it is given: a split of four cards runs one CPU lane for all of them, sized from the physical cores
of that node (by its sizing rule, on two 8-core Xeon Gold 6144s: 5 threads confined to one node, 12 unconfined; not
measured), and its RAM tier is still spread page by page over both sockets' memory (`MAYA_GLM_NUMA=0` keeps it from
doing that). Prefer `--preferred=<node>` to `--membind=<node>` for the memory: a strict bind caps the page cache
(DeepSeek reads its experts in place) at one node and can end the engine when that node fills.

**Passive cooling.** Server cards in a workstation reach their junction limit long before anything slows them down:
`--env MAYA_THERMAL_ABORT_C=90` stops serving at 90 °C on any card's edge, junction or memory sensor, with exit
status 90 ([Settings](SETTINGS.md)).

**In a container.** The image builds both engines and the CPU image encoder without the cards, and keeps the configs
outside itself; the setup then fetches the vision files with the model:

```sh
./maya.sh --backend hip --build-only --hip-archs gfx1030 --yes      # in the image's build: no GPU needed
MAYA_CONFIG_DIR=/maya-state XDG_CONFIG_HOME=/maya-state/config MAYA_UPDATE_CHECK=0 \
  ./maya.sh --backend hip --gpus 0,1,2,3 --models-dir /models --setup --yes --download-model --no-start
```

The run needs `/dev/kfd` and `/dev/dri`, the user's `render` and `video` groups, an unlimited `memlock` (the RAM tier
is pinned) and `/sys` (KFD's topology, the sensors). A recipe for this host - the image pinned by commit, a serve
script with a thermal guard of its own, NUMA placement and one engine on the cards at a time - is in
[jmacpolin/v620-Experimentation](https://github.com/jmacpolin/v620-Experimentation) (`06-MAYA.md`).

## Strix Halo

Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`) uses unified memory. Use the same
HIP setup command above, selecting its GPU number and omitting the
`--env MAYA_GLM_RAM_GB=60` override. Setup detects the APU even if sysfs
reports only a small VRAM carve-out. It writes a 1 GiB reserve, 16 GiB of
system headroom, `MAYA_GLM_PREFILL_SUB=1024`, and a 6144 MiB prompt budget
when system RAM is at least 96 GiB (4096 MiB otherwise), plus the existing
`gfx1151` hipBLASLt table. Use `--gpu N` for the APU alone, or `--gpus D,N` to pair it with a supported discrete GPU
(see [Two GPUs](#two-gpus)). It leaves `POOL_GB` and `RAM_GB` unset.

For integrated HIP devices the engine sizes the pool from `/proc/meminfo`'s
`MemAvailable` **after** dense weights, KV and prompt staging are allocated:
`min(expert bytes, max(0, MemAvailable - RAM_HEADROOM_GB - RESERVE_MB))`.
It does not treat HIP's reported GTT capacity (112 GiB on the 128 GB box) as
free physical memory. For example, 110 GiB available leaves 93 GiB after the
default headroom/reserve, capped at Maya-S's ~80.17 GiB expert pool. When every
expert fits, warm-up fills every pool slot without reserving idle spares;
the RAM tier keeps only its small per-class staging floor, and
the prompt disk landing ring defaults to 12 slots. Prompt buffers borrow the
pool's tail; they do not require a second 6 GiB allocation. Explicit `--env`
settings still override defaults; pool/VRAM caps still limit the pool.
Discrete HIP GPUs and CUDA keep their existing sizing behavior.

Measured on a 128 GB Strix Halo box with ROCm 7.2.2 and manually tuned settings:
~210 tok/s for 3-4K-token prompts and ~17.8 tok/s answers, with every expert
in the GPU pool (288 slots/layer, 12,096 total, ~99.9% hits). These measurements
precede the automatic sizing change; verify its startup log and repeated
prompt/answer rounds on the real box. See [tracking issue #6](https://github.com/mw00/project-maya/issues/6).

**Answers and prompts (v1.1.10).** Where every expert sits in the GPU pool, the APUs answer faster with the same
text (`MAYA_GLM_MTP_DECODE`, on by default on `gfx115x` only). On the Gorgon Halo of [Windows](#windows) (Radeon
8065S, Maya-S, 32K context), 256-token answers run at 28.4-29.6 tokens/s to short prompts and at 23.2-23.5 after
3K- and 8K-token prompts, against 18.4-19.7 without it; 3K- and 8K-token prompts read at 271 and 278 tokens/s
(the fused prompt experts and the `wmma2` attention, both on by default on `gfx1151`).

## Windows

`START-MAYA.bat --backend hip` sets Maya up natively on Windows 10/11 for the same GPUs, Strix Halo and Gorgon Halo
included (Ryzen AI Max 300 / 400, Radeon 8050S / 8060S / 8065S, all `gfx1151`). It compiles the engine with ROCm's
clang in Visual Studio's environment, with Ninja, as `tools\hip\build_maya_windows.bat` does (#54). The download, the
pack and the dashboard are the same as on an NVIDIA PC.

1. Install once: a current AMD driver (AMD Software: Adrenalin Edition),
   [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 2022 or 2026 with
   "Desktop development with C++", 64-bit Python 3.12 (`winget install -e --id Python.Python.3.12 --scope user`),
   Git, and [AMD's HIP SDK for Windows](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) (7.2 is
   measured below; without it the setup offers AMD's ROCm SDK wheels).
2. On an APU, give the GPU most of the memory: Variable Graphics Memory in AMD Software (Performance > Tuning), or
   the iGPU memory size in the BIOS - 160 GB of 192 GB on the PC below, 96 GB on a 128 GB one. Restart.
3. Then:

   ```bat
   START-MAYA.bat --backend hip --gpu 0 --check
   START-MAYA.bat --backend hip --gpu 0 --setup --model Maya-L
   ```

The setup takes the ROCm it finds first: `ROCM_PATH`, TheRock's `rocm-sdk` (`ROCM_VENV`, Maya's `.venv` or PATH), AMD's
HIP SDK (`HIP_PATH`, else the newest in `C:\Program Files\AMD\ROCm`). With none it offers AMD's ROCm SDK wheels in
Maya's `.venv` (pip, from `repo.amd.com/rocm/whl-multi-arch`: `rocm[libraries,devel,device-<arch>]==7.14.1`, the first
with Gorgon Halo); `--check` prints that command instead. GPU numbers are hipInfo's (HIP's own order); without
hipInfo, Windows' AMD display adapters in the registry's order.

The build copies the ROCm SDK's HIP runtime (`amdhip64_7.dll`, `amd_comgr*.dll`, `rocm_kpack.dll`) next to
`build-hip\maya.exe`: Windows would otherwise load the driver's own from System32 before the SDK's.

**Memory on a Windows APU.** Windows gives the APU's GPU a fixed carve-out and does not count it as RAM: the 192 GB PC
below, with 160 GB of Variable Graphics Memory, shows 32 GB of RAM, and HIP reports 171.9 GB (the carve-out and three
quarters of Windows' shared half of its RAM). The engine therefore sizes it like a discrete card - the expert pool from
the GPU memory HIP reports free, the RAM tier from the free RAM and commit, the rest read from the SSD - and not from
`MemAvailable` as on Linux above. Windows' HIP runtime allocates at most 64 GiB plus the RAM Windows sees in one
piece (95.7 GiB there), so a larger pool goes into several allocations, each holding whole layers. Setup writes a
3 GiB reserve (the desktop runs on the same GPU), `MAYA_GLM_PREFILL_SUB=1024`, and a 6144 MiB prompt budget when RAM
and GPU memory together are at least 96 GiB (4096 MiB otherwise).

**The page file on a Windows APU.** Windows charges every allocation on the APU's GPU to its commit - RAM plus page
file - although it does not count the carve-out as RAM: on the PC below, the free commit fell by each of the engine's
GPU allocations, whatever their size. So the page file must be at least about the GPU's share. With less, either the
experts that do not fit the commit stay off the GPU (slower answers), or other programs run out of memory while the
model runs. Both engines (GLM-5.3-Flash's and DeepSeek's) keep their GPU experts within what Windows can still
commit, less max(8 GB, 10% of the RAM) for the other programs, and say how many GB of experts that left off the GPU,
and the page file that keeps them on it, in their log, the dashboard's Monitor and `/v1/status`. `--check` and the setup ask for the GPU's share plus max(8 GB, 10% of the RAM) - the RAM
holds Windows and the other programs - and warn when the page file is smaller: set a custom size (initial and maximum,
in MB) of at least that under System > About > Advanced system settings > Performance > Settings > Advanced > Virtual
memory > Change, then restart Windows. It is a reserve more than a file in use: the PC below ran with 160 GB of page
file for its 160 GB GPU share (the setup asks for 180 GB there: at 160 GB about 6 GB of commit was left while the
model ran), and at most 12 GB of it was ever written.

**Kernel submission.** Windows' HIP runtime keeps launches in a batch until the host waits on the GPU; Linux submits
each one. The engine's service thread waits for routes the GPU writes into host memory, so after 1 ms without one it
submits the batch (`cudaStreamQuery`, which does not wait). `GPU_FLUSH_ON_EXECUTION=1`, the runtime's own switch to
submit every launch, also works but costs ~31 us a launch.

Checked on a Ryzen AI Max+ PRO 495 / Radeon 8065S (Gorgon Halo, 192 GB LPDDR5X, 160 GB of it the GPU's), Windows 11,
HIP SDK 7.2 (HIP 7.2.60201), driver 32.0.31041, Maya-L, 32K context: all 15 HIP checks of [Validation](#validation)
pass, and every expert sits in the GPU pool (134.3 GB, 288 slots a layer, warmed from the NVMe in 44 s).

## Prompt speed

Prompts run in chunks, and inside a chunk the mixers and the dense FFN run in
sub-batches. Two settings matter on AMD:

- `MAYA_GLM_PREFILL_SUB` sets the sub-batch size. It was a fixed 256; it is
  now a runtime setting with 256 as the default. Larger sub-batches give the
  matrix cores bigger GEMMs. Setup writes `1024` together with
  `MAYA_GLM_PREFILL_MB=4096` for cards with 20 GB or more; the extra prompt
  memory is borrowed from the expert pool only while a prompt runs.
- `MAYA_HIPBLASLT_TUNING` points at a table of hipBLASLt solutions for the
  prompt's FP16 projections, measured per architecture with
  `tools/hip/tune_hipblaslt` (`tools/hip/<arch>-glm-hipblaslt-<version>.txt`).
  Setup writes it when a table for the card exists. The engine accepts a table
  only for its own architecture and hipBLASLt version; otherwise, and for shapes
  without a row, it keeps plain hipBLAS. The tuner's absolute-error gate now
  scales with `sqrt(K / 4096)`, so long-K projections are no longer rejected.

Measured prompt speed, GLM-5.3-Flash Maya-S-v2 IQ2_XXS, ROCm 7.2:

| Setting | RX 7900 XT | Ryzen AI Max+ 395 (8060S) | R9700 |
|---|---|---|---|
| prompt chunk forced to 256 (the earlier config) | ~50 tok/s | ~70 | - |
| engine-sized chunk | ~198 | ~116 | - |
| + tuned hipBLASLt | ~253-259 | ~195 | - |
| + `MAYA_GLM_PREFILL_SUB=1024`, `MAYA_GLM_PREFILL_MB=4096` | ~413 | ~217 | ~490-560 |

The 8060S runs used the same prompt-path changes on a local build. The R9700
runs used a pinned RAM tier sized for every expert the VRAM pool does not hold. On
gfx1201, plain hipBLAS is already close to hipBLASLt for most shapes, so its
table keeps only the rows that were at least 2% faster.

## What changed

- Added HIP GPU detection, compilation, rebuild stamps, and serving configs to
  `maya.py`, with separate CUDA and HIP build directories.
- Completed the CUDA-shaped runtime/BLAS mappings needed by GLM, excluded
  NVIDIA's profiling ranges from HIP, and linked the GLM prompt path to HIP MMQ
  and hipBLAS.
- Replaced the GLM hyper-connection kernel's unsupported FP32 `__ldcg` with
  agent-scope acquire loads on HIP. CUDA keeps its existing implementation.
- Kept CUDA WMMA attention guarded to CUDA builds. HIP uses the F32 attention
  kernel; since v1.0.7 its 32-row tile keeps the latent rows FP16 in shared
  memory (34 KB), within RDNA's 64 KiB workgroup limit.
- Fixed the KDA parity fixture to initialize its recurrence state explicitly;
  its numerical tolerance is unchanged.
- The HIP post-store system fences in both shared doorbell kernels and Maya's
  fused routing signal (backported from Strata) arrived separately in v1.0.7
  (#2); the validation below includes them.

Upstream Strata was inspected at `d5ea713` (0.1.40.3). Its AMD backend documents
runtime selection, memory limits, desktop VRAM headroom, and newer packed-byte
optimizations. Maya's GLM engine is separate from Strata's current Qwen engine,
so Strata's model benchmarks do not establish Maya performance.

## Validation

Test host: RX 7900 XT 20 GiB, Ryzen 7 9800X3D, 192 GB installed DDR5, native
Ubuntu Linux, system ROCm 7.2.1 / Clang 22, plus AI PRO R9700 32 GiB. Maya base v1.0.10 (`e4bd57e`).

The HIP engine and selected test targets build. All 14 selected GPU checks
pass: device allocation, expert upload staging, packed-byte/shuffle
intrinsics, asynchronous mapped-memory handoff, IQ1_S arithmetic, GLM
hyper-connections, KDA, FFN, DSA, layer arithmetic, synthetic-model logits
against the committed reference, and HIP MMQ. Six isolated installer tests
pass. The fused HC decode kernel also agrees with the independent CPU
reference across eight consecutive 4096-wide steps, including in-place gate
updates, in all three variants (default HC3, forced HC2, and forced HC1).
The full-model runs are below.

The real batched MLA attention kernel agrees with a double-precision CPU
softmax reference (absolute tolerance `5e-5`), including empty and masked cell
lists, partial and multiple tiles, and multiple head groups. It reads the
FP16 latent cache added in Maya v1.0.4.

The GLM handoff test replays the real routing graph for 100 disk requests
and 100 CPU-lane requests with changing IDs, weights, inputs and answers. The
CPU observes each request without a driver query or stream synchronization.
The same multi-architecture engine and test build pass all 14 checks on the
7900 XT and R9700. HC1 and HC2 also pass on the R9700. The HIP intrinsics check
covers all 65,536 pairs of byte values with varying neighboring lanes and
permutation selectors, against per-byte CPU references. The runtime rejects
the uncompiled `gfx1036` integrated GPU with the actual device and compiled
architecture list in its message. See [R9700 build details](AMD_RDNA4.md) and
[the shared AMD backport audit](AMD_UPSTREAM.md).

```sh
python3 tools/test_maya_hip.py
LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=0 \
  ctest --test-dir build-hip --output-on-failure --timeout 60 \
  -R '^(glm_(hc|kda|ffn|dsa|layer)_parity|glm_model_test|hip_intrinsics|hip_(glm_handoff|glm_prefill_attention|handoff|device_selftest|prefill_mmq_parity|expert_cache_staging)|iq1_s_parity)$'
MAYA_GLM_HC2=1 LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=0 \
  build-hip/glm_hc_parity --selftest
MAYA_GLM_HC1=1 LD_LIBRARY_PATH=/opt/rocm/lib HIP_VISIBLE_DEVICES=0 \
  build-hip/glm_hc_parity --selftest
```

The relevant build targets must be built before running CTest. Build and test
logs are kept in `build-hip/`.

Full model (Maya-S, 8K context) on the RX 7900 XT with the config setup writes:
seven smoke requests (arithmetic, code, a two-turn conversation, three
400-token answers) and five back-to-back prompt+answer rounds complete, with
prompts at 307-385 tok/s on 1.8-2.5K-token prompts and the hipBLASLt table
loaded. The R9700 runs the full model with every expert in VRAM or a pinned
RAM tier; the prompt speeds above were measured there.
