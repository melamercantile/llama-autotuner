#!/usr/bin/env python3
"""
llamatune - find the fastest llama-server settings for YOUR machine, with a terminal UI.

Pure Python standard library (curses for the UI). Works on Linux and macOS; Windows needs
`pip install windows-curses` for the TUI (headless mode works everywhere).

Auto-detected
  * OS, CPU (physical / performance cores), RAM, GPUs and their free memory
    (NVIDIA, AMD, Intel, Apple: via nvidia-smi / rocm-smi / `llama-server --list-devices`)
  * llama-server binaries (PATH, common build dirs, running processes, systemd units),
    their backends (CUDA / ROCm / Vulkan / Metal / SYCL) and which flags they support
  * models (.gguf in the llama.cpp / Hugging Face caches and common folders) and, by reading
    the GGUF header: architecture, layers, MoE or dense, expert size, trained context, MTP
  * your current setup (running llama-server, systemd unit): model, context, extra flags

CPU-only mode (--cpu-only, or automatic when no GPU is found): ignores any GPU and tunes threads,
prefill threads, -ub, KV cache type, flash attention, mmap/mlock, NUMA and MTP against RAM.

What it tunes (only flags your build supports)
  * how much of the model stays in RAM: --n-cpu-moe for MoE models, -ngl for dense ones
  * -ub micro-batch, -t threads, KV cache types, --moe-cache-mib (GPU expert cache),
    MTP speculative decoding. If the context you want cannot fit, it halves it and tells you.
  * any extra flag you name with --sweep (e.g. a new llama.cpp option such as --moe-cache-cpu-share):
    each value is loaded and measured on top of the best config found, and the winner is kept.
  * every config is loaded for real, measured with a real request, and scored on a
    typical turn of your chosen workload; the fastest one that leaves VRAM margin wins.

Several models
  Every finished tuning is saved as a profile (~/.cache/llamatune/profiles.json), one per model.
  Export them all as a llama-swap config or a llama-server router-mode preset file:
    ./llamatune.py --export-swap ~/llama-swap.yaml --export-router ~/models.ini

Usage
  ./llamatune.py                 # TUI
  ./llamatune.py --detect        # print what was detected (JSON) and exit
  ./llamatune.py --headless --model ~/models/x.gguf --ctx 32768 --save-script run.sh
  ./llamatune.py --headless --model x.gguf --sweep moe-cache-cpu-share=0.25,0.5,0.75
"""
import argparse
import collections
import hashlib
import json
import mmap
import os
import platform
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import median

try:
    import curses
except ImportError:  # Windows without windows-curses
    curses = None

VERSION = "llamatune 1.1"
IS_WIN = os.name == "nt"
HOME = Path.home()
STATE_DIR = Path(os.environ.get("XDG_CACHE_HOME", str(HOME / ".cache"))) / "llamatune"
BIN_NAME = "llama-server.exe" if IS_WIN else "llama-server"

GOALS = {
    "agent": (4000, 300, "Agent turns: long context, lots of new prompt"),
    "chat": (400, 400, "Interactive chat"),
    "prefill": (8000, 100, "Prefill-heavy: RAG, summaries, code review"),
    "generation": (0, 1000, "Pure generation speed"),
}

# flags the tuner controls itself: they are removed from the "extra" flags we carry over
TUNED_FLAGS = {
    "-c", "--ctx-size", "-ngl", "--gpu-layers", "--n-gpu-layers", "--n-cpu-moe", "-ncmoe",
    "--cpu-moe", "-cmoe", "-fa", "--flash-attn", "--cache-type-k", "-ctk", "--cache-type-v",
    "-ctv", "-b", "--batch-size", "-ub", "--ubatch-size", "-t", "--threads", "--moe-cache-mib",
    "--cache-reuse", "--fit", "-np", "--parallel", "--host", "--port", "--alias", "-m",
    "--model", "-hf", "--hf-repo", "--spec-type", "-tb", "--threads-batch",
}
BOOL_TUNED = {"--cpu-moe", "-cmoe"}


class Aborted(Exception):
    pass


class BudgetExceeded(Exception):
    pass


def run(cmd, timeout=30, **kw):
    try:
        return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, **kw)
    except Exception as e:  # missing binary, timeout, ...
        return subprocess.CompletedProcess(cmd, 127, "", str(e))


def mib(n):
    return n / 1048576.0


# ============================================================================ hardware
def detect_cpu():
    logical = os.cpu_count() or 1
    phys, perf, name = logical, logical, platform.processor() or "CPU"
    numa = 1
    system = platform.system()
    if system == "Linux":
        env = dict(os.environ, LC_ALL="C")
        out = run(["lscpu", "-p=CPU,CORE,MAXMHZ"], env=env).stdout
        cores = {}
        for line in out.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(",")
            try:
                mhz = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
            except ValueError:
                mhz = 0.0
            cores[parts[1]] = max(cores.get(parts[1], 0.0), mhz)
        if cores:
            phys = len(cores)
            mx = max(cores.values())
            perf = sum(1 for v in cores.values() if v >= 0.9 * mx) if mx else phys
        m = re.search(r"Model name:\s*(.+)", run(["lscpu"], env=env).stdout)
        if m:
            name = m.group(1).strip()
        mn = re.search(r"NUMA node\(s\):\s*(\d+)", run(["lscpu"], env=env).stdout)
        if mn:
            numa = int(mn.group(1))
    elif system == "Darwin":
        def sc(key):
            r = run(["sysctl", "-n", key]).stdout.strip()
            return int(r) if r.isdigit() else None
        phys = sc("hw.physicalcpu") or logical
        perf = sc("hw.perflevel0.physicalcpu") or phys
        name = run(["sysctl", "-n", "machdep.cpu.brand_string"]).stdout.strip() or name
    else:
        phys = max(1, logical // 2)
        perf = phys
    return {"name": name, "logical": logical, "physical": phys, "perf": perf, "numa": numa}


def detect_ram():
    total = avail = swap_used = None
    try:
        if Path("/proc/meminfo").exists():
            d = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, v = line.split(":", 1)
                d[k] = int(v.split()[0]) // 1024
            total, avail = d.get("MemTotal"), d.get("MemAvailable")
            swap_used = d.get("SwapTotal", 0) - d.get("SwapFree", 0)
        else:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 1048576
    except Exception:
        pass
    return {"total": total, "avail": avail, "swap_used": swap_used or 0}


def swap_used_mib():
    return detect_ram()["swap_used"]


class GpuProbe:
    """Free/total memory per device, from whichever tool is available."""

    def __init__(self, binpath):
        self.bin = binpath

    def read(self):
        for fn in (self._nvidia, self._rocm, self._listdev):
            r = fn()
            if r:
                return r
        return []

    def _nvidia(self):
        if not shutil.which("nvidia-smi"):
            return []
        r = run(["nvidia-smi", "--query-gpu=name,memory.free,memory.total",
                 "--format=csv,noheader,nounits"], timeout=15)
        out = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                out.append({"name": parts[0], "free": int(parts[1]), "total": int(parts[2]),
                            "api": "nvidia-smi"})
        return out

    def _rocm(self):
        if not shutil.which("rocm-smi"):
            return []
        r = run(["rocm-smi", "--showmeminfo", "vram", "--json"], timeout=15)
        try:
            data = json.loads(r.stdout)
        except Exception:
            return []
        out = []
        for key, val in data.items():
            if not key.startswith("card") or not isinstance(val, dict):
                continue
            try:
                total = int(val.get("VRAM Total Memory (B)"))
                used = int(val.get("VRAM Total Used Memory (B)"))
            except (TypeError, ValueError):
                continue
            out.append({"name": key, "free": int(mib(total - used)), "total": int(mib(total)),
                        "api": "rocm-smi"})
        return out

    def _listdev(self):
        if not self.bin or not Path(self.bin).exists():
            return []
        return [d for d in parse_devices(self.bin) if not d["id"].upper().startswith("CPU")]


DEV_RE = re.compile(r"^\s*([A-Za-z]+\d*):\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)", re.M)


def parse_devices(binpath):
    r = run([binpath, "--list-devices"], timeout=60)
    out = []
    for m in DEV_RE.finditer(r.stdout + r.stderr):
        out.append({"id": m.group(1), "name": m.group(2), "total": int(m.group(3)),
                    "free": int(m.group(4)), "api": "llama-server"})
    return out


def detect_gpu_hardware():
    """Vendors present on the machine, even if no llama.cpp backend can use them."""
    vendors = set()
    if shutil.which("nvidia-smi") and run(["nvidia-smi", "-L"]).stdout.strip():
        vendors.add("nvidia")
    if platform.system() == "Linux":
        out = run(["lspci"]).stdout
        for line in out.splitlines():
            if re.search(r"VGA|3D|Display", line):
                if re.search(r"NVIDIA", line, re.I):
                    vendors.add("nvidia")
                elif re.search(r"AMD|ATI|Advanced Micro", line, re.I):
                    vendors.add("amd")
                elif re.search(r"Intel", line, re.I):
                    vendors.add("intel")
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        vendors.add("apple")
    return sorted(vendors)


def suggest_build(vendors):
    """(backend name, cmake flags, note) for the machine's GPU."""
    if "apple" in vendors:
        return "Metal", [], "Metal is enabled by default on Apple Silicon."
    if "nvidia" in vendors:
        arch = ""
        cap = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]).stdout.strip().splitlines()
        if cap and re.match(r"\d+\.\d+$", cap[0].strip()):
            arch = cap[0].strip().replace(".", "")
        flags = ["-DGGML_CUDA=ON"] + ([f"-DCMAKE_CUDA_ARCHITECTURES={arch}"] if arch else [])
        return "CUDA", flags, "Needs the CUDA toolkit (nvcc), e.g. `sudo pacman -S cuda` or `apt install nvidia-cuda-toolkit`."
    if "amd" in vendors:
        if shutil.which("rocminfo") or Path("/opt/rocm").exists():
            flags = ["-DGGML_HIP=ON"]
            gfx = re.search(r"(gfx[0-9a-f]+)", run(["rocminfo"]).stdout)
            if gfx:
                flags.append(f"-DGPU_TARGETS={gfx.group(1)}")
            return "ROCm/HIP", flags, "Needs the ROCm/HIP SDK."
        return "Vulkan", ["-DGGML_VULKAN=ON"], "Needs the Vulkan SDK and glslc (ROCm not found)."
    if "intel" in vendors:
        return "Vulkan", ["-DGGML_VULKAN=ON"], "Needs the Vulkan SDK and glslc."
    return "CPU", [], "No GPU found: a plain CPU build is fine."


# ============================================================================ llama.cpp
class Caps:
    def __init__(self, path):
        self.path = path
        h = run([path, "--help"], timeout=60)
        self.help = h.stdout + h.stderr
        self.devices = parse_devices(path)
        v = run([path, "--version"], timeout=30)
        m = re.search(r"version:?\s*([^\n]+)", v.stdout + v.stderr)
        self.version = m.group(1).strip() if m else "?"
        self.backends = sorted({re.match(r"[A-Za-z]+", d["id"]).group(0)
                                for d in self.devices if not d["id"].upper().startswith("CPU")})
        self.unified = any(b.upper() in ("MTL", "METAL") for b in self.backends)

    def has(self, flag):
        return flag in self.help

    @property
    def gpu_ok(self):
        return bool(self.backends)


def find_running_servers():
    procs = []
    if Path("/proc").exists():
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                raw = (d / "cmdline").read_bytes().split(b"\0")
            except Exception:
                continue
            args = [a.decode("utf-8", "replace") for a in raw if a]
            if args and Path(args[0]).name.startswith("llama-server") and os.getpid() != int(d.name):
                procs.append(args)
    elif not IS_WIN:
        for line in run(["ps", "-eo", "args"]).stdout.splitlines():
            try:
                args = shlex.split(line)
            except ValueError:
                continue
            if args and Path(args[0]).name.startswith("llama-server"):
                procs.append(args)
    return procs


def parse_cmdline(args):
    """Split a llama-server command line into known settings and 'extra' flags."""
    out = {"args": list(args), "bin": args[0] if args else "", "model": None, "hf": None, "ctx": None, "alias": None,
           "host": None, "port": None, "extra": []}
    flat = []
    for a in args[1:]:
        if a.startswith("--") and "=" in a:
            k, v = a.split("=", 1)
            flat += [k, v]
        else:
            flat.append(a)
    i = 0
    while i < len(flat):
        a = flat[i]
        nxt = flat[i + 1] if i + 1 < len(flat) else None
        has_val = nxt is not None and (not nxt.startswith("-") or re.match(r"-\d", nxt) is not None)
        if a in BOOL_TUNED:
            has_val = False
        if a in ("-fa", "--flash-attn"):
            has_val = nxt in ("on", "off", "auto", "1", "0")
        val = nxt if has_val else None
        step = 2 if has_val else 1
        if a in ("-m", "--model"):
            out["model"] = val
        elif a in ("-hf", "--hf-repo"):
            out["hf"] = val
        elif a in ("-c", "--ctx-size") and val and val.isdigit():
            out["ctx"] = int(val)
        elif a == "--alias":
            out["alias"] = val
        elif a == "--host":
            out["host"] = val
        elif a == "--port" and val and val.isdigit():
            out["port"] = int(val)
        elif a in TUNED_FLAGS:
            pass
        else:
            out["extra"] += [a] + ([val] if has_val else [])
        i += step
    return out


def find_units():
    """systemd user units that run llama-server: [(path, parsed command line)]"""
    d = HOME / ".config" / "systemd" / "user"
    found = []
    if not d.exists():
        return found
    for p in d.glob("*.service"):
        try:
            txt = p.read_text()
        except Exception:
            continue
        m = re.search(r"^ExecStart=((?:.*\\\n)*.*)", txt, re.M)
        if not m or "llama-server" not in m.group(1):
            continue
        s = m.group(1).replace("\\\n", " ").replace("%h", str(HOME))
        try:
            args = shlex.split(s)
        except ValueError:
            continue
        if args and args[0][:1] in "-@:+!":
            args[0] = args[0][1:]
        found.append((p, parse_cmdline(args)))
    return found


def find_binaries(extra_paths=()):
    cands, seen = [], set()

    def add(p):
        try:
            p = Path(p)
            if p.is_file() and os.access(p, os.X_OK):
                r = str(p.resolve())
                if r not in seen:
                    seen.add(r)
                    cands.append(r)
        except Exception:
            pass

    for p in extra_paths:
        add(p)
    w = shutil.which(BIN_NAME)
    if w:
        add(w)
    for d in (HOME / "llama.cpp/build/bin", HOME / "llama.cpp/build/bin/Release",
              HOME / "src/llama.cpp/build/bin", HOME / "git/llama.cpp/build/bin",
              Path.cwd() / "build/bin", Path("/usr/local/bin"), Path("/usr/bin"),
              Path("/opt/homebrew/bin"), Path("/opt/llama.cpp/bin")):
        add(d / BIN_NAME)
    return cands


# ============================================================================ GGUF
_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
        10: "<Q", 11: "<q", 12: "<d"}
_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _parse_gguf(path):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        if mm[:4] != b"GGUF":
            return None
        u = struct.unpack_from
        pos = 8
        nt, nkv = u("<QQ", mm, pos)
        pos += 16

        def rstr(p):
            n, = u("<Q", mm, p)
            return bytes(mm[p + 8:p + 8 + n]).decode("utf-8", "replace"), p + 8 + n

        def rval(t, p):
            if t in _FMT:
                return u(_FMT[t], mm, p)[0], p + _SZ[t]
            if t == 8:
                return rstr(p)
            if t == 9:
                et, = u("<I", mm, p)
                n, = u("<Q", mm, p + 4)
                p += 12
                if et in _SZ:
                    return ("array", n), p + n * _SZ[et]
                for _ in range(n):
                    if et == 8:
                        ln, = u("<Q", mm, p)
                        p += 8 + ln
                    else:
                        _, p = rval(et, p)
                return ("array", n), p
            raise ValueError(f"unknown gguf type {t}")

        kv = {}
        for _ in range(nkv):
            k, pos = rstr(pos)
            t, = u("<I", mm, pos)
            v, pos = rval(t, pos + 4)
            kv[k] = v
        tens = []
        for _ in range(nt):
            name, pos = rstr(pos)
            nd, = u("<I", mm, pos)
            pos += 4 + 8 * nd + 4
            off, = u("<Q", mm, pos)
            pos += 8
            tens.append((name, off))
        align = kv.get("general.alignment", 32)
        data_start = (pos + align - 1) // align * align
        tens.sort(key=lambda x: x[1])
        sized = []
        for i, (n, o) in enumerate(tens):
            end = tens[i + 1][1] if i + 1 < len(tens) else len(mm) - data_start
            sized.append((n, max(0, end - o)))
        return kv, sized
    finally:
        mm.close()


def read_model_info(path):
    """Architecture, layers, MoE-ness, sizes... from the GGUF header (and all shards)."""
    try:
        p = Path(path)
        shards = [p]
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", p.name)
        if m:
            n = int(m.group(2))
            base = p.name[:m.start()]
            shards = [p.with_name(f"{base}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]
            shards = [s for s in shards if s.exists()]
        kv0, expert, total = None, 0, 0
        for i, sp in enumerate(shards):
            r = _parse_gguf(sp)
            if not r:
                return None
            kv, tens = r
            if i == 0:
                kv0 = kv
            total += sp.stat().st_size
            expert += sum(sz for n, sz in tens if "_exps" in n)
        arch = kv0.get("general.architecture", "?")

        def g(key, default=None):
            v = kv0.get(f"{arch}.{key}", default)
            return None if isinstance(v, tuple) else v

        heads, embd = g("attention.head_count"), g("embedding_length")
        hdim = g("attention.key_length") or (embd // heads if heads and embd else None)
        return {
            "path": str(path), "name": kv0.get("general.name", p.stem), "arch": arch,
            "n_layer": g("block_count"), "n_expert": g("expert_count", 0) or 0,
            "n_expert_used": g("expert_used_count", 0) or 0,
            "ctx_train": g("context_length"), "n_head_kv": g("attention.head_count_kv"),
            "head_dim": hdim, "nextn": g("nextn_predict_layers", 0) or 0,
            "total_bytes": total, "expert_bytes": expert, "moe": expert > 0 or (g("expert_count", 0) or 0) > 1,
        }
    except Exception:
        return None


def find_models(limit=60):
    dirs = []
    if os.environ.get("LLAMA_CACHE"):
        dirs.append(Path(os.environ["LLAMA_CACHE"]))
    xdg = Path(os.environ.get("XDG_CACHE_HOME", str(HOME / ".cache")))
    dirs += [xdg / "llama.cpp", HOME / "Library/Caches/llama.cpp", HOME / ".cache/huggingface/hub",
             HOME / "models", HOME / "llama.cpp/models", HOME / "Downloads", Path.cwd()]
    if os.environ.get("LOCALAPPDATA"):
        dirs.append(Path(os.environ["LOCALAPPDATA"]) / "llama.cpp")
    found, seen = [], set()
    for root in dirs:
        if not root.exists() or root == HOME:
            continue
        base_depth = len(root.parts)
        for dp, dn, fn in os.walk(root, followlinks=False):
            if len(Path(dp).parts) - base_depth > 6:
                dn[:] = []
                continue
            for f in fn:
                if not f.lower().endswith(".gguf") or "mmproj" in f.lower():
                    continue
                if re.search(r"-\d{5}-of-\d{5}\.gguf$", f) and "-00001-of-" not in f:
                    continue
                p = Path(dp) / f
                try:
                    r = str(p.resolve())
                    if r in seen:
                        continue
                    seen.add(r)
                    found.append({"path": str(p), "size": p.stat().st_size, "mtime": p.stat().st_mtime})
                except OSError:
                    pass
            if len(found) >= limit:
                break
    found.sort(key=lambda m: -m["mtime"])
    return found


# ============================================================================ settings
class Settings:
    def __init__(self):
        self.bin = ""
        self.model = ""      # local .gguf path
        self.hf = ""         # or a Hugging Face spec (user/repo:quant)
        self.ctx = 32768
        self.depth = 8192
        self.margin = 700
        self.goal = "agent"
        self.kv_q4 = False
        self.moe_cache = True
        self.spec = True
        self.quick = False
        self.runs = 1
        self.port = 8099
        self.alias = ""
        self.host = "127.0.0.1"
        self.service_port = 8080
        self.extra = []
        self.sweep = []       # [(flag, [values])] user-named flags to try, see --sweep
        self.load_timeout = 1800
        self.max_probes = 60
        self.stop_service = True
        self.stop_ollama = True
        self.fresh = False
        self.cpu_only = False
        self.compare_current = True
        self.service_name = ""  # blank = llama-server-<model name>


class Detection:
    """Everything auto-detected about the machine and the llama.cpp install."""

    def __init__(self):
        self.cpu = detect_cpu()
        self.ram = detect_ram()
        self.vendors = detect_gpu_hardware()
        self.running = find_running_servers()
        self.units = find_units()
        self.bins = []
        self.caps = None
        self.models = []
        self.minfo = None
        self.gpus = []
        self.ollama_loaded = []
        self.current_args = None
        self.current_model = (None, None)

    def refresh(self, settings):
        extra_bins = [c["bin"] for c in (parse_cmdline(a) for a in self.running)]
        extra_bins += [u[1]["bin"] for u in self.units]
        extra_bins.append(settings.bin)
        self.bins = find_binaries([b for b in extra_bins if b])
        if not settings.bin and self.bins:
            settings.bin = self.bins[0]
        self.caps = Caps(settings.bin) if settings.bin and Path(settings.bin).exists() else None
        self.models = find_models()
        self.gpus = GpuProbe(settings.bin).read()
        self.ram = detect_ram()
        self.ollama_loaded = []
        if shutil.which("ollama"):
            lines = run(["ollama", "ps"]).stdout.strip().splitlines()[1:]
            self.ollama_loaded = [ln.split()[0] for ln in lines if ln.split()]
        self.load_model(settings)

    def load_model(self, settings):
        self.minfo = read_model_info(settings.model) if settings.model and Path(settings.model).exists() else None

    def warnings(self, s):
        w = []
        if not s.bin or not Path(s.bin).exists():
            w.append(("err", "No llama-server found. Press b to download/build one."))
        elif self.caps and not self.caps.gpu_ok and self.vendors and self.vendors != ["apple"] and not s.cpu_only:
            name, _, _ = suggest_build(self.vendors)
            w.append(("err", f"This llama-server has no GPU backend but the machine has a GPU. Press b to rebuild ({name})."))
        if not s.model and not s.hf:
            w.append(("err", "No model selected."))
        elif s.model and not Path(s.model).exists():
            w.append(("err", "Model file not found."))
        if s.cpu_only:
            if self.caps and self.caps.gpu_ok:
                w.append(("warn", "CPU-only mode: the GPU backend is present but will be ignored."))
            if self.minfo and self.ram["avail"] and mib(self.minfo["total_bytes"]) * 1.05 > self.ram["avail"]:
                w.append(("warn", "The model is larger than the available RAM; expect swapping."))
            if s.depth > 4096:
                w.append(("warn", "CPU prefill is slow: each probe at this depth can take minutes. Try depth 2048."))
        if self.running:
            w.append(("warn", f"{len(self.running)} llama-server process(es) running; they hold memory. Stop them for accurate results."))
        if self.ollama_loaded:
            w.append(("warn", "Ollama has models loaded (" + ", ".join(self.ollama_loaded) + "); they will be unloaded."))
        if self.minfo and self.minfo["ctx_train"] and s.ctx > self.minfo["ctx_train"]:
            w.append(("warn", f"Context {s.ctx} is above the model's trained context ({self.minfo['ctx_train']})."))
        return w

    def report(self, s):
        return {
            "cpu": self.cpu, "ram": self.ram, "gpu_vendors": self.vendors, "gpus": self.gpus,
            "binaries": self.bins, "selected_bin": s.bin, "cpu_only": s.cpu_only,
            "backends": self.caps.backends if self.caps else None,
            "version": self.caps.version if self.caps else None,
            "models": [m["path"] for m in self.models[:15]], "selected_model": s.model or s.hf,
            "model_info": self.minfo, "running_servers": self.running,
            "units": [str(u[0]) for u in self.units], "ollama_loaded": self.ollama_loaded,
        }


def apply_existing_config(s, det):
    """Import model/context/flags from a running server or a systemd unit."""
    src = None
    if det.running:
        src = parse_cmdline(det.running[0])
    elif det.units:
        src = det.units[0][1]
    if not src:
        return
    det.current_args = src["args"]
    det.current_model = (src["model"], src["hf"])
    if src["bin"] and not s.bin:
        s.bin = src["bin"]
    if src["model"] and not (s.model or s.hf):
        s.model = src["model"]
    if src["hf"] and not (s.model or s.hf):
        s.hf = src["hf"]
    if src["ctx"] and s.ctx == 32768:
        s.ctx = src["ctx"]
    s.alias = s.alias or src["alias"] or ""
    s.service_port = src["port"] or s.service_port
    s.host = src["host"] or s.host
    if not s.extra:
        s.extra = list(src["extra"])


def pick_default_model(s, det):
    if s.model or s.hf:
        return
    if det.models:
        s.model = det.models[0]["path"]
        det.load_model(s)


# ============================================================================ run state
class RunState:
    def __init__(self, echo=False):
        self.lock = threading.Lock()
        self.log = collections.deque(maxlen=800)
        self.kind = "tune"
        self.stage = "starting"
        self.current = ""
        self.probes = 0
        self.est = 20
        self.results = []
        self.best = None
        self.done = False
        self.error = None
        self.abort = False
        self.t0 = time.time()
        self.echo = echo
        self.notes = []

    def add(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with self.lock:
            self.log.append(line)
        if self.echo:
            print(line, flush=True)


# ============================================================================ engine
def norm(cfg):
    drop = {"cache": 0, "spec": False, "mmap": True, "lock": False, "tb": 0, "fa": True,
            "numa": False, "cpu": False, "x": []}
    return {k: v for k, v in cfg.items() if not (k in drop and v == drop[k] and type(v) == type(drop[k]))}


def describe(cfg):
    bits = [] if cfg.get("cpu") else [f"host={cfg['host']}"]
    bits += [f"ub={cfg['ub']}", f"t={cfg['t']}"]
    if cfg.get("tb"):
        bits.append(f"tb={cfg['tb']}")
    bits.append(f"kv={cfg['ctk']}/{cfg['ctv']}")
    if cfg.get("cache"):
        bits.append(f"cache={cfg['cache']}MiB")
    if cfg.get("spec"):
        bits.append("mtp")
    for fl, val in cfg.get("x") or []:
        bits.append(f"{fl.lstrip('-')}={val}")
    if not cfg.get("mmap", True):
        bits.append("no-mmap")
    if cfg.get("lock"):
        bits.append("mlock")
    if not cfg.get("fa", True):
        bits.append("fa=off")
    if cfg.get("numa"):
        bits.append("numa")
    if cfg.get("ctx"):
        bits.append(f"ctx={cfg['ctx']}")
    return " ".join(bits)


def kill_tree(p):
    if p.poll() is not None:
        return
    try:
        if IS_WIN:
            p.terminate()
        else:
            os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=20)
    except Exception:
        try:
            if IS_WIN:
                p.kill()
            else:
                os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


OOM_PAT = re.compile(r"out of memory|failed to allocate|cudamalloc failed|unable to allocate|"
                     r"alloc_buffer|cuda error|vk::.*outofdevicememory", re.I)


class Engine:
    def __init__(self, s, det, st):
        self.s, self.det, self.st = s, det, st
        self.caps = det.caps
        self.cpu = det.cpu
        self.minfo = det.minfo
        self.moe = bool(self.minfo and self.minfo["moe"])
        self.nl = (self.minfo or {}).get("n_layer") or None
        self.host_mib = None
        self.gpu = GpuProbe(s.bin)
        self.base_used = None
        self.fresh = 0
        self.stopped_units = []
        self.unloaded = []
        self.baseline = None
        self.first_ok = None
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        (STATE_DIR / "logs").mkdir(exist_ok=True)
        gpu_ident = [d["name"] for d in det.gpus]
        ident = json.dumps([s.bin, os.path.getmtime(s.bin), s.model or s.hf,
                            os.path.getsize(s.model) if s.model and Path(s.model).exists() else 0,
                            s.depth, s.margin, s.runs, gpu_ident, self.caps.version, s.cpu_only])
        self.cache_file = STATE_DIR / ("cache-" + hashlib.sha1(ident.encode()).hexdigest()[:12] + ".json")
        self.cache = {}
        if self.cache_file.exists() and not s.fresh:
            try:
                self.cache = json.loads(self.cache_file.read_text())
            except Exception:
                self.cache = {}
        extra = []
        toks = list(s.extra)
        swept = {fl for fl, _ in s.sweep}
        i = 0
        while i < len(toks):
            t = toks[i]
            if t in swept:  # the sweep sets this flag itself; do not pass it twice
                i += 1
                if i < len(toks) and not toks[i].startswith("-"):
                    i += 1
                continue
            if t.startswith("-") and not self.caps.has(t):
                st.add(f"note: this build does not know '{t}', dropped from the tuned command")
                i += 1
                if i < len(toks) and not toks[i].startswith("-"):
                    i += 1
                continue
            extra.append(t)
            i += 1
        self.extra = extra

    # ------------------------------------------------------------------ command line
    def fa_args(self):
        h = self.caps.help
        if not (self.caps.has("--flash-attn") or self.caps.has("-fa")):
            return []
        m = re.search(r"--flash-attn[^\n]*\n?[^\n]*", h)
        if m and re.search(r"\bauto\b", m.group(0)):
            return ["-fa", "on"]
        return ["-fa"]

    def cmd(self, cfg, port, alias=None, host="127.0.0.1"):
        s, caps = self.s, self.caps
        c = [s.bin] + (["-m", s.model] if s.model else ["-hf", s.hf])
        if alias:
            c += ["--alias", alias]
        c += ["--host", host, "--port", str(port), "-c", str(cfg["ctx"]), "-np", "1"]
        nl = self.nl or 99
        h = cfg["host"]
        if cfg.get("cpu"):
            c += ["-ngl", "0"]
            if caps.has("--device"):
                c += ["--device", "none"]
            if caps.has("--no-op-offload"):
                c += ["--no-op-offload"]
        elif self.moe:
            c += ["-ngl", "99"]
            if h > 0:
                if caps.has("--n-cpu-moe"):
                    c += ["--n-cpu-moe", str(h)]
                elif caps.has("--override-tensor"):
                    layers = "|".join(str(i) for i in range(min(h, nl)))
                    c += ["-ot", rf"blk\.({layers})\.ffn_.*_exps\.=CPU"]
        else:
            c += ["-ngl", "99" if h == 0 else str(max(0, nl - h))]
        fa = self.fa_args()
        if not cfg.get("fa", True):
            fa = ["-fa", "off"] if fa == ["-fa", "on"] else []
        c += fa
        if caps.has("--cache-type-k"):
            c += ["--cache-type-k", cfg["ctk"], "--cache-type-v", cfg["ctv"]]
        c += ["-b", str(max(2048, cfg["ub"])), "-ub", str(cfg["ub"]), "-t", str(cfg["t"])]
        if cfg.get("tb") and caps.has("--threads-batch"):
            c += ["-tb", str(cfg["tb"])]
        if not cfg.get("mmap", True) and caps.has("--no-mmap"):
            c += ["--no-mmap"]
        if cfg.get("lock") and caps.has("--mlock"):
            c += ["--mlock"]
        if cfg.get("numa") and caps.has("--numa"):
            c += ["--numa", "distribute"]
        if caps.has("--cache-reuse"):
            c += ["--cache-reuse", "256"]
        if caps.has("--fit"):
            c += ["--fit", "off"]
        if cfg.get("cache", 0) > 0 and caps.has("--moe-cache-mib"):
            c += ["--moe-cache-mib", str(cfg["cache"])]
        if cfg.get("spec") and caps.has("draft-mtp"):
            c += ["--spec-type", "draft-mtp"]
        for fl, val in cfg.get("x") or []:
            c += [fl, str(val)]
        return c + self.extra

    # ------------------------------------------------------------------ memory
    def mem(self):
        if self.s.cpu_only:
            return None
        devs = self.gpu.read()
        if not devs:
            return None
        return {"free_min": min(d["free"] for d in devs),
                "used": sum(d["total"] - d["free"] for d in devs),
                "total": sum(d["total"] for d in devs),
                "free": sum(d["free"] for d in devs)}

    def cpu_env(self):
        return {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "-1"}

    def final_env(self):
        if self.s.cpu_only and not self.caps.has("--device"):
            return self.cpu_env()
        return {}

    def free_now(self):
        if self.s.cpu_only:
            return detect_ram()["avail"]
        mem = self.mem()
        return mem["free_min"] if mem else None

    def low_memory(self, when):
        """Reason string if free VRAM (or RAM in CPU mode) dropped under the margin."""
        if self.s.cpu_only:
            a = detect_ram()["avail"]  # note: MemAvailable counts some mmap'd pages as reclaimable
            if a is not None and a < self.s.margin:
                return f"RAM margin {when} ({a} MiB available)"
            return None
        mem = self.mem()
        if mem and mem["free_min"] < self.s.margin:
            return f"VRAM margin {when} ({mem['free_min']} MiB free)"
        return None

    def per_layer_mib(self):
        nl = self.nl or 1
        if self.moe and self.minfo["expert_bytes"]:
            return mib(self.minfo["expert_bytes"]) / nl
        return mib(self.minfo["total_bytes"]) / (nl + 1) if self.minfo else 500.0

    def kv_mib(self, ctx):
        mi = self.minfo or {}
        if mi.get("n_head_kv") and mi.get("head_dim") and self.nl:
            return ctx * self.nl * 2 * mi["n_head_kv"] * mi["head_dim"] * 1.07 / 1048576.0
        return ctx * 0.05

    def guess_host(self, ctx):
        mem = self.mem()
        if self.caps.unified or not mem or not self.nl or not self.minfo:
            return 0
        avail = mem["free"] - self.s.margin - self.kv_mib(ctx) - 400
        if self.moe:
            room = avail - mib(self.minfo["total_bytes"] - self.minfo["expert_bytes"])
        else:
            room = avail
        on_gpu = max(0, int(room / max(1.0, self.per_layer_mib())))
        return max(0, min(self.nl, self.nl - on_gpu))

    # ------------------------------------------------------------------ scoring
    def turn_s(self, r):
        pp_t, tg_t, _ = GOALS[self.s.goal]
        return pp_t / max(r["pp"], 1e-6) + tg_t / max(r["tg"], 1e-6)

    def best_result(self):
        ok = [r for r in self.st.results if r["ok"]]
        return min(ok, key=self.turn_s) if ok else None

    # ------------------------------------------------------------------ http
    def post(self, path, data, timeout):
        req = urllib.request.Request(f"http://127.0.0.1:{self.s.port}{path}",
                                     data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def make_prompt(self, start=0):
        def line(i):
            return f"Entry {i}: the quick brown fox jumps over {i * 7919 % 1009} lazy dogs near river {i % 13}.\n"
        sample = "".join(line(i) for i in range(40))
        n = len(self.post("/tokenize", {"content": sample}, 30)["tokens"])
        cpt = len(sample) / max(1, n)
        want = int(self.s.depth * cpt)
        out, total, i = [], 0, start
        while total < want:
            ln = line(i)
            out.append(ln)
            total += len(ln)
            i += 1
        return "".join(out)

    def measure(self, warm):
        self.post("/completion", {"prompt": "Hello", "n_predict": 8, "cache_prompt": False}, 600)
        text = self.make_prompt()
        body = {"temperature": 0, "cache_prompt": False, "ignore_eos": True}
        if warm:  # let caches (expert cache, MTP) fill with a different context first
            self.post("/completion", dict(body, prompt=self.make_prompt(100000), n_predict=128), 3600)
        pps, tgs = [], []
        for _ in range(self.s.runs):
            r = self.post("/completion", dict(body, prompt=text, n_predict=128 if warm else 64), 3600)
            pps.append(r["timings"]["prompt_per_second"])
            tgs.append(r["timings"]["predicted_per_second"])
        return median(pps), median(tgs)

    # ------------------------------------------------------------------ one probe
    def check(self):
        if self.st.abort:
            raise Aborted()

    def evaluate(self, cfg):
        self.check()
        key = json.dumps(norm(cfg), sort_keys=True)
        cached = key in self.cache
        if cached:
            res = self.cache[key]
        else:
            if self.fresh >= self.s.max_probes:
                raise BudgetExceeded()
            self.fresh += 1
            res = self.probe(cfg)
            self.cache[key] = res
            try:
                self.cache_file.write_text(json.dumps(self.cache, indent=1))
            except OSError:
                pass
        self.st.probes += 1
        self.st.est = max(self.st.est, self.st.probes + 3)
        if res.get("host_mib"):
            self.host_mib = res["host_mib"]
        if res.get("n_layer") and not self.nl:
            self.nl = res["n_layer"]
        if res["ok"]:
            self.st.results.append(res)
            self.st.best = self.best_result()
            if self.first_ok is None:
                self.first_ok = res
        tag = " (cached)" if cached else ""
        if res["ok"]:
            self.st.add(f"OK   {describe(cfg)}  pp={res['pp']:.0f} tg={res['tg']:.1f} "
                        f"turn={self.turn_s(res):.1f}s free={res.get('free_min')}MiB{tag}")
        else:
            self.st.add(f"FAIL {describe(cfg)}  ({res['reason']}){tag}")
        return res

    def probe(self, cfg):
        s, st = self.s, self.st
        res = {"cfg": cfg, "ok": False, "reason": "?"}
        logp = STATE_DIR / "logs" / f"probe-{int(time.time() * 1000)}.log"
        res["log"] = logp.name
        swap0 = swap_used_mib()
        st.current = "loading " + describe(cfg)
        kw = {}
        if IS_WIN:
            kw["creationflags"] = 0x200  # CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        with open(logp, "w") as lf:
            if s.cpu_only:
                kw["env"] = dict(os.environ, **self.cpu_env())
            argv = self.cmd(cfg, s.port)
            lf.write("# " + " ".join(shlex.quote(x) for x in argv) + "\n")
            lf.flush()
            p = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, **kw)
        try:
            ok, reason = self.wait_ready(p, logp)
            txt = logp.read_text(errors="ignore")
            m = re.search(r"n_layer\s*=\s*(\d+)", txt)
            if m:
                res["n_layer"] = int(m.group(1))
            mh = re.search(r"MoE cache size = ([\d.]+) MiB for ([\d.]+) MiB of host experts", txt)
            if mh:
                res["host_mib"] = float(mh.group(2))
            small = re.search(r"MoE cache budget is too small", txt, re.I)
            if not ok:
                res["reason"] = reason
            elif cfg.get("cache", 0) > 0 and small:
                res["reason"] = "cache budget too small"
            else:
                why = self.low_memory("after load")
                if why:
                    res["reason"] = why
                else:
                    st.current = "measuring " + describe(cfg)
                    pp, tg = self.measure(warm=bool(cfg.get("cache") or cfg.get("spec")))
                    why = self.low_memory("after request")
                    if why:
                        res["reason"] = why
                    elif swap_used_mib() - swap0 > 512:
                        res["reason"] = "swapping (RAM too tight)"
                    elif p.poll() is not None:
                        res["reason"] = "server died during request"
                    else:
                        res.update(ok=True, reason="ok", pp=pp, tg=tg, free_min=self.free_now())
        except Aborted:
            raise
        except Exception as e:
            res["reason"] = f"runtime failure: {type(e).__name__}"
        finally:
            st.current = ""
            kill_tree(p)
            self.wait_release()
        return res

    def wait_ready(self, p, logp):
        t0 = time.time()
        while time.time() - t0 < self.s.load_timeout:
            if self.st.abort:
                raise Aborted()
            if p.poll() is not None:
                txt = logp.read_text(errors="ignore")
                return False, ("out of memory" if OOM_PAT.search(txt) else f"exited ({p.returncode})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.s.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return True, "ok"
            except Exception:
                pass
            time.sleep(0.5)
        return False, "load timeout"

    def wait_release(self):
        if self.base_used is None:
            return
        for _ in range(40):
            mem = self.mem()
            if not mem or mem["used"] <= self.base_used + 400:
                return
            time.sleep(0.5)

    # ------------------------------------------------------------------ boundary search
    def find_min_host(self, base, start):
        """Smallest 'layers kept in RAM' value that still loads (monotone in memory)."""
        nl = self.nl or 48

        def ev(n):
            return self.evaluate(dict(base, host=n))

        def refine(lo, hi, best):
            while hi - lo > 1:
                mid = (lo + hi) // 2
                r = ev(mid)
                if r["ok"]:
                    hi, best = mid, r
                else:
                    lo = mid
            return best

        n = max(0, min(nl, start))
        r = ev(n)
        if r["ok"]:
            best, step = r, 2
            while n > 0:
                m = max(0, n - step)
                r2 = ev(m)
                if r2["ok"]:
                    best, n = r2, m
                    step *= 2
                else:
                    return refine(m, n, best)
            return best
        lo, step = n, 2
        while lo < nl:
            m = min(nl, lo + step)
            r2 = ev(m)
            if r2["ok"]:
                return refine(lo, m, r2)
            lo = m
            step *= 2
        return None

    # ------------------------------------------------------------------ whole run
    def run(self):
        st = self.st
        try:
            self.prepare()
            self.baseline_stage()
            self.tune()
        except Aborted:
            st.add("aborted by user")
        except BudgetExceeded:
            st.add("probe budget reached; using the best result so far")
        except Exception as e:
            st.error = f"{type(e).__name__}: {e}"
            st.add("ERROR " + st.error)
        finally:
            self.cleanup()
            st.best = self.best_result()
            try:
                save_profile(self)
            except Exception:
                pass
            try:
                d = self.diff()
                if d:
                    (STATE_DIR / "last-diff.txt").write_text("\n".join(diff_lines(d)) + "\n")
            except Exception:
                pass
            try:
                with st.lock:
                    (STATE_DIR / "last-run.log").write_text("\n".join(st.log) + "\n")
            except Exception:
                pass
            st.stage = "finished"
            st.done = True

    def prepare(self):
        s, st = self.s, self.st
        st.stage = "preparing"
        if s.stop_service:
            for path, _ in self.det.units:
                name = path.name
                if run(["systemctl", "--user", "is-active", "--quiet", name]).returncode == 0:
                    st.add(f"stopping {name} while tuning (it is restarted afterwards)")
                    run(["systemctl", "--user", "stop", name])
                    self.stopped_units.append(name)
        if s.stop_ollama and shutil.which("ollama"):
            for ln in run(["ollama", "ps"]).stdout.strip().splitlines()[1:]:
                if ln.split():
                    run(["ollama", "stop", ln.split()[0]])
                    st.add(f"unloaded Ollama model {ln.split()[0]}")
        time.sleep(2)
        mem = self.mem()
        if mem:
            self.base_used = mem["used"]
            st.add(f"GPU memory: {mem['total']} MiB total, {mem['free']} MiB free "
                   f"({mem['used']} MiB used by other programs)")
        elif s.cpu_only:
            r = detect_ram()
            st.add(f"CPU-only mode: {r['avail']} MiB of {r['total']} MiB RAM available; GPUs are ignored")
        else:
            st.add("no GPU memory probe available; tuning by load success only")
        if not self.nl:
            st.add("could not read the layer count from the model; will use the server log")
            self.nl = 48

    def cleanup(self):
        for name in self.stopped_units:
            run(["systemctl", "--user", "start", name])
            self.st.add(f"restarted {name}")

    def tune(self):
        s, st, cpu = self.s, self.st, self.cpu
        if s.cpu_only:
            return self.tune_cpu()
        has_fa = bool(self.fa_args())
        base = {"ub": 1024, "t": cpu["perf"], "ctk": "q8_0" if has_fa else "f16",
                "ctv": "q8_0" if has_fa else "f16", "cache": 0, "spec": False, "ctx": s.ctx}
        st.est = 10 if s.quick else 24

        # ---- stage 1: how much stays in RAM
        st.stage = "searching the CPU/GPU split"
        ctx = s.ctx
        b1 = None
        while True:
            base["ctx"] = ctx
            b1 = self.find_min_host({k: v for k, v in base.items()}, self.guess_host(ctx))
            if b1 or ctx <= 4096:
                break
            ctx = max(4096, ctx // 2)
            st.add(f"nothing fits at the requested context; trying ctx={ctx}")
            st.notes.append(f"context reduced to {ctx}")
        if not b1:
            raise RuntimeError("nothing fits, even with everything in RAM. Use a smaller quant or model.")
        cur = b1
        base["ctx"] = ctx
        full_gpu = cur["cfg"]["host"] == 0

        # ---- stage 2: micro-batch
        if not s.quick:
            st.stage = "tuning micro-batch (-ub)"
            ubs = [512, 2048] + ([4096] if s.goal == "prefill" else [])
            for ub in ubs:
                self.find_min_host(dict(base, ub=ub), cur["cfg"]["host"])
            cur = self.best_result()

            # ---- stage 3: threads (only matters when something runs on the CPU)
            if cur["cfg"]["host"] > 0 or self.caps.unified:
                st.stage = "tuning threads (-t)"
                cands = {cpu["perf"], max(1, cpu["perf"] - 2), cpu["physical"], 4, 6, 8}
                for th in sorted(c for c in cands if 1 <= c <= cpu["logical"]):
                    if th != cur["cfg"]["t"]:
                        self.evaluate(dict(cur["cfg"], t=th))
                cur = self.best_result()

        # ---- optional: q4 V cache
        if s.kv_q4 and has_fa:
            st.stage = "trying a q4_0 V cache"
            cfg = dict(cur["cfg"], ctv="q4_0")
            self.find_min_host(cfg, cur["cfg"]["host"])
            cur = self.best_result()

        # ---- GPU expert cache
        if self.moe and s.moe_cache and self.caps.has("--moe-cache-mib") and cur["cfg"]["host"] >= 0:
            st.stage = "tuning the GPU expert cache"
            self.tune_cache(base)
            cur = self.best_result()
        elif self.moe and s.moe_cache and not self.caps.has("--moe-cache-mib"):
            st.add("note: this build has no --moe-cache-mib (update llama.cpp to get the GPU expert cache)")

        # ---- flags named with --sweep
        for fl, vals in s.sweep:
            if not self.caps.has(fl):
                st.add(f"note: this build has no {fl} (see `llama-server --help`); sweep skipped")
                continue
            st.stage = f"sweeping {fl}"
            for v in vals:
                cur = self.best_result()
                x = [e for e in (cur["cfg"].get("x") or []) if e[0] != fl] + [[fl, v]]
                self.evaluate(dict(cur["cfg"], x=x))
            cur = self.best_result()

        # ---- MTP speculative decoding
        if s.spec and self.minfo and self.minfo.get("nextn") and self.caps.has("draft-mtp"):
            st.stage = "trying MTP speculative decoding"
            self.evaluate(dict(cur["cfg"], spec=True))
        elif s.spec and self.minfo and self.minfo.get("nextn"):
            st.add("note: the model has an MTP block but this build has no draft-mtp support")

    # ------------------------------------------------------------------ baseline and diff
    def current_args(self):
        """Command line of the server you run today, if it is for the selected model."""
        d, s = self.det, self.s
        if not d.current_args:
            return None
        m, hf = d.current_model
        if s.model and m and Path(m).name == Path(s.model).name:
            return d.current_args
        if s.hf and hf and hf == s.hf:
            return d.current_args
        return None

    def baseline_cfg(self):
        args = self.current_args()
        if not args:
            return None
        f = flags_dict(args)
        nl = self.nl or 48

        def num(k, default):
            try:
                return int(f[k])
            except (KeyError, TypeError, ValueError):
                return default

        t = num("threads", self.cpu["physical"])
        tb = num("threads-batch", 0)
        fa = f.get("flash-attn", "on") not in ("off", "0", "false")
        cfg = {"ub": num("ubatch-size", 512), "t": t, "tb": 0 if tb == t else tb,
               "ctk": str(f.get("cache-type-k", "f16")), "ctv": str(f.get("cache-type-v", "f16")),
               "cache": num("moe-cache-mib", 0), "spec": f.get("spec-type") == "draft-mtp",
               "ctx": self.s.ctx, "fa": fa, "mmap": "no-mmap" not in f, "lock": "mlock" in f}
        xs = [[fl, str(f[fl.lstrip("-")])] for fl, _ in self.s.sweep
              if f.get(fl.lstrip("-")) not in (None, True)]
        if xs:
            cfg["x"] = xs
        if self.s.cpu_only:
            cfg.update(cpu=True, host=0)
        elif "cpu-moe" in f:
            cfg["host"] = nl
        elif self.moe:
            cfg["host"] = num("n-cpu-moe", 0)
        else:
            ngl = num("n-gpu-layers", 99)
            cfg["host"] = 0 if ngl >= nl else max(0, nl - ngl)
        return cfg

    def baseline_stage(self):
        if not self.s.compare_current:
            return
        cfg = self.baseline_cfg()
        if not cfg:
            return
        self.st.stage = "measuring your current config"
        self.st.est += 1
        self.st.add("measuring your current configuration first, to compare against")
        self.baseline = self.evaluate(cfg)

    def diff(self):
        """What changed between your current command (or the first working config) and the tuned one."""
        best = self.best_result()
        if not best:
            return None
        new = flags_dict(self.final_tokens(best["cfg"]))
        cur = self.current_args()
        if cur:
            old, label, b = flags_dict(cur), "your current command", self.baseline
        else:
            b = self.first_ok
            label = "the first working config of this run"
            old = flags_dict(self.cmd(b["cfg"], self.s.service_port, alias=self.s.alias or None)) if b else {}
        rows, same = diff_flags(old, new)
        perf, reason = None, None
        if b and b["ok"]:
            perf = {"pp": (b["pp"], best["pp"]), "tg": (b["tg"], best["tg"]),
                    "turn": (self.turn_s(b), self.turn_s(best))}
        elif b:
            reason = b["reason"]
        elif cur:
            reason = "not measured"
        return {"label": label, "rows": rows, "same": same, "perf": perf, "reason": reason,
                "depth": self.s.depth}

    def tune_cpu(self):
        s, st, cpu, caps = self.s, self.st, self.cpu, self.caps
        has_fa = bool(self.fa_args())
        st.est = 12 if s.quick else 28
        ctx = s.ctx
        base = {"cpu": True, "host": 0, "ub": 512, "t": cpu["physical"], "tb": 0,
                "ctk": "q8_0" if has_fa else "f16", "ctv": "q8_0" if has_fa else "f16",
                "cache": 0, "spec": False, "ctx": ctx}
        st.stage = "loading the model on the CPU"
        while True:
            base["ctx"] = ctx
            r = self.evaluate(dict(base))
            if r["ok"] or ctx <= 4096:
                break
            ctx = max(4096, ctx // 2)
            st.add(f"does not fit in RAM at the requested context; trying ctx={ctx}")
            st.notes.append(f"context reduced to {ctx}")
        if not r["ok"]:
            raise RuntimeError("the model does not load in RAM. Use a smaller quant or model.")
        cur = r

        st.stage = "tuning threads (-t)"
        cands = {cpu["perf"], cpu["physical"], cpu["logical"], max(1, cpu["perf"] - 2),
                 max(1, cpu["physical"] // 2)}
        sweep = [cur]
        for th in sorted(c for c in cands if 1 <= c <= cpu["logical"]):
            if th != cur["cfg"]["t"]:
                sweep.append(self.evaluate(dict(cur["cfg"], t=th)))
        sweep = [r for r in sweep if r["ok"]]
        cur = self.best_result()
        if caps.has("--threads-batch") and len(sweep) > 1:
            # generation and prefill often like different thread counts: combine the best of each
            t_gen = max(sweep, key=lambda r: r["tg"])["cfg"]["t"]
            t_pre = max(sweep, key=lambda r: r["pp"])["cfg"]["t"]
            if t_gen != t_pre:
                st.stage = "combining generation and prefill threads (-t / -tb)"
                self.evaluate(dict(cur["cfg"], t=t_gen, tb=t_pre))
            cur = self.best_result()

        st.stage = "tuning micro-batch (-ub)"
        for ub in (256, 1024, 2048):
            if ub != cur["cfg"]["ub"]:
                self.evaluate(dict(cur["cfg"], ub=ub))
        cur = self.best_result()
        if s.quick:
            return

        st.stage = "tuning the KV cache type"
        if has_fa:
            self.evaluate(dict(cur["cfg"], ctk="f16", ctv="f16"))
            if s.kv_q4:
                self.evaluate(dict(cur["cfg"], ctk="q8_0", ctv="q4_0"))
            cur = self.best_result()
            st.stage = "trying flash attention off"
            self.evaluate(dict(cur["cfg"], fa=False, ctk="f16", ctv="f16"))
            cur = self.best_result()

        st.stage = "tuning memory mode (mmap / mlock)"
        if caps.has("--no-mmap"):
            self.evaluate(dict(cur["cfg"], mmap=False))
        if caps.has("--mlock"):
            self.evaluate(dict(cur["cfg"], lock=True))
        cur = self.best_result()

        if cpu.get("numa", 1) > 1 and caps.has("--numa"):
            st.stage = "trying NUMA distribute"
            self.evaluate(dict(cur["cfg"], numa=True))
            cur = self.best_result()

        if s.spec and self.minfo and self.minfo.get("nextn") and caps.has("draft-mtp"):
            st.stage = "trying MTP speculative decoding"
            self.evaluate(dict(cur["cfg"], spec=True))

    def tune_cache(self, base):
        st, s = self.st, self.s
        cur = self.best_result()
        c0 = cur["cfg"]
        nl = self.nl
        per_layer = self.per_layer_mib()
        full = nl - c0["host"]
        best_turn, stale = self.turn_s(cur), 0
        for dm in sorted({0, 2, 4, 8, 12, 16, full}):
            if dm < 0 or dm > full:
                continue
            if stale >= 2 and dm != full:
                continue
            host = c0["host"] + dm
            budget = (cur.get("free_min") or 0) - s.margin + dm * per_layer
            cache = int(budget * 0.9) // 256 * 256
            if self.host_mib:
                cache = min(cache, int(self.host_mib * 0.95) // 256 * 256)
            if cache < 512:
                continue
            r = None
            for _ in range(2):
                r = self.evaluate(dict(c0, host=host, cache=cache))
                if r["ok"]:
                    break
                cache = int(cache * 0.75) // 256 * 256
                if cache < 512:
                    break
            if r and r["ok"]:
                if self.turn_s(r) < best_turn:
                    best_turn, stale = self.turn_s(r), 0
                else:
                    stale += 1

    # ------------------------------------------------------------------ outputs
    def final_tokens(self, cfg):
        return self.cmd(cfg, self.s.service_port, alias=self.s.alias or None, host=self.s.host)


# ============================================================================ build helper
def build_job(s, det, st):
    """git clone/pull llama.cpp and build llama-server for this machine's GPU."""
    st.kind = "build"
    try:
        backend, flags, note = suggest_build(det.vendors)
        bindir = Path(s.bin).resolve().parent if s.bin and Path(s.bin).exists() else None
        src = bindir.parent.parent if bindir and bindir.parent.name.startswith("build") else HOME / "llama.cpp"
        st.add(f"backend: {backend}. {note}")
        env = os.environ.copy()
        if backend == "CUDA":
            env["PATH"] = "/opt/cuda/bin:/usr/local/cuda/bin:" + env.get("PATH", "")
            if not shutil.which("nvcc", path=env["PATH"]):
                raise RuntimeError("nvcc not found: install the CUDA toolkit first")
        for tool in ("cmake", "git"):
            if not shutil.which(tool):
                raise RuntimeError(f"{tool} not found: install it first")

        def stream(cmd, cwd=None):
            st.add("$ " + " ".join(cmd))
            p = subprocess.Popen(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            for line in p.stdout:
                if st.abort:
                    p.terminate()
                    raise Aborted()
                st.add(line.rstrip()[:200])
            if p.wait() != 0:
                raise RuntimeError(f"command failed: {' '.join(cmd[:3])}")

        st.stage = "fetching the source"
        if (src / ".git").exists():
            try:
                stream(["git", "pull", "--ff-only"], cwd=src)
            except RuntimeError:
                st.add("git pull failed (local changes?); building the current checkout")
        else:
            stream(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(src)])
        st.stage = "configuring"
        stream(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"] + flags, cwd=src)
        st.stage = "compiling (this takes a while)"
        stream(["cmake", "--build", "build", "--config", "Release", "--target", "llama-server",
                "-j", str(os.cpu_count() or 4)], cwd=src)
        for cand in (src / "build/bin" / BIN_NAME, src / "build/bin/Release" / BIN_NAME):
            if cand.exists():
                s.bin = str(cand)
                break
        st.add("build finished: " + s.bin)
    except Aborted:
        st.add("build aborted")
    except Exception as e:
        st.error = str(e)
        st.add("ERROR " + str(e))
    finally:
        st.stage = "finished"
        st.done = True


# ============================================================================ outputs
def render_unit(tokens, desc="llama-server (tuned by llamatune)", env=None):
    lines, cur = [], []
    for t in tokens:
        t = t.replace("%", "%%")
        if " " in t or '"' in t:
            t = shlex.quote(t)
        if t.startswith("-") and cur:
            lines.append(" ".join(cur))
            cur = [t]
        else:
            cur.append(t)
    if cur:
        lines.append(" ".join(cur))
    return ("[Unit]\nDescription=" + desc + "\nAfter=network-online.target\n\n[Service]\n"
            + "".join(f'Environment="{k}={v}"\n' for k, v in (env or {}).items()) + "ExecStart="
            + " \\\n  ".join(lines) + "\nRestart=on-failure\nRestartSec=5\n\n"
            "[Install]\nWantedBy=default.target\n")


def render_script(tokens, env=None):
    envs = env or {}
    if IS_WIN:
        return ("@echo off\r\n" + "".join(f"set {k}={v}\r\n" for k, v in envs.items())
                + subprocess.list2cmdline(tokens) + "\r\n")
    lines, cur = [], []
    for t in tokens:
        q = shlex.quote(t)
        if t.startswith("-") and cur:
            lines.append(" ".join(cur))
            cur = [q]
        else:
            cur.append(q)
    if cur:
        lines.append(" ".join(cur))
    exports = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in envs.items())
    return "#!/bin/sh\n" + exports + "exec " + " \\\n  ".join(lines) + "\n"


def save_script(tokens, path, env=None):
    path = Path(path).expanduser()
    path.write_text(render_script(tokens, env))
    if not IS_WIN:
        path.chmod(0o755)
    return str(path)


def service_slug(text):
    t = re.sub(r"[^A-Za-z0-9._:-]+", "-", text).strip("-.")
    return t or "model"


def model_label(model, hf=""):
    """Readable model name: 'abc3B.gguf' -> 'abc3B' (case kept, shard suffix and .gguf removed)."""
    if model:
        stem = re.sub(r"\.gguf$", "", Path(model).name, flags=re.I)
        stem = re.sub(r"-\d{5}-of-\d{5}$", "", stem)
    elif hf:
        stem = hf.split("/")[-1].replace(":", "-")
    else:
        stem = "model"
    return service_slug(stem)


def default_service_name(model, hf=""):
    return "llama-server-" + model_label(model, hf)


def normalize_unit_name(name):
    name = name.strip()
    if name.endswith(".service"):
        name = name[:-8]
    return service_slug(name) + ".service"


def find_conflicts(det, unit_name, model, hf):
    """Other units that run the same model (they would fight over the port)."""
    out = []
    for path, parsed in det.units:
        if path.name == unit_name:
            continue
        same = (model and parsed["model"] and Path(parsed["model"]).name == Path(model).name) or \
               (hf and parsed["hf"] == hf)
        if same:
            out.append(path)
    return out


def install_unit(tokens, det, env=None, name=None, model="", hf="", replace_old=False):
    """Write ~/.config/systemd/user/<name>.service (default: llama-server-<model name>.service)."""
    if platform.system() != "Linux":
        raise RuntimeError("systemd units are Linux-only; save a script instead")
    unit_name = normalize_unit_name(name) if name else default_service_name(model, hf) + ".service"
    path = HOME / ".config" / "systemd" / "user" / unit_name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(f".service.bak-{int(time.time())}"))
    path.write_text(render_unit(tokens, desc=f"llama-server ({model_label(model, hf)}, tuned by llamatune)", env=env))
    run(["systemctl", "--user", "daemon-reload"])
    notes = [f"wrote {path}"]
    conflicts = find_conflicts(det, unit_name, model, hf)
    was_active = False
    if replace_old:
        for old in conflicts:
            if run(["systemctl", "--user", "is-active", "--quiet", old.name]).returncode == 0:
                was_active = True
            run(["systemctl", "--user", "stop", old.name])
            run(["systemctl", "--user", "disable", old.name])
            notes.append(f"{old.name} stopped and disabled (its file is kept)")
    elif conflicts:
        notes.append("WARNING: " + ", ".join(c.name for c in conflicts) + " runs the same model and will "
                     "fight over the port. Disable it before starting the new one.")
    if run(["systemctl", "--user", "is-active", "--quiet", unit_name]).returncode == 0:
        run(["systemctl", "--user", "restart", unit_name])
        notes.append(f"{unit_name} restarted")
    elif was_active:
        run(["systemctl", "--user", "enable", "--now", unit_name])
        notes.append(f"{unit_name} enabled and started")
    else:
        notes.append(f"start it with: systemctl --user enable --now {unit_name}")
    det.units = find_units()
    return "; ".join(notes)


def save_json(engine, path):
    st = engine.st
    best = st.best
    rows = [{"cfg": r["cfg"], "pp": r["pp"], "tg": r["tg"], "turn_s": engine.turn_s(r)}
            for r in sorted([r for r in st.results if r["ok"]], key=engine.turn_s)]
    data = {"best": best["cfg"] if best else None, "command": engine.final_tokens(best["cfg"]) if best else None,
            "goal": engine.s.goal, "diff": engine.diff(), "results": rows}
    Path(path).expanduser().write_text(json.dumps(data, indent=2))
    return str(Path(path).expanduser())


def results_rows(engine):
    seen, rows = set(), []
    for r in sorted([r for r in engine.st.results if r["ok"]], key=engine.turn_s):
        k = json.dumps(norm(r["cfg"]), sort_keys=True)
        if k not in seen:
            seen.add(k)
            rows.append(r)
    return rows


# ============================================================================ debug report
def redact(text):
    text = text.replace(str(HOME), "~")
    user = os.environ.get("USER") or os.environ.get("USERNAME") or HOME.name
    if user and len(user) > 2:
        text = re.sub(r"\b" + re.escape(user) + r"\b", "<user>", text)

    def ip(m):
        v = m.group(0)
        return v if v.startswith(("127.", "0.0.0.0")) else "x.x.x.x"
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", ip, text)
    text = re.sub(r"hf_[A-Za-z0-9]{10,}", "hf_<redacted>", text)
    text = re.sub(r"(api[-_ ]?key[\"'=: ]+)\S+", r"\1<redacted>", text, flags=re.I)
    return text


def write_debug_report(s, det, path):
    """One text file with everything needed to diagnose a problem. Paths, user name, IPs and tokens are redacted."""
    out = []

    def sec(title, body):
        out.append(f"\n===== {title} =====\n{body}")

    sec("llamatune", f"{VERSION}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\npython {sys.version.split()[0]} on "
        f"{platform.platform()}\nstate dir: {STATE_DIR}")
    sec("settings", json.dumps({k: v for k, v in vars(s).items()}, indent=1, default=str))
    sec("detection", json.dumps(det.report(s), indent=1, default=str))
    if s.bin and Path(s.bin).exists():
        flags = ["--n-cpu-moe", "--override-tensor", "--moe-cache-mib", "--moe-cache-cpu-share", "--spec-type", "draft-mtp", "--cache-reuse",
                 "--fit", "--device", "--no-op-offload", "--threads-batch", "--no-mmap", "--mlock", "--numa",
                 "--flash-attn", "--cache-type-k", "--models-preset", "--models-max"]
        caps = det.caps or Caps(s.bin)
        sec("binary", f"{s.bin}\nversion: {caps.version}\nbackends: {caps.backends}\nflags in --help:\n"
            + "\n".join(f"  {'yes' if caps.has(f) else 'NO '}  {f}" for f in flags))
        sec("help lines mentioning moe-cache",
            "\n".join(l for l in caps.help.splitlines() if "moe-cache" in l.lower()) or "(none)")
        r = run([s.bin, "--list-devices"], timeout=60)
        sec("llama-server --list-devices", (r.stdout + r.stderr).strip())
    if shutil.which("nvidia-smi"):
        sec("nvidia-smi", run(["nvidia-smi"]).stdout[:2500])
    ld = STATE_DIR / "last-diff.txt"
    if ld.exists():
        sec("config changes in the last run", ld.read_text(errors="ignore"))
    lr = STATE_DIR / "last-run.log"
    if lr.exists():
        sec("last run (log tail)", "\n".join(lr.read_text(errors="ignore").splitlines()[-150:]))
    caches = sorted(STATE_DIR.glob("cache-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if caches:
        try:
            data = json.loads(caches[0].read_text())
            rows = []
            for r in list(data.values())[-60:]:
                c = r.get("cfg", {})
                rows.append(f"{'OK  ' if r.get('ok') else 'FAIL'} {describe(c) if c else '?'}  "
                            f"{'pp=%.0f tg=%.1f' % (r['pp'], r['tg']) if r.get('ok') else r.get('reason')}  [{r.get('log', '')}]")
            sec(f"results cache ({caches[0].name})", "\n".join(rows))
        except Exception as e:
            sec("results cache", f"unreadable: {e}")
    logs = sorted((STATE_DIR / "logs").glob("probe-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:6]
    for lg in logs:
        lines = lg.read_text(errors="ignore").splitlines()
        body = "\n".join(lines[:10] + (["   [...]"] + lines[-20:] if len(lines) > 30 else lines[10:]))
        sec(f"server log {lg.name}", body)
    profs = load_profiles()
    if profs:
        sec("saved profiles", "\n".join(f"{v['name']}: pp={v['pp']:.0f} tg={v['tg']:.1f} {' '.join(v['tokens'][1:])}"
                                        for v in profs.values()))
    text = redact("\n".join(out)) + "\n"
    Path(path).expanduser().write_text(text)
    return str(Path(path).expanduser())


# ============================================================================ profiles & export
PROFILES = STATE_DIR / "profiles.json"
SHORT_TO_LONG = {
    "-c": "ctx-size", "-ngl": "n-gpu-layers", "-fa": "flash-attn", "-ctk": "cache-type-k",
    "-ctv": "cache-type-v", "-b": "batch-size", "-ub": "ubatch-size", "-t": "threads",
    "-tb": "threads-batch", "-np": "parallel", "-m": "model", "-hf": "hf-repo",
    "-ot": "override-tensor", "-ncmoe": "n-cpu-moe", "-cmoe": "cpu-moe", "-dev": "device",
}
ROUTER_OWNED = {"host", "port", "alias"}  # the router / proxy decides these


def slug(text):
    t = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower()
    return t or "model"


def profile_name(s):
    if s.alias:
        return s.alias
    return slug(Path(s.model).stem if s.model else s.hf)


def load_profiles():
    try:
        return json.loads(PROFILES.read_text())
    except Exception:
        return {}


def save_profiles(data):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES.write_text(json.dumps(data, indent=2))


def save_profile(engine):
    """Remember the best config of a finished run, one profile per model."""
    st, s = engine.st, engine.s
    best = st.best
    if not best or st.abort:
        return None
    data = load_profiles()
    key = s.model or s.hf
    name = profile_name(s)
    taken = {v["name"] for k, v in data.items() if k != key}
    base, n = name, 2
    while name in taken:
        name, n = f"{base}-{n}", n + 1
    data[key] = {
        "name": name, "model": s.model, "hf": s.hf, "tokens": engine.final_tokens(best["cfg"]),
        "env": engine.final_env(), "goal": s.goal, "pp": best["pp"], "tg": best["tg"],
        "turn_s": engine.turn_s(best), "ctx": best["cfg"]["ctx"], "cpu_only": s.cpu_only,
        "tuned_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    save_profiles(data)
    return name


def split_flags(tokens):
    """tokens -> (binary, [(flag, value-or-None), ...]) without host/port/alias."""
    pairs, i = [], 1
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        has_val = nxt is not None and (not nxt.startswith("-") or re.match(r"-\d", nxt) is not None)
        if t in ("-fa", "--flash-attn"):
            has_val = nxt in ("on", "off", "auto", "1", "0")
        long = SHORT_TO_LONG.get(t, t.lstrip("-"))
        if long not in ROUTER_OWNED:
            pairs.append((t, nxt if has_val else None))
        i += 2 if has_val else 1
    return tokens[0], pairs


DIFF_ORDER = ["n-cpu-moe", "n-gpu-layers", "cpu-moe", "moe-cache-mib", "moe-cache-cpu-share", "ctx-size", "ubatch-size", "batch-size",
              "threads", "threads-batch", "cache-type-k", "cache-type-v", "flash-attn", "spec-type",
              "no-mmap", "mlock", "numa", "device", "cache-reuse", "fit"]


def flags_dict(tokens):
    """Command line -> {long flag name: value or True}, ignoring host/port/alias/model."""
    flat = []
    for t in tokens:
        if t.startswith("--") and "=" in t:
            flat += t.split("=", 1)
        else:
            flat.append(t)
    _, pairs = split_flags(flat)
    d = {}
    for flag, val in pairs:
        long = SHORT_TO_LONG.get(flag, flag.lstrip("-"))
        if long in ("model", "hf-repo"):
            continue
        if long == "flash-attn" and val is None:
            val = "on"
        d[long] = True if val is None else val
    return d


def diff_flags(old, new):
    def fmt(v):
        return "(unset)" if v is None else ("set" if v is True else str(v))
    rows, same = [], []
    keys = sorted(set(old) | set(new),
                  key=lambda k: (DIFF_ORDER.index(k) if k in DIFF_ORDER else len(DIFF_ORDER), k))
    for k in keys:
        o, n = old.get(k), new.get(k)
        if o == n:
            same.append(k)
        else:
            rows.append((k, fmt(o), fmt(n)))
    return rows, same


def diff_lines(d):
    """Human-readable text for the 'what changed' report (TUI, headless, debug report)."""
    out = [f"Changes vs {d['label']}:"]
    if not d["rows"]:
        out.append("  (none: the tuned command equals the baseline)")
    for k, o, n in d["rows"]:
        out.append(f"  --{k:<18} {o}  ->  {n}")
    if d["same"]:
        out.append("  unchanged: " + ", ".join("--" + k for k in d["same"]))
    out.append("")
    if d["perf"]:
        out.append(f"Speed at a {d['depth']}-token prompt (before -> after):")
        for key, label, unit, up in (("pp", "prefill", "t/s", True), ("tg", "generation", "t/s", True),
                                     ("turn", "one turn", "s", False)):
            a, b = d["perf"][key]
            pct = (b / a - 1.0) * 100.0 if a else 0.0
            better = (pct > 0) == up
            out.append(f"  {label:<11} {a:>8.1f} -> {b:>8.1f} {unit:<4} {pct:+.0f}% {'(better)' if better and abs(pct) >= 1 else ''}".rstrip())
    elif d.get("reason"):
        out.append(f"Your current config could not be compared: {d['reason']}")
    return out


def diff_summary(d):
    if not d or not d["perf"]:
        return ""
    pp0, pp1 = d["perf"]["pp"]
    tg0, tg1 = d["perf"]["tg"]
    return (f"vs {d['label']}: generation {(tg1 / tg0 - 1) * 100:+.0f}%, prefill {(pp1 / pp0 - 1) * 100:+.0f}%"
            if pp0 and tg0 else "")


def export_swap(profiles):
    """llama-swap config: one model entry per tuned profile, each with its own command."""
    out = ["# generated by llamatune. Check the llama-swap README for the full list of keys.",
           "healthCheckTimeout: 900", "models:"]
    for prof in profiles:
        binp, pairs = split_flags(prof["tokens"])
        parts = [shlex.quote(binp), "--host", "127.0.0.1", "--port", "${PORT}"]
        for flag, val in pairs:
            parts.append(shlex.quote(flag))
            if val is not None:
                parts.append(shlex.quote(val))
        out.append(f'  "{prof["name"]}":')
        out.append(f"    # tuned {prof['tuned_at']} for goal '{prof['goal']}': "
                   f"pp {prof['pp']:.0f} t/s, tg {prof['tg']:.1f} t/s at ctx {prof['ctx']}")
        out.append("    cmd: >-")
        out.append("      " + " ".join(parts))
        out.append("    proxy: http://127.0.0.1:${PORT}")
        if prof.get("env"):
            out.append("    env:")
            for k, v in prof["env"].items():
                out.append(f'      - "{k}={v}"')
    return "\n".join(out) + "\n"


def export_router(profiles):
    """llama-server router-mode preset file (section name = model id in the API)."""
    out = ["; generated by llamatune",
           "; run:  llama-server --models-preset THIS_FILE --models-max 1 --port 8080",
           "; (--models-max 1 keeps one model in memory at a time, right for a single GPU)", "",
           "[*]", "; global options for every model go here", ""]
    for prof in profiles:
        _, pairs = split_flags(prof["tokens"])
        out.append(f"[{prof['name']}]")
        out.append(f"; tuned {prof['tuned_at']} for goal '{prof['goal']}': "
                   f"pp {prof['pp']:.0f} t/s, tg {prof['tg']:.1f} t/s at ctx {prof['ctx']}")
        for flag, val in pairs:
            long = SHORT_TO_LONG.get(flag, flag.lstrip("-"))
            if val is None:
                out.append(f"{long[3:]} = false" if long.startswith("no-") else f"{long} = true")
            else:
                out.append(f"{long} = {val}")
        if prof.get("env"):
            out.append("; NOTE: this build needs these environment variables set for the router: "
                       + ", ".join(f"{k}={v!r}" for k, v in prof["env"].items()))
        out.append("")
    return "\n".join(out)


def write_export(kind, path):
    profs = list(load_profiles().values())
    if not profs:
        raise RuntimeError("no tuned models yet: run a tuning first")
    text = export_swap(profs) if kind == "swap" else export_router(profs)
    Path(path).expanduser().write_text(text)
    return f"{len(profs)} model(s) written to {Path(path).expanduser()}"


# ============================================================================ TUI
class App:
    FIELDS = [
        ("bin", "llama-server binary", "path"),
        ("model", "Model (.gguf or HF spec)", "model"),
        ("ctx", "Context needed (-c)", "int"),
        ("cpu_only", "CPU-only mode (ignore GPU)", "bool"),
        ("goal", "Optimise for", "goal"),
        ("depth", "Measure at prompt depth", "int"),
        ("compare_current", "Measure my current config", "bool"),
        ("margin", "Keep free VRAM/RAM (MiB)", "int"),
        ("moe_cache", "GPU expert cache stage", "bool"),
        ("spec", "MTP speculative stage", "bool"),
        ("kv_q4", "Allow q4 V cache", "bool"),
        ("quick", "Quick mode (skip -ub/-t)", "bool"),
        ("service_name", "Service name (blank = auto)", "text"),
    ]

    def __init__(self, s, det):
        self.s, self.det = s, det
        self.mode = "setup"
        self.sel = 0
        self.msg = ""
        self.st = None
        self.engine = None
        self.thread = None
        self.scroll = 0

    # ---- helpers
    def field_value(self, key):
        s, d = self.s, self.det
        if key == "model":
            return s.model or s.hf or "(none)"
        if key == "goal":
            return f"{s.goal}: {GOALS[s.goal][2]}"
        if key == "service_name":
            return normalize_unit_name(s.service_name) if s.service_name else \
                "auto: " + default_service_name(s.model, s.hf) + ".service"
        v = getattr(s, key)
        if isinstance(v, bool):
            avail = ""
            if key == "moe_cache" and d.caps and not d.caps.has("--moe-cache-mib"):
                avail = "  (not in this build)"
            if key == "spec" and d.minfo and not d.minfo.get("nextn"):
                avail = "  (model has no MTP block)"
            return ("yes" if v else "no") + avail
        return str(v)


def safe(win, y, x, text, attr=0):
    try:
        h, w = win.getmaxyx()
        if 0 <= y < h and x < w:
            win.addstr(y, x, text[:max(0, w - x - 1)], attr)
    except curses.error:
        pass


def prompt_input(scr, label, initial=""):
    buf = list(initial)
    while True:
        h, w = scr.getmaxyx()
        safe(scr, h - 1, 0, " " * (w - 1))
        safe(scr, h - 1, 0, f"{label}: {''.join(buf)}_", curses.A_BOLD)
        scr.refresh()
        scr.timeout(-1)
        ch = scr.getch()
        if ch in (10, 13):
            scr.timeout(250)
            return "".join(buf)
        if ch == 27:
            scr.timeout(250)
            return None
        if ch in (curses.KEY_BACKSPACE, 127, 8):
            if buf:
                buf.pop()
        elif ch == 21:  # ctrl-u
            buf = []
        elif 32 <= ch < 256:
            buf.append(chr(ch))


def pick_menu(scr, title, items, current=0):
    idx = current
    while True:
        scr.erase()
        h, w = scr.getmaxyx()
        safe(scr, 0, 0, title, curses.A_BOLD)
        top = max(0, min(idx - (h - 4) // 2, len(items) - (h - 3)))
        for i, it in enumerate(items[top:top + h - 3]):
            real = top + i
            safe(scr, 2 + i, 2, ("> " if real == idx else "  ") + it,
                 curses.A_REVERSE if real == idx else 0)
        safe(scr, h - 1, 0, "up/down move, Enter select, Esc cancel")
        scr.refresh()
        scr.timeout(-1)
        ch = scr.getch()
        if ch == curses.KEY_UP:
            idx = max(0, idx - 1)
        elif ch == curses.KEY_DOWN:
            idx = min(len(items) - 1, idx + 1)
        elif ch in (10, 13):
            scr.timeout(250)
            return idx
        elif ch == 27:
            scr.timeout(250)
            return None


def fmt_gib(mib_val):
    return "?" if mib_val is None else f"{mib_val / 1024:.1f} GiB"


def draw_setup(scr, app, colors):
    s, d = app.s, app.det
    h, w = scr.getmaxyx()
    scr.erase()
    safe(scr, 0, 0, " llamatune - llama.cpp auto-tuner ", curses.A_REVERSE | curses.A_BOLD)
    y = 2
    safe(scr, y, 0, "Detected", colors["head"]); y += 1
    cpu, ram = d.cpu, d.ram
    safe(scr, y, 2, f"CPU  {cpu['name'][:44]}  {cpu['perf']}P/{cpu['physical']}C/{cpu['logical']}T"); y += 1
    safe(scr, y, 2, f"RAM  {fmt_gib(ram['total'])} total, {fmt_gib(ram['avail'])} available"); y += 1
    if d.gpus:
        for g in d.gpus[:3]:
            safe(scr, y, 2, f"GPU  {g['name'][:40]}  {fmt_gib(g['total'])}, {fmt_gib(g['free'])} free"); y += 1
    else:
        safe(scr, y, 2, "GPU  none usable by llama.cpp" + (f" (hardware: {', '.join(d.vendors)})" if d.vendors else "")); y += 1
    if d.caps:
        safe(scr, y, 2, f"llama.cpp  {d.caps.version[:30]}  backends: {', '.join(d.caps.backends) or 'CPU only'}"); y += 1
    mi = d.minfo
    if mi:
        kind = f"MoE {mi['n_expert']} experts" if mi["moe"] else "dense"
        mtp = ", MTP block" if mi.get("nextn") else ""
        safe(scr, y, 2, f"Model  {mi['arch']}, {mi['n_layer']} layers, {kind}{mtp}, "
                        f"{fmt_gib(mib(mi['total_bytes']))}, trained ctx {mi['ctx_train']}"); y += 1
    elif s.hf:
        safe(scr, y, 2, f"Model  {s.hf} (downloaded on first load)"); y += 1
    y += 1
    profs = load_profiles()
    if profs:
        names = ", ".join(v["name"] for v in profs.values())
        safe(scr, y, 2, f"Tuned models saved: {len(profs)} ({names})"[:w - 3]); y += 1
    y += 0
    safe(scr, y, 0, "Settings   (up/down select, Enter change)", colors["head"]); y += 1
    for i, (key, label, _) in enumerate(App.FIELDS):
        attr = curses.A_REVERSE if i == app.sel else 0
        safe(scr, y, 2, f"{label:<28} {app.field_value(key)}"[:w - 3], attr); y += 1
    y += 1
    for level, text in d.warnings(s):
        safe(scr, y, 0, ("! " if level == "warn" else "x ") + text,
             colors["warn"] if level == "warn" else colors["err"]); y += 1
    if app.msg:
        safe(scr, y + 1, 0, app.msg, colors["ok"])
    safe(scr, h - 1, 0, " r tune | b build | d re-detect | w llama-swap | p router INI | g debug | q quit ", curses.A_REVERSE)
    scr.refresh()


def draw_run(scr, app, colors):
    st, eng = app.st, app.engine
    h, w = scr.getmaxyx()
    scr.erase()
    el = int(time.time() - st.t0)
    title = "building llama.cpp" if st.kind == "build" else "tuning"
    safe(scr, 0, 0, f" llamatune - {title}   {el // 60:02d}:{el % 60:02d} ", curses.A_REVERSE | curses.A_BOLD)
    safe(scr, 2, 0, f"Stage: {st.stage}", colors["head"])
    if st.kind == "tune":
        frac = min(1.0, st.probes / max(1, st.est))
        bw = max(10, min(40, w - 30))
        safe(scr, 3, 0, f"[{'#' * int(bw * frac)}{'.' * (bw - int(bw * frac))}] {st.probes} probes (about {st.est} total)")
    safe(scr, 4, 0, ("Now: " + st.current) if st.current else "")
    y = 6
    if eng and st.kind == "tune":
        rows = results_rows(eng)[:6]
        if rows:
            safe(scr, y, 0, "Best configs so far (estimated turn time)", colors["head"]); y += 1
            for i, r in enumerate(rows):
                safe(scr, y, 2, f"{'*' if i == 0 else ' '} {describe(r['cfg'])}  pp={r['pp']:.0f} tg={r['tg']:.1f}  turn={eng.turn_s(r):.1f}s",
                     colors["ok"] if i == 0 else 0); y += 1
            y += 1
    safe(scr, y, 0, "Log", colors["head"]); y += 1
    with st.lock:
        lines = list(st.log)
    room = h - y - 1
    for ln in lines[-room:] if room > 0 else []:
        attr = colors["err"] if " FAIL " in ln or "ERROR" in ln else (colors["ok"] if " OK " in ln else 0)
        safe(scr, y, 0, ln, attr); y += 1
    safe(scr, h - 1, 0, " q abort " if not st.done else " finished ", curses.A_REVERSE)
    scr.refresh()


def draw_diff(scr, app, colors):
    h, w = scr.getmaxyx()
    scr.erase()
    safe(scr, 0, 0, " llamatune - what changed ", curses.A_REVERSE | curses.A_BOLD)
    d = app.engine.diff() if app.engine else None
    if not d:
        safe(scr, 2, 0, "Nothing to compare yet.")
    else:
        y = 2
        for ln in diff_lines(d):
            if y >= h - 2:
                break
            attr = 0
            if ln.startswith("Changes") or ln.startswith("Speed"):
                attr = colors["head"]
            elif "->" in ln and ln.startswith("  --"):
                attr = colors["warn"]
            elif "(better)" in ln:
                attr = colors["ok"]
            elif "could not" in ln:
                attr = colors["err"]
            safe(scr, y, 0, ln, attr)
            y += 1
    safe(scr, h - 1, 0, " c back to results | l log | q quit ", curses.A_REVERSE)
    scr.refresh()


def draw_results(scr, app, colors):
    st, eng, s = app.st, app.engine, app.s
    h, w = scr.getmaxyx()
    scr.erase()
    safe(scr, 0, 0, " llamatune - results ", curses.A_REVERSE | curses.A_BOLD)
    y = 2
    if st.error:
        safe(scr, y, 0, "Error: " + st.error, colors["err"]); y += 2
    best = st.best
    if not best:
        safe(scr, y, 0, "No working configuration was found. See the log (l).", colors["err"])
    else:
        rows = results_rows(eng)
        safe(scr, y, 0, f"Goal: {s.goal}.  Ranked by estimated time of one turn.", colors["head"]); y += 1
        for i, r in enumerate(rows[:max(3, h - 16)]):
            safe(scr, y, 2, f"{'*' if i == 0 else ' '} {describe(r['cfg'])}  pp={r['pp']:.0f} tg={r['tg']:.1f}  turn={eng.turn_s(r):.1f}s",
                 colors["ok"] if i == 0 else 0); y += 1
        for n in st.notes:
            safe(scr, y, 0, "note: " + n, colors["warn"]); y += 1
        summ = diff_summary(eng.diff())
        if summ:
            safe(scr, y, 0, summ + "  (press c for the full list of changes)", colors["ok"]); y += 1
        y += 1
        safe(scr, y, 0, "Best command", colors["head"]); y += 1
        text = " ".join(shlex.quote(t) for t in eng.final_tokens(best["cfg"]))
        for i in range(0, len(text), max(20, w - 3)):
            safe(scr, y, 1, text[i:i + w - 3]); y += 1
            if y > h - 4:
                break
    if app.msg:
        safe(scr, h - 3, 0, app.msg, colors["ok"])
    safe(scr, h - 1, 0, " s script | u systemd | j JSON | c changes | w swap | p router | g debug | l log | n new | q quit ", curses.A_REVERSE)
    scr.refresh()


def tui(stdscr, s, det):
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    colors = {"head": curses.A_BOLD, "ok": 0, "warn": curses.A_BOLD, "err": curses.A_BOLD}
    if curses.has_colors():
        curses.start_color()
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = curses.COLOR_BLACK
        curses.init_pair(1, curses.COLOR_GREEN, bg)
        curses.init_pair(2, curses.COLOR_RED, bg)
        curses.init_pair(3, curses.COLOR_YELLOW, bg)
        curses.init_pair(4, curses.COLOR_CYAN, bg)
        colors = {"ok": curses.color_pair(1), "err": curses.color_pair(2) | curses.A_BOLD,
                  "warn": curses.color_pair(3), "head": curses.color_pair(4) | curses.A_BOLD}
    app = App(s, det)
    stdscr.timeout(250)
    stdscr.keypad(True)
    show_log = False
    show_diff = False
    while True:
        if app.mode == "setup":
            draw_setup(stdscr, app, colors)
        elif app.mode == "run":
            draw_run(stdscr, app, colors)
            if app.st.done:
                if app.st.kind == "build":
                    det.refresh(s)
                    app.msg = "build finished" if not app.st.error else "build failed: " + app.st.error
                    app.mode = "setup"
                else:
                    app.mode = "results"
                    app.msg = ""
        else:
            if show_diff:
                draw_diff(stdscr, app, colors)
            elif show_log:
                draw_run(stdscr, app, colors)
            else:
                draw_results(stdscr, app, colors)
        ch = stdscr.getch()
        if ch == -1:
            continue
        if app.mode == "setup":
            if ch in (ord("q"), 27):
                return
            elif ch == curses.KEY_UP:
                app.sel = (app.sel - 1) % len(App.FIELDS)
            elif ch == curses.KEY_DOWN:
                app.sel = (app.sel + 1) % len(App.FIELDS)
            elif ch in (10, 13, curses.KEY_RIGHT, ord(" ")):
                edit_field(stdscr, app)
            elif ch == ord("d"):
                det.refresh(s)
                app.msg = "re-detected"
            elif ch in (ord("w"), ord("p")):
                do_export(stdscr, app, "swap" if ch == ord("w") else "router")
            elif ch == ord("g"):
                app.msg = "debug report: " + write_debug_report(s, det, HOME / "llamatune-debug.txt")
            elif ch == ord("b"):
                app.st = RunState()
                app.engine = None
                app.thread = threading.Thread(target=build_job, args=(s, det, app.st), daemon=True)
                app.thread.start()
                app.mode = "run"
            elif ch == ord("r"):
                errs = [t for lv, t in det.warnings(s) if lv == "err"]
                if errs:
                    app.msg = errs[0]
                    continue
                app.st = RunState()
                app.engine = Engine(s, det, app.st)
                app.thread = threading.Thread(target=app.engine.run, daemon=True)
                app.thread.start()
                app.mode = "run"
                app.msg = ""
        elif app.mode == "run":
            if ch in (ord("q"), 27):
                app.st.abort = True
                app.st.add("abort requested...")
        else:
            if ch == ord("q"):
                return
            elif ch == ord("l"):
                show_log = not show_log
                show_diff = False
            elif ch == ord("c"):
                show_diff = not show_diff
                show_log = False
            elif ch == ord("n"):
                app.mode = "setup"
                app.msg = ""
            elif ch in (ord("w"), ord("p")):
                do_export(stdscr, app, "swap" if ch == ord("w") else "router")
            elif ch == ord("g"):
                app.msg = "debug report: " + write_debug_report(s, det, HOME / "llamatune-debug.txt")
            elif app.st.best:
                toks = app.engine.final_tokens(app.st.best["cfg"])
                try:
                    if ch == ord("s"):
                        p = prompt_input(stdscr, "Save script to", str(HOME / f"llama-server-{profile_name(s)}.sh"))
                        if p:
                            app.msg = "saved " + save_script(toks, p, app.engine.final_env())
                    elif ch == ord("u"):
                        app.msg = ask_install_unit(stdscr, app, toks)
                    elif ch == ord("j"):
                        app.msg = "saved " + save_json(app.engine, HOME / "llamatune-result.json")
                except Exception as e:
                    app.msg = f"failed: {e}"


def ask_install_unit(scr, app, toks):
    s, det = app.s, app.det
    default = (normalize_unit_name(s.service_name) if s.service_name
               else default_service_name(s.model, s.hf) + ".service")
    name = prompt_input(scr, "Service name (Enter accepts, or type your own)", default)
    if not name:
        return "install cancelled"
    unit = normalize_unit_name(name)
    replace = False
    conflicts = find_conflicts(det, unit, s.model, s.hf)
    if conflicts:
        v = prompt_input(scr, f"{', '.join(c.name for c in conflicts)} runs the same model. Stop and disable it? (y/n)", "y")
        replace = bool(v and v.strip().lower().startswith("y"))
    return install_unit(toks, det, app.engine.final_env(), name=unit, model=s.model, hf=s.hf, replace_old=replace)


def do_export(scr, app, kind):
    default = HOME / ("llama-swap.yaml" if kind == "swap" else "llamatune-models.ini")
    p = prompt_input(scr, "Write to", str(default))
    if p:
        try:
            app.msg = write_export(kind, p)
        except Exception as e:
            app.msg = f"failed: {e}"


def edit_field(scr, app):
    s, det = app.s, app.det
    key, label, kind = App.FIELDS[app.sel]
    if kind == "bool":
        setattr(s, key, not getattr(s, key))
        if key == "cpu_only":  # CPU prefill is slow: use a shorter measurement prompt
            if s.cpu_only and s.depth == 8192:
                s.depth = 2048
            elif not s.cpu_only and s.depth == 2048:
                s.depth = 8192
    elif kind == "goal":
        names = list(GOALS)
        s.goal = names[(names.index(s.goal) + 1) % len(names)]
    elif kind == "text":
        v = prompt_input(scr, label + " (empty = auto)", getattr(s, key))
        if v is not None:
            setattr(s, key, v.strip())
    elif kind == "int":
        v = prompt_input(scr, label, str(getattr(s, key)))
        if v and v.strip().isdigit():
            setattr(s, key, int(v.strip()))
    elif kind == "path":
        if key == "bin":
            items = det.bins + ["(type a path...)"]
            i = pick_menu(scr, "llama-server binary", items,
                          items.index(s.bin) if s.bin in items else 0)
            if i is None:
                return
            path = det.bins[i] if i < len(det.bins) else prompt_input(scr, "Path", s.bin)
            if path:
                s.bin = path
                det.refresh(s)
    elif kind == "model":
        items = [f"{m['path']}  ({fmt_gib(mib(m['size']))})" for m in det.models] + ["(type a path or user/repo:quant...)"]
        i = pick_menu(scr, "Model", items)
        if i is None:
            return
        if i < len(det.models):
            s.model, s.hf = det.models[i]["path"], ""
        else:
            v = prompt_input(scr, "Path or HF spec", s.model or s.hf)
            if v:
                if Path(v).expanduser().exists():
                    s.model, s.hf = str(Path(v).expanduser()), ""
                else:
                    s.model, s.hf = "", v
        det.load_model(s)
        if det.minfo and det.minfo["ctx_train"]:
            s.ctx = min(s.ctx, det.minfo["ctx_train"])


# ============================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="Auto-tune llama-server (TUI by default).")
    ap.add_argument("--bin")
    ap.add_argument("--model")
    ap.add_argument("--hf")
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--margin", type=int)
    ap.add_argument("--goal", choices=list(GOALS))
    ap.add_argument("--runs", type=int)
    ap.add_argument("--port", type=int)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cpu-only", action="store_true", help="ignore the GPU and tune for CPU inference")
    ap.add_argument("--kv-q4", action="store_true")
    ap.add_argument("--no-moe-cache", action="store_true")
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--no-baseline", action="store_true", help="do not measure your current config first")
    ap.add_argument("--fresh", action="store_true", help="ignore cached results")
    ap.add_argument("--detect", action="store_true", help="print detection as JSON and exit")
    ap.add_argument("--profiles", action="store_true", help="list the saved per-model profiles")
    ap.add_argument("--debug-report", nargs="?", const="", default=None, metavar="PATH",
                    help="write a redacted diagnostic file (default ~/llamatune-debug.txt) and exit")
    ap.add_argument("--max-probes", type=int, help="stop after this many server loads (good for a first test)")
    ap.add_argument("--forget", metavar="NAME", help="delete a saved profile")
    ap.add_argument("--export-swap", metavar="PATH", help="write a llama-swap YAML for all saved profiles")
    ap.add_argument("--export-router", metavar="PATH", help="write a llama-server router preset INI for all saved profiles")
    ap.add_argument("--sweep", action="append", default=[], metavar="FLAG=V1,V2,..",
                    help="also try these values of any llama-server flag, e.g. "
                         "moe-cache-cpu-share=0.25,0.5,0.75 (repeatable; leading -- is optional)")
    ap.add_argument("--headless", action="store_true", help="no TUI; print progress")
    ap.add_argument("--save-script", metavar="PATH")
    ap.add_argument("--install-unit", action="store_true", help="headless: write the systemd user unit")
    ap.add_argument("--unit-name", metavar="NAME", help="custom service name (default: llama-server-<model name>)")
    ap.add_argument("--replace-old-unit", action="store_true",
                    help="with --install-unit: stop and disable other units that run the same model")
    a = ap.parse_args()

    if a.profiles or a.forget or a.export_swap or a.export_router:
        data = load_profiles()
        if a.forget:
            gone = [k for k, v in data.items() if v["name"] == a.forget]
            for k in gone:
                del data[k]
            save_profiles(data)
            print(f"removed {len(gone)} profile(s)")
        if a.profiles:
            for v in data.values():
                print(f"{v['name']:<28} {v['tuned_at']}  goal={v['goal']}  pp={v['pp']:.0f} tg={v['tg']:.1f}  {v['model'] or v['hf']}")
            if not data:
                print("no saved profiles")
        try:
            if a.export_swap:
                print(write_export("swap", a.export_swap))
            if a.export_router:
                print(write_export("router", a.export_router))
        except RuntimeError as e:
            print(e, file=sys.stderr)
            return 1
        return 0

    s = Settings()
    for k in ("bin", "model", "hf", "ctx", "depth", "margin", "goal", "runs", "port"):
        if getattr(a, k) is not None:
            setattr(s, k, getattr(a, k))
    s.quick, s.kv_q4, s.fresh = a.quick, a.kv_q4, a.fresh
    if a.max_probes:
        s.max_probes = a.max_probes
    s.cpu_only = a.cpu_only
    s.service_name = a.unit_name or ""
    s.compare_current = not a.no_baseline
    s.moe_cache, s.spec = not a.no_moe_cache, not a.no_spec
    for spec in a.sweep:
        name, sep, vals = spec.partition("=")
        vals = [v.strip() for v in vals.split(",") if v.strip()]
        if not sep or not vals or not name.strip("-"):
            ap.error(f"--sweep wants FLAG=V1,V2,... (got {spec!r})")
        s.sweep.append(("--" + name.lstrip("-"), vals))

    det = Detection()
    apply_existing_config(s, det)
    det.refresh(s)
    pick_default_model(s, det)
    if det.minfo and det.minfo["ctx_train"] and a.ctx is None:
        s.ctx = min(s.ctx, det.minfo["ctx_train"])
    if not s.cpu_only and det.caps and not det.caps.gpu_ok and not det.vendors:
        s.cpu_only = True  # no GPU on this machine: CPU mode automatically
    if s.cpu_only and a.depth is None and s.depth == 8192:
        s.depth = 2048

    if a.detect:
        print(json.dumps(det.report(s), indent=2, default=str))
        return 0
    if a.debug_report is not None:
        path = write_debug_report(s, det, a.debug_report or str(HOME / "llamatune-debug.txt"))
        print(f"wrote {path}\nIt is redacted (home path, user name, IP addresses, tokens). Attach or paste it.")
        return 0

    use_tui = curses is not None and sys.stdout.isatty() and sys.stdin.isatty() and not a.headless
    if use_tui:
        curses.wrapper(tui, s, det)
        return 0

    # ---- headless
    errs = [t for lv, t in det.warnings(s) if lv == "err"]
    if errs:
        print("cannot start: " + errs[0], file=sys.stderr)
        return 2
    st = RunState(echo=True)
    eng = Engine(s, det, st)
    try:
        eng.run()
    except KeyboardInterrupt:
        st.abort = True
    best = st.best
    if not best:
        print("no working configuration found", file=sys.stderr)
        return 1
    print("\nRanked results (estimated turn time):")
    for r in results_rows(eng)[:10]:
        print(f"  {describe(r['cfg'])}  pp={r['pp']:.0f} tg={r['tg']:.1f}  turn={eng.turn_s(r):.1f}s")
    d = eng.diff()
    if d:
        print("\n" + "\n".join(diff_lines(d)))
    toks = eng.final_tokens(best["cfg"])
    print("\nBest command:\n  " + " ".join(shlex.quote(t) for t in toks))
    if a.save_script:
        print("saved", save_script(toks, a.save_script, eng.final_env()))
    if a.install_unit:
        print(install_unit(toks, det, eng.final_env(), name=s.service_name, model=s.model, hf=s.hf,
                           replace_old=a.replace_old_unit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
