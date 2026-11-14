# llama-autotuner (llamatune.py)

Find the fastest `llama-server` settings for your machine. It loads each candidate configuration for real, measures it with a real request, and keeps the fastest one that leaves enough VRAM free.

Pure Python standard library. Linux and macOS (Windows: `pip install windows-curses` for the TUI; headless mode works everywhere).

## What it does

- Detects CPU, RAM, GPUs, your `llama-server` build and the flags it supports
- Reads the GGUF header: architecture, layers, MoE or dense, expert size, MTP
- Imports your current setup (running server or systemd user unit) and measures it as a baseline
- Tunes `--n-cpu-moe` (MoE) or `-ngl` (dense), `-ub`, `-t`, KV cache types, `--moe-cache-mib`, MTP
- Sweeps any extra flag you name with `--sweep` (new in 1.1)
- Saves one profile per model and exports llama-swap YAML or a router-mode `models.ini`

## Usage

```bash
./llamatune.py                       # terminal UI
./llamatune.py --detect              # print detection as JSON
./llamatune.py --headless --model ~/models/x.gguf --ctx 32768 --save-script run.sh
./llamatune.py --profiles
./llamatune.py --export-router ~/.config/llama-server/models.ini
./llamatune.py --debug-report        # redacted diagnostics for bug reports
```

### Sweeping a new flag

`--sweep FLAG=V1,V2,...` loads each value on top of the best configuration found so far and keeps the winner. It works for any flag your build lists in `llama-server --help`, and can be repeated. The leading `--` is optional.

```bash
./llamatune.py --headless --model x.gguf --sweep moe-cache-cpu-share=0.25,0.5,0.75
```

- llamatune does not know what a swept flag means; it only measures. Check the flag's range in `--help` first.
- If the build does not list the flag, the sweep is skipped with a note.
- If your current command already sets the flag, that value is used in the baseline measurement.
- Flags you do not sweep are carried over unchanged from your current command (flags the build does not know are dropped with a note).

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests run offline against a stub of the `llama-server` capabilities.

## Notes

- Results are cached per binary, model, GPU and llama.cpp version in `~/.cache/llamatune`. Use `--fresh` to ignore the cache.
- Use `--max-probes 5` for a quick first test.
