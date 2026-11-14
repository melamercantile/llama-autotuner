"""Offline tests for --sweep support (no llama-server or GPU needed)."""
import importlib.machinery
import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
loader = importlib.machinery.SourceFileLoader("llamatune", str(ROOT / "llamatune.py"))
spec = importlib.util.spec_from_loader("llamatune", loader)
lt = importlib.util.module_from_spec(spec)
sys.modules["llamatune"] = lt
loader.exec_module(lt)

HELP = """--n-cpu-moe N
--moe-cache-mib N
--moe-cache-cpu-share X
--flash-attn [on|off|auto]
--cache-type-k --cache-reuse --fit --threads-batch
"""


class FakeCaps:
    help = HELP
    version = "test"

    def has(self, flag):
        return flag in self.help


def make_engine(sweep, extra=()):
    e = object.__new__(lt.Engine)
    e.s = lt.Settings()
    e.s.bin, e.s.model, e.s.sweep = "/bin/llama-server", "/m.gguf", sweep
    e.caps, e.moe, e.nl, e.extra = FakeCaps(), True, 40, list(extra)
    return e


CFG = dict(ctx=8192, host=10, ub=1024, t=8, ctk="q8_0", ctv="q8_0", cache=2048)


class SweepTests(unittest.TestCase):
    def test_flag_added_to_command(self):
        e = make_engine([("--moe-cache-cpu-share", ["0.5"])])
        cmd = e.cmd(dict(CFG, x=[["--moe-cache-cpu-share", "0.5"]]), 8099)
        i = cmd.index("--moe-cache-cpu-share")
        self.assertEqual(cmd[i + 1], "0.5")
        self.assertEqual(cmd.count("--moe-cache-cpu-share"), 1)

    def test_no_flag_without_sweep_value(self):
        self.assertNotIn("--moe-cache-cpu-share", make_engine([]).cmd(dict(CFG), 8099))

    def test_norm_and_describe(self):
        self.assertEqual(lt.norm(dict(CFG, x=[])), lt.norm(dict(CFG)))
        self.assertIn("moe-cache-cpu-share=0.25",
                      lt.describe(dict(CFG, x=[["--moe-cache-cpu-share", "0.25"]])))

    def test_cache_keys_differ_per_value(self):
        import json
        a = json.dumps(lt.norm(dict(CFG, x=[["--moe-cache-cpu-share", "0.25"]])), sort_keys=True)
        b = json.dumps(lt.norm(dict(CFG, x=[["--moe-cache-cpu-share", "0.5"]])), sort_keys=True)
        self.assertNotEqual(a, b)

    def test_parse_cmdline_keeps_flag_as_extra(self):
        out = lt.parse_cmdline(["llama-server", "-m", "x.gguf", "--moe-cache-cpu-share", "0.4"])
        self.assertEqual(out["extra"], ["--moe-cache-cpu-share", "0.4"])


if __name__ == "__main__":
    unittest.main()
