"""CPU-only synthetic pairing regressions (unittest or pytest; no Torch needed)."""
from __future__ import annotations

import ast
from collections.abc import Sequence
from contextlib import contextmanager
import importlib.util
import multiprocessing
from pathlib import Path
import pickle
import random
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PAIRING = ROOT / "src/lerobot/policies/BPV5StageB/pairing.py"
MODULE = "_bpv5_pairing_cpu_test"


@contextmanager
def load_pairing(tasks=None):
    tasks = tasks if tasks is not None else {}
    data = types.ModuleType("lerobot.policies.BPV5QFormer.data80d")
    data.normalize_task_text = lambda raw: "robot demonstration" if raw is None else str(raw).strip()
    data.episode_tasks = lambda rel: tasks.get(rel, {})
    data.task_text = lambda rel, ep: data.normalize_task_text(tasks.get(rel, {}).get(ep))
    data.info = lambda rel: {"robot_type": "synthetic"}
    packages = {}
    for name in ("lerobot", "lerobot.policies", "lerobot.policies.BPV5QFormer"):
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package
    packages["lerobot.policies.BPV5QFormer"].data80d = data
    packages[data.__name__] = data
    spec = importlib.util.spec_from_file_location(MODULE, PAIRING)
    module = importlib.util.module_from_spec(spec)
    packages[MODULE] = module
    with patch.dict(sys.modules, packages):
        spec.loader.exec_module(module)
        yield module


def _spawn_round_trip(payload, connection):
    try:
        with load_pairing():
            partners = pickle.loads(payload)
            connection.send(("ok", {key: list(value) for key, value in partners.items()},
                             len(pickle.dumps(partners, protocol=5))))
    except BaseException as exc:
        connection.send(("error", repr(exc)))
    finally:
        connection.close()


class SharedPairingTests(unittest.TestCase):
    def setUp(self):
        self.tasks = {}
        context = load_pairing(self.tasks)
        self.pairing = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def compare(self, items, pools, **kwargs):
        legacy, old_stats = self.pairing.build_partner_map(items, pools, storage="list", **kwargs)
        shared, new_stats = self.pairing.build_partner_map(items, pools, storage="shared_bucket", **kwargs)
        self.assertEqual(old_stats, new_stats)
        self.assertEqual(list(legacy), list(shared))
        self.assertEqual(legacy, {key: list(value) for key, value in shared.items()})
        for value in legacy.values():
            self.assertIs(type(value), list)
        for value in shared.values():
            self.assertIsInstance(value, self.pairing.SharedPartnerCandidates)
            self.assertIsInstance(value, Sequence)
        return legacy, shared, new_stats

    def test_default_stays_list_and_timing_preserves_results(self):
        items = [("agibot/a", ep, 20) for ep in range(4)]
        pools = {"agibot/a": list(range(4))}
        default, stats = self.pairing.build_partner_map(items, pools)
        explicit, shared, _ = self.compare(items, pools)
        self.assertEqual(default, explicit)
        self.assertTrue(all(type(value) is list for value in default.values()))
        for mode, expected in (("list", explicit), ("shared_bucket", shared)):
            timed, timed_stats = self.pairing.build_partner_map(items, pools, storage=mode, timing=True)
            self.assertEqual({key: list(value) for key, value in timed.items()},
                             {key: list(value) for key, value in expected.items()})
            self.assertEqual(timed_stats, stats)

    def test_complete_ordering_l0_l1_and_duplicate_rows(self):
        self.tasks.update({"droid/a": {ep: "put cup" if ep < 4 else "other" for ep in range(8)}})
        pools = {"agibot/a": [9, 3, 8, 2], "droid/a": [3, 1, 0, 2, 7, 6, 4, 5],
                 "unknown/a": [40, 41, 42, 43], "agibot/small": [0, 1, 2]}
        items = [(rel, ep, 5) for rel, eps in pools.items() for ep in eps]
        items += [("agibot/a", 9, 99), ("missing/a", 9, 1)]
        legacy, _, stats = self.compare(items, pools)
        self.assertEqual(legacy[("droid/a", 1)], [3, 0, 2])
        self.assertEqual(legacy[("agibot/a", 9)], [3, 8, 2])
        self.assertEqual(stats, {"kept": 17, "dropped": 4})

    def test_sequence_all_indices_slices_and_index_protocol(self):
        legacy, shared, _ = self.compare([("agibot/a", 2, 1)], {"agibot/a": [5, 2, 7, 2, 8, 7]})
        expected, value = legacy[("agibot/a", 2)], shared[("agibot/a", 2)]
        self.assertEqual(expected, [5, 7, 8, 7])
        for index in range(-len(expected), len(expected)):
            self.assertEqual(value[index], expected[index])
        for index in (-100, -len(expected) - 1, len(expected), 100):
            with self.assertRaises(IndexError):
                value[index]
        for start in (None, -20, -3, 0, 1, 20):
            for stop in (None, -20, -1, 0, 3, 20):
                for step in (None, -3, -1, 1, 2, 5):
                    index = slice(start, stop, step)
                    self.assertEqual(list(value[index]), expected[index])
        with self.assertRaises(ValueError):
            value[::0]
        for index in (1.5, "1", None):
            with self.assertRaises(TypeError):
                value[index]
        class Index:
            def __index__(self):
                return -1
        self.assertEqual(value[Index()], expected[-1])

    def test_missing_self_duplicates_and_no_partner(self):
        self.tasks.update({"droid/a": {ep: "same" for ep in range(4)}})
        pools = {"agibot/a": [0, 1, 1, 2, 3], "droid/a": [0, 1, 1, 2, 3],
                 "agibot/repeated": [9, 9, 9, 9]}
        items = [("agibot/a", 99, 1), ("droid/a", 99, 1), ("agibot/a", 0, 1),
                 ("droid/a", 0, 1), ("agibot/repeated", 9, 1)]
        legacy, _, stats = self.compare(items, pools)
        self.assertEqual(legacy[("agibot/a", 99)], pools["agibot/a"])
        self.assertEqual(legacy[("droid/a", 99)], pools["droid/a"])
        self.assertEqual(legacy[("agibot/a", 0)], [1, 1, 2, 3])
        self.assertEqual(legacy[("droid/a", 0)], [1, 2, 3])
        self.assertEqual(stats, {"kept": 4, "dropped": 2})

    def test_invalid_text_small_l1_and_overrides(self):
        self.tasks.update({"droid/a": {0: "", 1: "robot demonstration", 2: "unique", 3: "other"},
                           "agibot/a": {ep: "same" if ep < 4 else "other" for ep in range(5)}})
        pools = {"droid/a": list(range(4)), "agibot/a": list(range(5))}
        items = [(rel, ep, 1) for rel, eps in pools.items() for ep in eps]
        legacy, _, stats = self.compare(items, pools, tier_overrides={"agibot": "L1"})
        self.assertEqual(legacy[("agibot/a", 0)], [1, 2, 3])
        self.assertEqual(legacy[("droid/a", 0)], [1, 2, 3])
        self.assertEqual(stats, {"kept": 9, "dropped": 5})
        self.compare(items, pools, tier_overrides={"droid": "L0"}, min_bucket=2)
        for mode in ("list", "shared_bucket"):
            with self.assertRaises(ValueError):
                self.pairing.build_partner_map(items, pools, storage=mode, min_bucket=1)
            with self.assertRaises(ValueError):
                self.pairing.build_partner_map(items, pools, storage=mode, tier_overrides={"droid": "L2"})
        with self.assertRaises(ValueError):
            self.pairing.build_partner_map(items, pools, storage="invalid")

    def test_choose_partner_fixed_indices_and_no_rng(self):
        state = random.getstate()
        with patch.object(random, "random", side_effect=AssertionError("RNG used")), \
             patch.object(random, "choice", side_effect=AssertionError("RNG used")), \
             patch.object(random, "randrange", side_effect=AssertionError("RNG used")):
            legacy, shared, _ = self.compare([("agibot/a", 2, 1)], {"agibot/a": [2, 7, 8, 9, 2, 7]})
            for index in (0, 1, -1, -10**30, 10**30, 999999):
                expected = [7, 8, 9, 7][index % 4]
                for candidates in (legacy[("agibot/a", 2)], shared[("agibot/a", 2)], [2, 7, 8, 9, 2, 7]):
                    self.assertEqual(self.pairing.choose_partner(candidates, 2, index), expected)
            for candidates in ([], [2], [2, 2]):
                with self.assertRaises(ValueError):
                    self.pairing.choose_partner(candidates, 2, 0)
        self.assertEqual(random.getstate(), state)

    def test_pickle_and_real_spawn_preserve_values_and_bucket_sharing(self):
        items = [("agibot/a", ep, 1) for ep in range(64)]
        legacy, shared, _ = self.compare(items, {"agibot/a": list(range(64))})
        for protocol in (4, 5):
            payload = pickle.dumps(shared, protocol=protocol)
            restored = pickle.loads(payload)
            self.assertEqual({key: list(value) for key, value in restored.items()}, legacy)
            independent = sum(len(pickle.dumps(value, protocol=protocol)) for value in shared.values())
            self.assertLess(len(payload), independent // 2)
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_spawn_round_trip, args=(payload, child))
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(20), "spawn worker did not reply within 20s")
            result = parent.recv()
            self.assertEqual(result[0], "ok", result)
            self.assertEqual(result[1], legacy)
            self.assertEqual(result[2], len(pickle.dumps(shared, protocol=5)))
            process.join(5)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            parent.close()


class IntegrationTests(unittest.TestCase):
    def test_config_storage_default_and_validation(self):
        path = PAIRING.with_name("configuration_bpv5stageb.py")
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "BPV5StageBDatasetConfig")
        field = next(node for node in cls.body if isinstance(node, ast.AnnAssign)
                     and node.target.id == "pairing_storage")
        self.assertEqual(ast.literal_eval(field.value), "list")
        guard = next(node for node in ast.walk(cls) if isinstance(node, ast.If)
                     and "self.pairing_storage" in ast.unparse(node.test))
        for mode in ("list", "shared_bucket", "bad", None):
            scope = {"self": types.SimpleNamespace(pairing_storage=mode)}
            code = compile(ast.Module(body=[guard], type_ignores=[]), str(path), "exec")
            if mode in ("list", "shared_bucket"):
                exec(code, scope)
            else:
                with self.assertRaises(ValueError):
                    exec(code, scope)

    def test_dataset_passes_storage_and_hom_uses_sequence(self):
        import logging

        class Base:
            def _apply_split(self, by_repo):
                self.items = [row for rows in by_repo.values() for row in rows]

        def load_class(filename, name, method, scope, bases):
            path = PAIRING.with_name(filename)
            tree = ast.parse(path.read_text())
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
            cls.decorator_list = []
            cls.bases = bases
            cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method]
            module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
            return scope[name]

        with load_pairing() as pairing:
            for mode in ("list", "shared_bucket"):
                calls = []
                def build(*args, **kwargs):
                    calls.append(kwargs)
                    return pairing.build_partner_map(*args, **kwargs)
                scope = {"Base": Base, "build_partner_map": build, "logging": logging}
                dataset_class = load_class("dataset.py", "BPV5StageBDataset", "_apply_split", scope,
                                           [ast.Name(id="Base", ctx=ast.Load())])
                ds = dataset_class()
                ds.cfg = types.SimpleNamespace(min_repo_episodes=4, pairing_min_bucket=4,
                    pairing_tiers=None, pairing_storage=mode, pairing_timing=False,
                    split="val", heldout_monitor_max_episodes=4, heldout_monitor_batch=2,
                    heldout_monitor_alternate=True, heldout_monitor_wrong=False)
                ds._apply_split({"agibot/a": [("agibot/a", ep, 20) for ep in range(4)]})
                self.assertEqual(calls[0]["storage"], mode)
                self.assertEqual(ds.num_episodes, 4)
                self.assertEqual(ds.num_frames, 80)
                ds.segment_starts = lambda length, mode: [0]
                # __len__ lives on the class, not the instance.
                dataset_class.__len__ = lambda self: len(self.items)
                rng = types.SimpleNamespace(choice=lambda n, size, replace: list(range(size)))
                fake_np = types.SimpleNamespace(random=types.SimpleNamespace(default_rng=lambda seed: rng))
                scope = {"np": fake_np, "logging": logging, "choose_partner": pairing.choose_partner,
                         "choose_wrong_partner": pairing.choose_wrong_partner,
                         "_init_timer": lambda cfg: None, "_log_init_timing": lambda *args: None}
                monitor_class = load_class("heldout_monitor.py", "StageBHeldoutMonitor", "__init__", scope, [])
                monitor = monitor_class(ds)
                self.assertEqual(len(monitor.matched), 4)
                self.assertEqual(len(monitor.alt), 4)
                for index, matched in monitor.matched.items():
                    self.assertNotEqual(matched[1], ds.items[index][1])
                    self.assertNotEqual(matched, monitor.alt[index])


if __name__ == "__main__":
    unittest.main()
