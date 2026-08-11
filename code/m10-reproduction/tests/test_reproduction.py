import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


m10_train = load_module("m10_train", ROOT / "code" / "train.py")
m10_split = load_module("m10_split", ROOT / "code" / "prepare_split.py")
m10_reproduce = load_module("m10_reproduce", ROOT / "code" / "reproduce_m10.py")


class ReproductionTests(unittest.TestCase):
    def test_m10_command_contract(self):
        config = m10_train.load_config(ROOT / "code" / "config.yaml")
        command = m10_train.build_command(
            config,
            base_model=Path("/base"),
            train_data=Path("/data/train.jsonl"),
            output_dir=Path("/output"),
            cache_dir=Path("/cache"),
            nproc_per_node=2,
            resume_from_checkpoint=None,
        )
        self.assertEqual(config["model_id"], "M10")
        self.assertEqual(config["training"]["learning_rate"], 2e-6)
        self.assertEqual(config["training"]["max_steps"], 11_764)
        self.assertEqual(config["training_data"]["rows"], 94_113)
        self.assertIn("--negatives_cross_device", command)
        self.assertIn("--max_steps", command)
        self.assertIn("11764", command)

    def test_canonical_reproduction_pins_all_input_identities(self):
        commands = m10_reproduce.build_commands(
            python="/venv/bin/python",
            code_root=ROOT / "code",
            pairs=Path("/data/LRAT-training-pairs.jsonl"),
            trajectories=Path("/data/LRAT-trajectories.tar.gz"),
            corpus=Path("/data/offline_corpus.jsonl"),
            base_model=Path("/models/Qwen3-Embedding-0.6B"),
            output_root=Path("/output/m10"),
            dry_run=True,
        )
        provenance, preprocess, split, train = commands
        self.assertIn("--expected-archive-sha256", provenance)
        self.assertIn(m10_reproduce.TRAJECTORY_SHA256, provenance)
        self.assertIn("--expected-tokenizer-sha256", provenance)
        self.assertIn(m10_reproduce.TOKENIZER_SHA256, provenance)
        self.assertIn("--expected-corpus-sha256", preprocess)
        self.assertIn(m10_reproduce.CORPUS_SHA256, preprocess)
        self.assertIn("--salt", split)
        self.assertIn("ccir-early-stop-v1", split)
        self.assertIn("--dry-run", train)

    def test_query_split_is_deterministic_and_disjoint(self):
        rows = []
        for index in range(8):
            rows.append(
                {
                    "query": f"Query {index}",
                    "pos": [f"positive {index}"],
                    "pos_id": [f"p{index}"],
                    "neg": [f"negative {index}"],
                    "neg_id": [f"n{index}"],
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "pairs.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            first = root / "first"
            second = root / "second"
            a = m10_split.prepare(
                source,
                first,
                dev_queries=2,
                test_queries=2,
                salt="fixed",
                write_train=True,
            )
            b = m10_split.prepare(
                source,
                second,
                dev_queries=2,
                test_queries=2,
                salt="fixed",
                write_train=True,
            )
            self.assertEqual(a["train"]["output"]["sha256"], b["train"]["output"]["sha256"])
            self.assertEqual(a["dev"]["output"]["sha256"], b["dev"]["output"]["sha256"])
            self.assertEqual(a["test"]["output"]["sha256"], b["test"]["output"]["sha256"])
            self.assertEqual(a["train"]["normalized_query_overlap_with_dev_or_test"], 0)

    def test_no_generated_training_assets_are_tracked(self):
        forbidden_suffixes = {".jsonl", ".safetensors", ".pt", ".pth", ".zip", ".tar"}
        tracked_candidates = [
            path for path in ROOT.rglob("*")
            if path.is_file() and path.suffix in forbidden_suffixes
        ]
        self.assertEqual(tracked_candidates, [])


if __name__ == "__main__":
    unittest.main()
