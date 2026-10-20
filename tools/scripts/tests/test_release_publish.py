"""Check that release publication covers the Cargo dependency graph in order."""

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class PublishDependencyOrderTests(unittest.TestCase):
    def test_local_dependencies_publish_before_their_consumers(self):
        workflow = (ROOT / ".github/workflows/release-publish.yml").read_text()
        order = re.findall(r"^\s+publish_crate ([a-z0-9-]+)\s*$", workflow, re.MULTILINE)
        self.assertTrue(order, "release workflow has no crate publication steps")
        self.assertEqual(len(order), len(set(order)), "duplicate crate publication")

        workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())
        manifests = {}
        for member in workspace["workspace"]["members"]:
            for directory in ROOT.glob(member):
                path = directory / "Cargo.toml"
                manifest = tomllib.loads(path.read_text())
                manifests[manifest["package"]["name"]] = (path, manifest)

        for consumer in order:
            path, manifest = manifests[consumer]
            scopes = [manifest, *manifest.get("target", {}).values()]
            for scope in scopes:
                for kind in ("dependencies", "build-dependencies"):
                    for spec in scope.get(kind, {}).values():
                        if not isinstance(spec, dict) or "path" not in spec:
                            continue
                        dependency_path = path.parent / spec["path"] / "Cargo.toml"
                        dependency = tomllib.loads(dependency_path.read_text())
                        name = dependency["package"]["name"]
                        with self.subTest(consumer=consumer, dependency=name):
                            self.assertIn(name, order, f"publish {name} before {consumer}")
                            self.assertLess(order.index(name), order.index(consumer))


if __name__ == "__main__":
    unittest.main()
