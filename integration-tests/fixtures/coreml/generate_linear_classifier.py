"""Generate the tiny deterministic Core ML fixture used by integration tests.

Requires coremltools 9.0 and numpy. The generated model and this source are
licensed under the repository's Apache-2.0 license.
"""

import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder


OUTPUT_PATH = Path(__file__).with_name("xybrid_linear.mlpackage")
LEGACY_MODEL_PATH = Path(__file__).with_name("xybrid_linear.mlmodel")
MODEL_IDENTIFIER = "00000000-0000-0000-0000-000000000149"

builder = NeuralNetworkBuilder(
    [("input", datatypes.Array(4))],
    [("scores", datatypes.Array(3))],
    use_float_arraytype=True,
)
builder.add_inner_product(
    name="linear_classifier",
    W=np.asarray(
        [
            1.0,
            0.0,
            -1.0,
            0.5,
            0.0,
            2.0,
            0.0,
            -0.5,
            -1.0,
            0.5,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    ),
    b=np.asarray([0.25, -0.5, 1.0], dtype=np.float32),
    input_channels=4,
    output_channels=3,
    has_bias=True,
    input_name="input",
    output_name="scores",
)
builder.spec.description.metadata.shortDescription = (
    "Tiny deterministic classifier for Xybrid Core ML integration tests"
)
builder.spec.description.metadata.author = "Xybrid contributors"
if OUTPUT_PATH.exists():
    shutil.rmtree(OUTPUT_PATH)
LEGACY_MODEL_PATH.unlink(missing_ok=True)

model = ct.models.MLModel(builder.spec, compute_units=ct.ComputeUnit.CPU_ONLY)
model.save(str(OUTPUT_PATH))

# coremltools creates a random package item UUID. Stabilize it so regenerating
# this committed fixture produces an identical manifest.
manifest_path = OUTPUT_PATH / "Manifest.json"
manifest = json.loads(manifest_path.read_text())
model_entry = next(iter(manifest["itemInfoEntries"].values()))
manifest["itemInfoEntries"] = {MODEL_IDENTIFIER: model_entry}
manifest["rootModelIdentifier"] = MODEL_IDENTIFIER
manifest_path.write_text(json.dumps(manifest, indent=4, sort_keys=True) + "\n")
print(OUTPUT_PATH)
