"""Host JSON probe (Python): key order and number identity. Usage: python3 probe.py "$(cat input.json)"."""
import json
import sys

o = json.loads(sys.argv[1])
print("order", list(o["questions"]))
print("state", json.dumps(o["state"], separators=(",", ":")))
print("-inf", json.dumps(float("-inf")))
