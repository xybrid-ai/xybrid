# Host JSON probes

What each host's own JSON library does to object order and number identity —
the evidence for carrying decision JSON content across the FFI as text parsed
in Rust (report §5). Same input for all: `input.json`.

```bash
cd tools/spikes/p4-decision-bindings/host-json
node probe.mjs "$(cat input.json)"
python3 probe.py "$(cat input.json)"
dart run probe.dart "$(cat input.json)"
swiftc -O probe.swift -o /tmp/probe-swift && /tmp/probe-swift "$(cat input.json)"
dotnet run --project csharp/HostJsonProbe.csproj -c Release -- "$(cat input.json)"
```
