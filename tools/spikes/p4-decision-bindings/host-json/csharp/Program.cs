// Host JSON probe (.NET System.Text.Json): key order and number identity. Usage: see ../README.md.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
var t = args[0];
var node = JsonNode.Parse(t)!.AsObject();
Console.WriteLine("JsonNode order " + string.Join(",", node["questions"]!.AsObject().Select(p => p.Key)));
Console.WriteLine("JsonNode state " + node["state"]!.ToJsonString());
var dict = JsonSerializer.Deserialize<Dictionary<string, object>>(node["state"]!.ToJsonString())!;
Console.WriteLine("Dictionary<string,object> round trip " + JsonSerializer.Serialize(dict));
var d = JsonSerializer.Deserialize<Dictionary<string, double>>("{\"x\":1.0,\"negf\":-0.0}")!;
Console.WriteLine("double values re-serialized " + JsonSerializer.Serialize(d));
try { Console.WriteLine("-inf " + JsonSerializer.Serialize(double.NegativeInfinity)); } catch (Exception e) { Console.WriteLine("-inf throws " + e.GetType().Name); }
