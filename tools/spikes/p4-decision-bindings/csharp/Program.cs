// P4 spike: decision records round-trip through the generated C# bindings
// (the Unity-down-leveled copy in bindings/unity/Runtime/Bolt, compiled at
// C# 9) and the real cdylib. Mirrors ../python/roundtrip.py check for check.

using System;
using System.Collections.Generic;
using System.Linq;
using XybridBolt;

internal static class Program
{
    const string StateJson =
        "{\"zeta\":1,\"alpha\":{\"2\":true,\"10\":false,\"1\":null}," +
        "\"mid\":[1.0,-0.0,1e300,5e-324,18446744073709551615,-9223372036854775808," +
        "9223372036854775808,18446744073709551616,-0]," +
        "\"é\":\"\\u00e9\\n\",\"nested\":[[{\"b\":1,\"a\":2}]],\"k\":\"v\"}";
    const string StateJsonCanonical =
        "{\"zeta\":1,\"alpha\":{\"2\":true,\"10\":false,\"1\":null}," +
        "\"mid\":[1.0,-0.0,1e300,5e-324,18446744073709551615,-9223372036854775808," +
        "9223372036854775808,1.8446744073709552e19,0]," +
        "\"é\":\"é\\n\",\"nested\":[[{\"b\":1,\"a\":2}]],\"k\":\"v\"}";

    static int failures;

    static void Check(string name, bool condition, string detail = "")
    {
        if (!condition) { failures++; Console.WriteLine($"FAIL {name}: {detail}"); }
        else Console.WriteLine($"ok   {name}");
    }

    static XybridDecisionContent Text(string value) => new XybridDecisionContent.Text(value);
    static XybridDecisionContent Json(string json) => new XybridDecisionContent.Json(json);

    static string[] OptionKeys() => Enumerable.Range(0, 30).Select(i => ((i * 7) % 30).ToString()).ToArray();

    static XybridDecisionContent? OptionDescription(int i) => (i % 4) switch
    {
        0 => Text($"option {i}"),
        1 => Json($"{{\"z\":{i},\"a\":[{i}.0,\"x\"]}}"),
        2 => null,
        _ => Json("\"a JSON string becomes text\""),
    };

    static XybridDecisionRequest Request(string state = StateJson)
    {
        var keys = OptionKeys();
        var choice = new XybridDecisionQuestion(
            "10",
            new XybridDecisionInstructions.Content(Json("{\"task\":\"pick\",\"why\":[3,1,2]}")),
            new XybridDecisionQuestionKind.Choice(
                keys.Select((key, i) => new XybridDecisionOption(key, OptionDescription(i))).ToArray()));
        var yesNo = new XybridDecisionQuestion(
            "2",
            new XybridDecisionInstructions.Content(Text("Is it urgent?")),
            new XybridDecisionQuestionKind.YesNo(Json("{\"means\":\"urgent\",\"rank\":1}"), null));
        var score = new XybridDecisionQuestion(
            "1",
            null,
            new XybridDecisionQuestionKind.Score(new[]
            {
                new XybridDecisionLevel(null),
                new XybridDecisionLevel(Text("low")),
                new XybridDecisionLevel(Json("[1,2.5]")),
                new XybridDecisionLevel(Json("\"mid\"")),
                new XybridDecisionLevel(Json("null")),
            }));
        return new XybridDecisionRequest(Json(state), new[] { choice, yesNo, score });
    }

    static List<string> ExpectedOutline()
    {
        var lines = new List<string> { $"state=json:{StateJsonCanonical}" };
        lines.Add("q0:id=10:choice:instructions=json:{\"task\":\"pick\",\"why\":[3,1,2]}");
        var keys = OptionKeys();
        for (int i = 0; i < 30; i++)
        {
            string description = (i % 4) switch
            {
                0 => $"text:option {i}",
                1 => $"json:{{\"z\":{i},\"a\":[{i}.0,\"x\"]}}",
                2 => "none",
                _ => "text:a JSON string becomes text",
            };
            lines.Add($"q0.o{i}:key={keys[i]}:{description}");
        }
        lines.AddRange(new[]
        {
            "q1:id=2:yes_no:instructions=text:Is it urgent?",
            "q1.yes:json:{\"means\":\"urgent\",\"rank\":1}",
            "q1.no:none",
            "q2:id=1:score:instructions=none",
            "q2.l0:none", "q2.l1:text:low", "q2.l2:json:[1,2.5]", "q2.l3:text:mid", "q2.l4:none",
        });
        return lines;
    }

    // Structural rendering: the down-leveled records are plain structs whose
    // Equals compares arrays by reference, so compare renderings instead.
    static string Render(XybridDecisionContent? content) => content switch
    {
        null => "none",
        XybridDecisionContent.Text t => $"text:{t.Value}",
        XybridDecisionContent.Json j => $"json:{j.JsonText}",
        _ => "unknown",
    };

    static List<string> Render(XybridDecisionRequest request)
    {
        var lines = new List<string> { $"state={Render(request.State)}" };
        foreach (var question in request.Questions)
        {
            string instructions = question.Instructions is XybridDecisionInstructions.Content c ? Render(c.Value) : "none";
            lines.Add($"{question.Id}|{instructions}");
            switch (question.Kind)
            {
                case XybridDecisionQuestionKind.Choice choice:
                    lines.AddRange(choice.Options.Select(o => $"o:{o.Key}:{Render(o.Description)}"));
                    break;
                case XybridDecisionQuestionKind.YesNo yesNo:
                    lines.Add($"yes:{Render(yesNo.Yes)}|no:{Render(yesNo.No)}");
                    break;
                case XybridDecisionQuestionKind.Score score:
                    lines.AddRange(score.Levels.Select(l => $"l:{Render(l.Description)}"));
                    break;
            }
        }
        return lines;
    }

    static string Bits32(float value) => ((uint)BitConverter.SingleToInt32Bits(value)).ToString("x8");
    static string Bits64(double value) => ((ulong)BitConverter.DoubleToInt64Bits(value)).ToString("x16");

    static List<string> DecodedBits(XybridDecisions decisions)
    {
        var lines = new List<string>();
        for (int a = 0; a < decisions.Answers.Length; a++)
        {
            var answer = decisions.Answers[a];
            for (int s = 0; s < answer.Distribution.Length; s++)
            {
                lines.Add($"a{a}.s{s}.raw={Bits32(answer.Distribution[s].RawScore)}");
                lines.Add($"a{a}.s{s}.p={Bits64(answer.Distribution[s].Probability)}");
            }
            lines.Add(answer.Value switch
            {
                XybridDecisionValue.Choice c => $"a{a}.value=choice:{c.Key}",
                XybridDecisionValue.YesNo y => $"a{a}.value=yes_no:{Bits64(y.PYes)}",
                XybridDecisionValue.Score sc => $"a{a}.value=score:{Bits64(sc.Value)}:{sc.Statistic}",
                _ => "unknown",
            });
            lines.Add(answer.Confidence is double confidence ? $"a{a}.confidence={Bits64(confidence)}" : $"a{a}.confidence=none");
        }
        return lines;
    }

    static string FirstDifference(IList<string> got, IList<string> expected)
    {
        for (int i = 0; i < Math.Min(got.Count, expected.Count); i++)
            if (got[i] != expected[i]) return $"{got[i]} != {expected[i]}";
        return got.Count == expected.Count ? "" : $"{got.Count} lines != {expected.Count}";
    }

    static int Main()
    {
        var request = Request();

        var outline = XybridBolt.XybridBolt.DecisionSpikeRequestOutline(request);
        var expected = ExpectedOutline();
        Check("Rust saw the state, 3 questions and 30 options in order",
            outline.SequenceEqual(expected), FirstDifference(outline, expected));

        var echoed = XybridBolt.XybridBolt.DecisionSpikeEchoRequest(request);
        var normalized = Render(Request(StateJsonCanonical))
            .Select(line => line
                .Replace("json:\"a JSON string becomes text\"", "text:a JSON string becomes text")
                .Replace("l:json:\"mid\"", "l:text:mid")
                .Replace("l:json:null", "l:none"))
            .ToList();
        var renderedEcho = Render(echoed);
        Check("echo returns the normalized request", renderedEcho.SequenceEqual(normalized),
            FirstDifference(renderedEcho, normalized));

        var decisions = XybridBolt.XybridBolt.DecisionSpikeDecide(request);
        Check("answers in request order",
            decisions.Answers.Select(a => a.QuestionId).SequenceEqual(new[] { "10", "2", "1" }));
        Check("choice distribution in option order",
            decisions.Answers[0].Distribution.Select(s => s.Key).SequenceEqual(OptionKeys()));
        Check("yes/no order is true, false",
            decisions.Answers[1].Distribution.Select(s => s.Key).SequenceEqual(new[] { "true", "false" }));
        Check("yes/no has no confidence", decisions.Answers[1].Confidence == null);
        Check("score level 0 raw score is -inf", float.IsNegativeInfinity(decisions.Answers[2].Distribution[0].RawScore));
        Check("u64 usage crosses", decisions.Usage.EvaluatedTokens == 1000UL + 1017UL + 1034UL);

        var expectedBits = XybridBolt.XybridBolt.DecisionSpikeDecideBits(request);
        var gotBits = DecodedBits(decisions);
        Check($"{expectedBits.Length} float bit patterns decode exactly",
            gotBits.SequenceEqual(expectedBits), FirstDifference(gotBits, expectedBits));

        Check("C# re-encodes the result bit for bit",
            XybridBolt.XybridBolt.DecisionSpikeVerifyDecisions(request, decisions) == null);
        var first = decisions.Answers[0].Distribution[0];
        var nudgedDistribution = decisions.Answers[0].Distribution.ToArray();
        nudgedDistribution[0] = new XybridDecisionScore(first.Key, first.RawScore, Math.BitIncrement(first.Probability), first.Origin);
        var nudgedAnswers = decisions.Answers.ToArray();
        nudgedAnswers[0] = new XybridDecisionAnswer(
            decisions.Answers[0].QuestionId, nudgedDistribution, decisions.Answers[0].Value, decisions.Answers[0].Confidence);
        Check("a one-ULP change is detected",
            XybridBolt.XybridBolt.DecisionSpikeVerifyDecisions(request, new XybridDecisions(nudgedAnswers, decisions.Usage))
                == "answers[0].distribution[0]");

        try
        {
            XybridBolt.XybridBolt.DecisionSpikeDecide(Request("{\"secret\":1,\"secret\":2}"));
            Check("invalid JSON is refused", false);
        }
        catch (XybridErrorException e)
        {
            var message = e.Error is XybridError.InferenceError inference ? inference.Message : "";
            Check("invalid JSON is refused as a typed, content-free error",
                message.Contains("state") && !message.Contains("secret"), message);
        }

        var probe = new XybridProbeOptionalList(
            new XybridDecisionContent?[] { null, Text("x"), Json("[1]"), null },
            new string?[] { null, "a", null });
        var probeBack = XybridBolt.XybridBolt.DecisionSpikeProbeOptionalList(probe);
        Check("Vec<Option<data enum>> survives",
            probeBack.Levels.Select(Render).SequenceEqual(probe.Levels.Select(Render))
            && probeBack.Labels.SequenceEqual(probe.Labels));

        double nanPayload = BitConverter.Int64BitsToDouble(0x7FF8_0000_0000_1234);
        var specials = new[] { -0.0, 5e-324, double.MaxValue, double.NegativeInfinity, nanPayload };
        var nesting = new XybridProbeNesting(
            new[] { new[] { 1.5f, float.NegativeInfinity }, new float[0], new[] { -0.0f } },
            new[] { specials, new[] { 0.1 + 0.2 } },
            new XybridDecisionUsage(uint.MaxValue, ulong.MaxValue));
        var nestingBack = XybridBolt.XybridBolt.DecisionSpikeProbeNesting(nesting);
        Check("nested lists keep f64 bits (-0.0, subnormal, max, -inf, NaN payload)",
            nestingBack.Probabilities[0].Select(Bits64).SequenceEqual(specials.Select(Bits64))
            && Bits64(nestingBack.Probabilities[1][0]) == Bits64(0.1 + 0.2));
        Check("nested f32 rows and Option<record> survive",
            nestingBack.RawScores.Length == 3
            && nestingBack.RawScores.Zip(nesting.RawScores, (x, y) => x.Select(Bits32).SequenceEqual(y.Select(Bits32))).All(ok => ok)
            && nestingBack.Usage is XybridDecisionUsage usage && usage.Passes == uint.MaxValue && usage.EvaluatedTokens == ulong.MaxValue);

        var tree = (XybridProbeJson.Obj)XybridBolt.XybridBolt.DecisionSpikeProbeJson(
            "{\"2\":1,\"10\":[1.0,-0,18446744073709551615],\"1\":{\"z\":null,\"a\":\"s\"}}");
        Check("JSON tree keeps integer-like key order",
            tree.Entries.Select(e => e.Key).SequenceEqual(new[] { "2", "10", "1" }));
        var items = ((XybridProbeJson.Seq)tree.Entries[1].Value).Items;
        Check("JSON tree keeps number identity",
            items[0] is XybridProbeJson.Float && items[1] is XybridProbeJson.Int && items[2] is XybridProbeJson.Uint);

        var payload = XybridBolt.XybridBolt.DecisionSpikeProbePayload(new XybridProbePayload.DecisionResult(decisions));
        Check("a result inside a data-enum variant survives",
            payload is XybridProbePayload.DecisionResult result
            && XybridBolt.XybridBolt.DecisionSpikeVerifyDecisions(request, result.Decisions) == null);
        var requestPayload = XybridBolt.XybridBolt.DecisionSpikeProbePayload(new XybridProbePayload.DecisionRequest(request));
        Check("a request inside a data-enum variant survives",
            requestPayload is XybridProbePayload.DecisionRequest r && Render(r.Request).SequenceEqual(Render(request)));

        Console.WriteLine($"note ToString() of a request shows caller content: {request.ToString().Contains("urgent")}");
        Console.WriteLine($"note ToString() of a content variant shows caller content: {Text("Is it urgent?").ToString().Contains("urgent")}");
        Console.WriteLine(failures == 0 ? "C# ROUND TRIP: ALL CHECKS PASSED" : $"C# ROUND TRIP: {failures} FAILED");
        return failures == 0 ? 0 : 1;
    }
}
