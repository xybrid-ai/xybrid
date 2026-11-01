"""P4 spike: decision records round-trip through the generated Python package.

Runs against the real cdylib through boltffi's generated wire layer and
compiled CPython bridge, so every check below crosses the FFI boundary:

1. a request with ordered JSON state, three question kinds and 30 options
   reaches the Rust IR in order (Rust reports what it saw, line by line);
2. the echoed request comes back normalized (JSON null -> None, JSON string ->
   text, compact JSON) and otherwise identical;
3. every float of a stub result decodes bit for bit (Rust sends the expected
   bit patterns separately), including a -inf raw score;
4. the result re-encoded by Python is bit-identical on the Rust side, and a
   one-ULP change is detected;
5. the probe shapes (Vec<Option<enum>>, nested lists, recursive JSON tree,
   records inside variants) survive an echo.

Usage: python3.11 roundtrip.py [module]   (default module: xybrid_bolt)
"""

from __future__ import annotations

import importlib
import math
import struct
import sys

b = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "xybrid_bolt")

STATE_JSON = (
    '{"zeta":1,"alpha":{"2":true,"10":false,"1":null},'
    '"mid":[1.0,-0.0,1e300,5e-324,18446744073709551615,-9223372036854775808,'
    '9223372036854775808,18446744073709551616,-0],'
    '"é":"\\u00e9\\n","nested":[[{"b":1,"a":2}]],"k":"v"}'
)
STATE_JSON_CANONICAL = (
    '{"zeta":1,"alpha":{"2":true,"10":false,"1":null},'
    '"mid":[1.0,-0.0,1e300,5e-324,18446744073709551615,-9223372036854775808,'
    '9223372036854775808,1.8446744073709552e19,0],'
    '"é":"é\\n","nested":[[{"b":1,"a":2}]],"k":"v"}'
)

Text = b.XybridDecisionContentText
Json = b.XybridDecisionContentJson


def option_keys() -> list[str]:
    return [str((i * 7) % 30) for i in range(30)]


def option_description(i: int):
    return [
        Text(value=f"option {i}"),
        Json(json_text=f'{{"z":{i},"a":[{i}.0,"x"]}}'),
        None,
        Json(json_text='"a JSON string becomes text"'),
    ][i % 4]


def request() -> "b.XybridDecisionRequest":
    choice = b.XybridDecisionQuestion(
        id="10",
        instructions=b.XybridDecisionInstructionsContent(
            value=Json(json_text='{"task":"pick","why":[3,1,2]}')
        ),
        kind=b.XybridDecisionQuestionKindChoice(
            options=[
                b.XybridDecisionOption(key=key, description=option_description(i))
                for i, key in enumerate(option_keys())
            ]
        ),
    )
    yes_no = b.XybridDecisionQuestion(
        id="2",
        instructions=b.XybridDecisionInstructionsContent(value=Text(value="Is it urgent?")),
        kind=b.XybridDecisionQuestionKindYesNo(
            yes=Json(json_text='{"means":"urgent","rank":1}'), no=None
        ),
    )
    score = b.XybridDecisionQuestion(
        id="1",
        instructions=None,
        kind=b.XybridDecisionQuestionKindScore(
            levels=[
                b.XybridDecisionLevel(description=None),
                b.XybridDecisionLevel(description=Text(value="low")),
                b.XybridDecisionLevel(description=Json(json_text="[1,2.5]")),
                b.XybridDecisionLevel(description=Json(json_text='"mid"')),
                b.XybridDecisionLevel(description=Json(json_text="null")),
            ]
        ),
    )
    return b.XybridDecisionRequest(
        state=Json(json_text=STATE_JSON), questions=[choice, yes_no, score]
    )


def expected_outline() -> list[str]:
    lines = [f"state=json:{STATE_JSON_CANONICAL}"]
    lines.append('q0:id=10:choice:instructions=json:{"task":"pick","why":[3,1,2]}')
    for i, key in enumerate(option_keys()):
        description = [
            f"text:option {i}",
            f'json:{{"z":{i},"a":[{i}.0,"x"]}}',
            "none",
            "text:a JSON string becomes text",
        ][i % 4]
        lines.append(f"q0.o{i}:key={key}:{description}")
    lines += [
        "q1:id=2:yes_no:instructions=text:Is it urgent?",
        'q1.yes:json:{"means":"urgent","rank":1}',
        "q1.no:none",
        "q2:id=1:score:instructions=none",
        "q2.l0:none",
        "q2.l1:text:low",
        "q2.l2:json:[1,2.5]",
        "q2.l3:text:mid",
        "q2.l4:none",
    ]
    return lines


def bits32(value: float) -> str:
    return format(struct.unpack("<I", struct.pack("<f", value))[0], "08x")


def bits64(value: float) -> str:
    return format(struct.unpack("<Q", struct.pack("<d", value))[0], "016x")


def decoded_bits(decisions) -> list[str]:
    lines = []
    for a, answer in enumerate(decisions.answers):
        for s, score in enumerate(answer.distribution):
            lines.append(f"a{a}.s{s}.raw={bits32(score.raw_score)}")
            lines.append(f"a{a}.s{s}.p={bits64(score.probability)}")
        value = answer.value
        if isinstance(value, b.XybridDecisionValueChoice):
            lines.append(f"a{a}.value=choice:{value.key}")
        elif isinstance(value, b.XybridDecisionValueYesNo):
            lines.append(f"a{a}.value=yes_no:{bits64(value.p_yes)}")
        else:
            statistic = b.XybridScoreStatistic(value.statistic).name.title()
            lines.append(f"a{a}.value=score:{bits64(value.value)}:{statistic}")
        confidence = answer.confidence
        lines.append(
            f"a{a}.confidence=none" if confidence is None else f"a{a}.confidence={bits64(confidence)}"
        )
    return lines


def check(name: str, condition: bool, detail: object = "") -> None:
    if not condition:
        raise SystemExit(f"FAIL {name}: {detail}")
    print(f"ok   {name}")


def main() -> int:
    req = request()

    outline = b.decision_spike_request_outline(req)
    expected = expected_outline()
    check(
        "Rust saw the state, 3 questions and 30 options in order",
        outline == expected,
        next((f"{o!r} != {e!r}" for o, e in zip(outline, expected) if o != e), len(outline)),
    )

    echoed = b.decision_spike_echo_request(req)
    normalized = request()
    choice = normalized.questions[0].kind.options
    for i in range(30):
        if i % 4 == 3:
            choice[i] = b.XybridDecisionOption(
                key=choice[i].key, description=Text(value="a JSON string becomes text")
            )
    levels = normalized.questions[2].kind.levels
    levels[3] = b.XybridDecisionLevel(description=Text(value="mid"))
    levels[4] = b.XybridDecisionLevel(description=None)
    normalized = b.XybridDecisionRequest(
        state=Json(json_text=STATE_JSON_CANONICAL), questions=normalized.questions
    )
    check("echo returns the normalized request", echoed == normalized)
    check(
        "echo keeps option order",
        [o.key for o in echoed.questions[0].kind.options] == option_keys(),
    )

    decisions = b.decision_spike_decide(req)
    check("answers in request order", [a.question_id for a in decisions.answers] == ["10", "2", "1"])
    check(
        "choice distribution in option order",
        [s.key for s in decisions.answers[0].distribution] == option_keys(),
    )
    check("yes/no order is true, false", [s.key for s in decisions.answers[1].distribution] == ["true", "false"])
    check("yes/no has no confidence", decisions.answers[1].confidence is None)
    check("score level 0 raw score is -inf", decisions.answers[2].distribution[0].raw_score == -math.inf)
    check("u64 usage crosses", decisions.usage.evaluated_tokens == 1000 + 1017 + 1034)

    expected_bits = b.decision_spike_decide_bits(req)
    got_bits = decoded_bits(decisions)
    check(
        f"{len(expected_bits)} float bit patterns decode exactly",
        got_bits == expected_bits,
        next((f"{g} != {e}" for g, e in zip(got_bits, expected_bits) if g != e), ""),
    )

    check("Python re-encodes the result bit for bit", b.decision_spike_verify_decisions(req, decisions) is None)
    first = decisions.answers[0].distribution[0]
    nudged_score = b.XybridDecisionScore(
        key=first.key,
        raw_score=first.raw_score,
        probability=math.nextafter(first.probability, 1.0),
        origin=first.origin,
    )
    nudged = b.XybridDecisions(
        answers=[
            b.XybridDecisionAnswer(
                question_id=decisions.answers[0].question_id,
                distribution=[nudged_score, *decisions.answers[0].distribution[1:]],
                value=decisions.answers[0].value,
                confidence=decisions.answers[0].confidence,
            ),
            *decisions.answers[1:],
        ],
        usage=decisions.usage,
    )
    check(
        "a one-ULP change is detected",
        b.decision_spike_verify_decisions(req, nudged) == "answers[0].distribution[0]",
    )

    try:
        b.decision_spike_decide(
            b.XybridDecisionRequest(
                state=Json(json_text='{"secret":1,"secret":2}'), questions=req.questions
            )
        )
        check("invalid JSON is refused", False)
    except b.XybridErrorException as error:
        message = getattr(error.error, "message", "")
        check(
            "invalid JSON is refused as a typed, content-free error",
            isinstance(error.error, b.XybridErrorInferenceError)
            and "state" in message
            and "secret" not in message,
            message,
        )

    probe = b.XybridProbeOptionalList(
        levels=[None, Text(value="x"), Json(json_text="[1]"), None], labels=[None, "a", None]
    )
    check("Vec<Option<data enum>> survives", b.decision_spike_probe_optional_list(probe) == probe)

    nan_payload = struct.unpack("<d", struct.pack("<Q", 0x7FF8_0000_0000_1234))[0]
    specials = [-0.0, 5e-324, sys.float_info.max, -math.inf, nan_payload]
    nesting = b.XybridProbeNesting(
        raw_scores=[[1.5, -math.inf], [], [-0.0]],
        probabilities=[specials, [0.1 + 0.2]],
        usage=b.XybridDecisionUsage(passes=2**32 - 1, evaluated_tokens=2**64 - 1),
    )
    back = b.decision_spike_probe_nesting(nesting)
    check(
        "nested lists keep f64 bits (-0.0, subnormal, max, -inf, NaN payload)",
        [bits64(x) for x in back.probabilities[0]] == [bits64(x) for x in specials]
        and bits64(back.probabilities[1][0]) == bits64(0.1 + 0.2),
    )
    check(
        "nested f32 rows and Option<record> survive",
        [[bits32(x) for x in row] for row in back.raw_scores]
        == [[bits32(x) for x in row] for row in nesting.raw_scores]
        and back.usage == nesting.usage,
    )

    tree = b.decision_spike_probe_json('{"2":1,"10":[1.0,-0,18446744073709551615],"1":{"z":null,"a":"s"}}')
    keys = [entry.key for entry in tree.entries]
    check("JSON tree keeps integer-like key order", keys == ["2", "10", "1"], keys)
    items = tree.entries[1].value.items
    check(
        "JSON tree keeps number identity",
        isinstance(items[0], b.XybridProbeJsonFloat)
        and isinstance(items[1], b.XybridProbeJsonInt)
        and isinstance(items[2], b.XybridProbeJsonUint),
        items,
    )
    check("recursive JSON tree echoes", b.decision_spike_probe_json_echo(tree) == tree)

    payload = b.XybridProbePayloadDecisionResult(decisions=decisions)
    back_payload = b.decision_spike_probe_payload(payload)
    check(
        "a result inside a data-enum variant survives",
        isinstance(back_payload, b.XybridProbePayloadDecisionResult)
        and b.decision_spike_verify_decisions(req, back_payload.decisions) is None,
    )
    request_payload = b.XybridProbePayloadDecisionRequest(request=req)
    check("a request inside a data-enum variant survives", b.decision_spike_probe_payload(request_payload) == request_payload)

    leaked = "Is it urgent?" in repr(req)
    print(f"note repr() of a request shows caller content: {leaked}")
    print("PYTHON ROUND TRIP: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
