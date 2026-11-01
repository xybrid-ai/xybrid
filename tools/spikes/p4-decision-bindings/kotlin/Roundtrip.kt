// P4 spike: decision records round-trip through the generated Kotlin bindings
// (bindings/kotlin/.../XybridBolt.kt) and the generated JNI glue on a desktop
// JVM. Mirrors ../python/roundtrip.py check for check.

import ai.xybrid.*

private const val STATE_JSON =
    "{\"zeta\":1,\"alpha\":{\"2\":true,\"10\":false,\"1\":null}," +
        "\"mid\":[1.0,-0.0,1e300,5e-324,18446744073709551615,-9223372036854775808," +
        "9223372036854775808,18446744073709551616,-0]," +
        "\"é\":\"\\u00e9\\n\",\"nested\":[[{\"b\":1,\"a\":2}]],\"k\":\"v\"}"
private const val STATE_JSON_CANONICAL =
    "{\"zeta\":1,\"alpha\":{\"2\":true,\"10\":false,\"1\":null}," +
        "\"mid\":[1.0,-0.0,1e300,5e-324,18446744073709551615,-9223372036854775808," +
        "9223372036854775808,1.8446744073709552e19,0]," +
        "\"é\":\"é\\n\",\"nested\":[[{\"b\":1,\"a\":2}]],\"k\":\"v\"}"

private var failures = 0

private fun check(name: String, condition: Boolean, detail: Any = "") {
    if (condition) println("ok   $name") else { failures++; println("FAIL $name: $detail") }
}

private fun text(value: String): XybridDecisionContent = XybridDecisionContent.Text(value = value)
private fun json(value: String): XybridDecisionContent = XybridDecisionContent.Json(jsonText = value)

private fun optionKeys(): List<String> = (0 until 30).map { ((it * 7) % 30).toString() }

private fun optionDescription(i: Int): XybridDecisionContent? = when (i % 4) {
    0 -> text("option $i")
    1 -> json("{\"z\":$i,\"a\":[$i.0,\"x\"]}")
    2 -> null
    else -> json("\"a JSON string becomes text\"")
}

private fun request(state: String = STATE_JSON) = XybridDecisionRequest(
    state = json(state),
    questions = listOf(
        XybridDecisionQuestion(
            id = "10",
            instructions = XybridDecisionInstructions.Content(value = json("{\"task\":\"pick\",\"why\":[3,1,2]}")),
            kind = XybridDecisionQuestionKind.Choice(
                options = optionKeys().mapIndexed { i, key -> XybridDecisionOption(key = key, description = optionDescription(i)) },
            ),
        ),
        XybridDecisionQuestion(
            id = "2",
            instructions = XybridDecisionInstructions.Content(value = text("Is it urgent?")),
            kind = XybridDecisionQuestionKind.YesNo(yes = json("{\"means\":\"urgent\",\"rank\":1}"), no = null),
        ),
        XybridDecisionQuestion(
            id = "1",
            instructions = null,
            kind = XybridDecisionQuestionKind.Score(
                levels = listOf(null, text("low"), json("[1,2.5]"), json("\"mid\""), json("null"))
                    .map { XybridDecisionLevel(description = it) },
            ),
        ),
    ),
)

private fun expectedOutline(): List<String> = buildList {
    add("state=json:$STATE_JSON_CANONICAL")
    add("q0:id=10:choice:instructions=json:{\"task\":\"pick\",\"why\":[3,1,2]}")
    optionKeys().forEachIndexed { i, key ->
        val description = when (i % 4) {
            0 -> "text:option $i"
            1 -> "json:{\"z\":$i,\"a\":[$i.0,\"x\"]}"
            2 -> "none"
            else -> "text:a JSON string becomes text"
        }
        add("q0.o$i:key=$key:$description")
    }
    addAll(
        listOf(
            "q1:id=2:yes_no:instructions=text:Is it urgent?",
            "q1.yes:json:{\"means\":\"urgent\",\"rank\":1}",
            "q1.no:none",
            "q2:id=1:score:instructions=none",
            "q2.l0:none", "q2.l1:text:low", "q2.l2:json:[1,2.5]", "q2.l3:text:mid", "q2.l4:none",
        ),
    )
}

private fun bits32(value: Float) = "%08x".format(value.toRawBits())
private fun bits64(value: Double) = "%016x".format(value.toRawBits())

private fun decodedBits(decisions: XybridDecisions): List<String> = buildList {
    decisions.answers.forEachIndexed { a, answer ->
        answer.distribution.forEachIndexed { s, score ->
            add("a$a.s$s.raw=${bits32(score.rawScore)}")
            add("a$a.s$s.p=${bits64(score.probability)}")
        }
        add(
            when (val value = answer.value) {
                is XybridDecisionValue.Choice -> "a$a.value=choice:${value.key}"
                is XybridDecisionValue.YesNo -> "a$a.value=yes_no:${bits64(value.pYes)}"
                is XybridDecisionValue.Score ->
                    "a$a.value=score:${bits64(value.value)}:${value.statistic.name.lowercase().replaceFirstChar { it.uppercase() }}"
            },
        )
        add(answer.confidence?.let { "a$a.confidence=${bits64(it)}" } ?: "a$a.confidence=none")
    }
}

private fun firstDifference(got: List<String>, expected: List<String>): String =
    got.zip(expected).firstOrNull { (g, e) -> g != e }?.let { (g, e) -> "$g != $e" } ?: "${got.size} vs ${expected.size} lines"

fun main() {
    val req = request()

    val outline = decisionSpikeRequestOutline(req)
    val expected = expectedOutline()
    check("Rust saw the state, 3 questions and 30 options in order", outline == expected, firstDifference(outline, expected))

    val normalizedQuestions = request(STATE_JSON_CANONICAL).questions.map { question ->
        when (val kind = question.kind) {
            is XybridDecisionQuestionKind.Choice -> question.copy(
                kind = kind.copy(
                    options = kind.options.mapIndexed { i, option ->
                        if (i % 4 == 3) option.copy(description = text("a JSON string becomes text")) else option
                    },
                ),
            )
            is XybridDecisionQuestionKind.Score -> question.copy(
                kind = kind.copy(
                    levels = kind.levels.mapIndexed { l, level ->
                        when (l) {
                            3 -> XybridDecisionLevel(description = text("mid"))
                            4 -> XybridDecisionLevel(description = null)
                            else -> level
                        }
                    },
                ),
            )
            else -> question
        }
    }
    val echoed = decisionSpikeEchoRequest(req)
    check("echo returns the normalized request", echoed == XybridDecisionRequest(json(STATE_JSON_CANONICAL), normalizedQuestions))

    val decisions = decisionSpikeDecide(req)
    check("answers in request order", decisions.answers.map { it.questionId } == listOf("10", "2", "1"))
    check("choice distribution in option order", decisions.answers[0].distribution.map { it.key } == optionKeys())
    check("yes/no order is true, false", decisions.answers[1].distribution.map { it.key } == listOf("true", "false"))
    check("yes/no has no confidence", decisions.answers[1].confidence == null)
    check("score level 0 raw score is -inf", decisions.answers[2].distribution[0].rawScore == Float.NEGATIVE_INFINITY)
    check("u64 usage crosses", decisions.usage.evaluatedTokens == (1000UL + 1017UL + 1034UL))

    val expectedBits = decisionSpikeDecideBits(req)
    val gotBits = decodedBits(decisions)
    check("${expectedBits.size} float bit patterns decode exactly", gotBits == expectedBits, firstDifference(gotBits, expectedBits))

    check("Kotlin re-encodes the result bit for bit", decisionSpikeVerifyDecisions(req, decisions) == null)
    val first = decisions.answers[0].distribution[0]
    val nudged = decisions.copy(
        answers = listOf(
            decisions.answers[0].copy(
                distribution = listOf(first.copy(probability = Math.nextUp(first.probability))) +
                    decisions.answers[0].distribution.drop(1),
            ),
        ) + decisions.answers.drop(1),
    )
    check("a one-ULP change is detected", decisionSpikeVerifyDecisions(req, nudged) == "answers[0].distribution[0]")

    try {
        decisionSpikeDecide(request("{\"secret\":1,\"secret\":2}"))
        check("invalid JSON is refused", false)
    } catch (error: XybridError.InferenceError) {
        check(
            "invalid JSON is refused as a typed, content-free error",
            error.message.contains("state") && !error.message.contains("secret"),
            error.message,
        )
    }

    val probe = XybridProbeOptionalList(levels = listOf(null, text("x"), json("[1]"), null), labels = listOf(null, "a", null))
    check("Vec<Option<data enum>> survives", decisionSpikeProbeOptionalList(probe) == probe)

    val nanPayload = Double.fromBits(0x7FF8_0000_0000_1234L)
    val specials = doubleArrayOf(-0.0, 5e-324, Double.MAX_VALUE, Double.NEGATIVE_INFINITY, nanPayload)
    val nesting = XybridProbeNesting(
        rawScores = listOf(floatArrayOf(1.5f, Float.NEGATIVE_INFINITY), floatArrayOf(), floatArrayOf(-0.0f)),
        probabilities = listOf(specials, doubleArrayOf(0.1 + 0.2)),
        usage = XybridDecisionUsage(passes = UInt.MAX_VALUE, evaluatedTokens = ULong.MAX_VALUE),
    )
    val back = decisionSpikeProbeNesting(nesting)
    check(
        "nested lists keep f64 bits (-0.0, subnormal, max, -inf, NaN payload)",
        back.probabilities[0].map(::bits64) == specials.map(::bits64) &&
            bits64(back.probabilities[1][0]) == bits64(0.1 + 0.2),
    )
    check(
        "nested f32 rows and Option<record> survive",
        back.rawScores.map { row -> row.map(::bits32) } == nesting.rawScores.map { row -> row.map(::bits32) } &&
            back.usage == nesting.usage,
    )

    val tree = decisionSpikeProbeJson("{\"2\":1,\"10\":[1.0,-0,18446744073709551615],\"1\":{\"z\":null,\"a\":\"s\"}}") as XybridProbeJson.Obj
    check("JSON tree keeps integer-like key order", tree.entries.map { it.key } == listOf("2", "10", "1"))
    val items = (tree.entries[1].value as XybridProbeJson.Seq).items
    check(
        "JSON tree keeps number identity",
        items[0] is XybridProbeJson.Float && items[1] is XybridProbeJson.Int && items[2] is XybridProbeJson.Uint,
        items,
    )
    check("recursive JSON tree echoes", decisionSpikeProbeJsonEcho(tree) == tree)

    val payload = decisionSpikeProbePayload(XybridProbePayload.DecisionResult(decisions = decisions))
    check(
        "a result inside a data-enum variant survives",
        payload is XybridProbePayload.DecisionResult && decisionSpikeVerifyDecisions(req, payload.decisions) == null,
    )
    val requestPayload = XybridProbePayload.DecisionRequest(request = req)
    check("a request inside a data-enum variant survives", decisionSpikeProbePayload(requestPayload) == requestPayload)

    println("note toString() of a request shows caller content: ${req.toString().contains("urgent")}")
    println(if (failures == 0) "KOTLIN ROUND TRIP: ALL CHECKS PASSED" else "KOTLIN ROUND TRIP: $failures FAILED")
    if (failures != 0) kotlin.system.exitProcess(1)
}
