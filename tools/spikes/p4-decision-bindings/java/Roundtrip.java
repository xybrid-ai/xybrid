// P4 spike: core decision round trip through boltffi's Java target (JVM).
// The Java target is disabled in crates/xybrid-bolt/boltffi.toml, so this runs
// against `boltffi pack java` output generated with it switched on locally.
// Mirrors the order, normalization and bit-exactness checks of
// ../python/roundtrip.py.

import com.example.xybrid_bolt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public final class Roundtrip {
    static final String STATE_JSON =
        "{\"zeta\":1,\"alpha\":{\"2\":true,\"10\":false,\"1\":null},"
            + "\"mid\":[1.0,-0.0,1e300,5e-324,18446744073709551615,-9223372036854775808,"
            + "9223372036854775808,18446744073709551616,-0],"
            + "\"é\":\"\\u00e9\\n\",\"nested\":[[{\"b\":1,\"a\":2}]],\"k\":\"v\"}";
    static final String STATE_JSON_CANONICAL =
        "{\"zeta\":1,\"alpha\":{\"2\":true,\"10\":false,\"1\":null},"
            + "\"mid\":[1.0,-0.0,1e300,5e-324,18446744073709551615,-9223372036854775808,"
            + "9223372036854775808,1.8446744073709552e19,0],"
            + "\"é\":\"é\\n\",\"nested\":[[{\"b\":1,\"a\":2}]],\"k\":\"v\"}";

    static int failures = 0;

    static void check(String name, boolean condition, Object detail) {
        if (condition) {
            System.out.println("ok   " + name);
        } else {
            failures++;
            System.out.println("FAIL " + name + ": " + detail);
        }
    }

    static Optional<XybridDecisionContent> text(String value) {
        return Optional.of(new XybridDecisionContent.Text(value));
    }

    static Optional<XybridDecisionContent> json(String value) {
        return Optional.of(new XybridDecisionContent.Json(value));
    }

    static List<String> optionKeys() {
        List<String> keys = new ArrayList<>();
        for (int i = 0; i < 30; i++) keys.add(Integer.toString((i * 7) % 30));
        return keys;
    }

    static Optional<XybridDecisionContent> optionDescription(int i) {
        switch (i % 4) {
            case 0: return text("option " + i);
            case 1: return json("{\"z\":" + i + ",\"a\":[" + i + ".0,\"x\"]}");
            case 2: return Optional.empty();
            default: return json("\"a JSON string becomes text\"");
        }
    }

    static XybridDecisionRequest request(String state) {
        List<XybridDecisionOption> options = new ArrayList<>();
        List<String> keys = optionKeys();
        for (int i = 0; i < keys.size(); i++) options.add(new XybridDecisionOption(keys.get(i), optionDescription(i)));
        List<XybridDecisionLevel> levels = List.of(
            new XybridDecisionLevel(Optional.empty()),
            new XybridDecisionLevel(text("low")),
            new XybridDecisionLevel(json("[1,2.5]")),
            new XybridDecisionLevel(json("\"mid\"")),
            new XybridDecisionLevel(json("null")));
        return new XybridDecisionRequest(json(state), List.of(
            new XybridDecisionQuestion("10",
                Optional.of(new XybridDecisionInstructions.Content(json("{\"task\":\"pick\",\"why\":[3,1,2]}").get())),
                new XybridDecisionQuestionKind.Choice(options)),
            new XybridDecisionQuestion("2",
                Optional.of(new XybridDecisionInstructions.Content(text("Is it urgent?").get())),
                new XybridDecisionQuestionKind.YesNo(json("{\"means\":\"urgent\",\"rank\":1}"), Optional.empty())),
            new XybridDecisionQuestion("1", Optional.empty(), new XybridDecisionQuestionKind.Score(levels))));
    }

    static List<String> expectedOutline() {
        List<String> lines = new ArrayList<>();
        lines.add("state=json:" + STATE_JSON_CANONICAL);
        lines.add("q0:id=10:choice:instructions=json:{\"task\":\"pick\",\"why\":[3,1,2]}");
        List<String> keys = optionKeys();
        for (int i = 0; i < 30; i++) {
            String description;
            switch (i % 4) {
                case 0: description = "text:option " + i; break;
                case 1: description = "json:{\"z\":" + i + ",\"a\":[" + i + ".0,\"x\"]}"; break;
                case 2: description = "none"; break;
                default: description = "text:a JSON string becomes text";
            }
            lines.add("q0.o" + i + ":key=" + keys.get(i) + ":" + description);
        }
        lines.addAll(List.of(
            "q1:id=2:yes_no:instructions=text:Is it urgent?",
            "q1.yes:json:{\"means\":\"urgent\",\"rank\":1}",
            "q1.no:none",
            "q2:id=1:score:instructions=none",
            "q2.l0:none", "q2.l1:text:low", "q2.l2:json:[1,2.5]", "q2.l3:text:mid", "q2.l4:none"));
        return lines;
    }

    static String bits32(float value) {
        return String.format("%08x", Float.floatToRawIntBits(value));
    }

    static String bits64(double value) {
        return String.format("%016x", Double.doubleToRawLongBits(value));
    }

    static List<String> decodedBits(XybridDecisions decisions) {
        List<String> lines = new ArrayList<>();
        for (int a = 0; a < decisions.answers.size(); a++) {
            XybridDecisionAnswer answer = decisions.answers.get(a);
            for (int s = 0; s < answer.distribution.size(); s++) {
                lines.add("a" + a + ".s" + s + ".raw=" + bits32(answer.distribution.get(s).rawScore));
                lines.add("a" + a + ".s" + s + ".p=" + bits64(answer.distribution.get(s).probability));
            }
            if (answer.value instanceof XybridDecisionValue.Choice) {
                lines.add("a" + a + ".value=choice:" + ((XybridDecisionValue.Choice) answer.value).key);
            } else if (answer.value instanceof XybridDecisionValue.YesNo) {
                lines.add("a" + a + ".value=yes_no:" + bits64(((XybridDecisionValue.YesNo) answer.value).pYes));
            } else {
                XybridDecisionValue.Score score = (XybridDecisionValue.Score) answer.value;
                String name = score.statistic.name();
                lines.add("a" + a + ".value=score:" + bits64(score.value) + ":"
                    + name.charAt(0) + name.substring(1).toLowerCase(java.util.Locale.ROOT));
            }
            lines.add(answer.confidence.isPresent()
                ? "a" + a + ".confidence=" + bits64(answer.confidence.get())
                : "a" + a + ".confidence=none");
        }
        return lines;
    }

    public static void main(String[] args) {
        XybridDecisionRequest req = request(STATE_JSON);

        List<String> outline = XybridBolt.decisionSpikeRequestOutline(req);
        check("Rust saw the state, 3 questions and 30 options in order", outline.equals(expectedOutline()), outline);

        XybridDecisionRequest echoed = XybridBolt.decisionSpikeEchoRequest(req);
        List<String> echoedOutline = XybridBolt.decisionSpikeRequestOutline(echoed);
        check("echo is stable under a second conversion", echoedOutline.equals(outline), echoedOutline);
        XybridDecisionQuestionKind.Score score = (XybridDecisionQuestionKind.Score) echoed.questions.get(2).kind;
        check("JSON string and null normalize on echo",
            score.levels.get(3).description.equals(text("mid")) && score.levels.get(4).description.isEmpty(), score.levels);

        XybridDecisions decisions = XybridBolt.decisionSpikeDecide(req);
        List<String> ids = new ArrayList<>();
        decisions.answers.forEach(answer -> ids.add(answer.questionId));
        check("answers in request order", ids.equals(List.of("10", "2", "1")), ids);
        List<String> keys = new ArrayList<>();
        decisions.answers.get(0).distribution.forEach(s -> keys.add(s.key));
        check("choice distribution in option order", keys.equals(optionKeys()), keys);
        check("score level 0 raw score is -inf",
            decisions.answers.get(2).distribution.get(0).rawScore == Float.NEGATIVE_INFINITY, "");
        check("u64 usage crosses as a signed long", decisions.usage.evaluatedTokens == 3051L, decisions.usage.evaluatedTokens);

        List<String> expectedBits = XybridBolt.decisionSpikeDecideBits(req);
        List<String> gotBits = decodedBits(decisions);
        check(expectedBits.size() + " float bit patterns decode exactly", gotBits.equals(expectedBits), gotBits);

        check("Java re-encodes the result bit for bit",
            XybridBolt.decisionSpikeVerifyDecisions(req, decisions).isEmpty(), "");
        XybridDecisionScore first = decisions.answers.get(0).distribution.get(0);
        List<XybridDecisionScore> nudgedDistribution = new ArrayList<>(decisions.answers.get(0).distribution);
        nudgedDistribution.set(0, new XybridDecisionScore(first.key, first.rawScore, Math.nextUp(first.probability), first.origin));
        List<XybridDecisionAnswer> nudgedAnswers = new ArrayList<>(decisions.answers);
        XybridDecisionAnswer a0 = decisions.answers.get(0);
        nudgedAnswers.set(0, new XybridDecisionAnswer(a0.questionId, nudgedDistribution, a0.value, a0.confidence));
        check("a one-ULP change is detected",
            XybridBolt.decisionSpikeVerifyDecisions(req, new XybridDecisions(nudgedAnswers, decisions.usage))
                .equals(Optional.of("answers[0].distribution[0]")), "");

        try {
            XybridBolt.decisionSpikeDecide(request("{\"secret\":1,\"secret\":2}"));
            check("invalid JSON is refused", false, "");
        } catch (XybridError.InferenceError error) {
            check("invalid JSON is refused as a typed, content-free error",
                error.message.contains("state") && !error.message.contains("secret"), error.message);
        }

        System.out.println("note toString() of a request shows caller content: " + req.toString().contains("urgent"));
        System.out.println(failures == 0 ? "JAVA ROUND TRIP: ALL CHECKS PASSED" : "JAVA ROUND TRIP: " + failures + " FAILED");
        System.exit(failures == 0 ? 0 : 1);
    }
}
