package ai.xybrid.smoke

import ai.xybrid.Envelope
import ai.xybrid.Xybrid
import ai.xybrid.XybridError
import ai.xybrid.XybridExecutionLocation
import ai.xybrid.XybridInferenceRequest
import ai.xybrid.XybridModel
import ai.xybrid.XybridModelSource
import ai.xybrid.XybridObservationRef
import ai.xybrid.XybridSignalStats
import ai.xybrid.XybridToolCallInfo
import ai.xybrid.childOptions
import ai.xybrid.deviceId
import ai.xybrid.displayMessage
import ai.xybrid.jsonProtocol
import ai.xybrid.observationOptions
import ai.xybrid.omitContent
import ai.xybrid.redactLiteral
import ai.xybrid.run
import ai.xybrid.runOptions
import ai.xybrid.service
import ai.xybrid.start
import ai.xybrid.version
import java.io.File

/**
 * The shared observability scenario, performed through the public Kotlin API.
 *
 * Implements `tools/scripts/observability_capture/scenario.md` step for step,
 * like the Python reference (`bindings/python/tests/observability_scenario.py`):
 * export to the capture receiver at [endpoint], then describe in
 * `expectations.json` what the binding said it did, which `check_capture.py`
 * compares with what arrived. No model weights are involved; [workDir] holds
 * the weightless model directory.
 *
 * Plain Kotlin with no Android types, so the instrumented
 * [ObservabilityScenarioTest] stays a thin shell around it.
 */
class ObservabilityScenario(private val endpoint: String, private val workDir: File) {

    /** The expectations document, plus the failure that interrupted it, if any. */
    class Outcome(val expectations: Map<String, Any?>, val error: Throwable?) {
        /** `expectations.json`, two-space indented. */
        fun toJson(): String = StringBuilder().also { writeJson(it, expectations, "") }.append('\n').toString()
    }

    /**
     * Runs steps (a) to (e). Never throws: a failed step is recorded as
     * `scenario_error`, export is shut down, and the error is returned with
     * whatever was recorded until then.
     */
    fun run(): Outcome {
        val references = linkedMapOf<String, Any?>()
        REFERENCE_NAMES.forEach { references[it] = null }
        val expectations = linkedMapOf<String, Any?>(
            "schema" to SCHEMA,
            "binding" to BINDING,
            "sdk_version" to version(),
            "trace_id" to TRACE_ID,
            "parent_span_id" to PARENT_SPAN_ID,
            "references" to references,
            "automatic" to linkedMapOf<String, Any?>(),
            "stats" to linkedMapOf<String, Any?>(),
            "lifecycle" to linkedMapOf<String, Any?>(),
            "scenario_error" to null,
        )
        val error = try {
            steps(expectations)
            null
        } catch (failure: Throwable) {
            expectations["scenario_error"] = failure.stackTraceToString().lineSequence().take(12).joinToString("\n")
            runCatching { Xybrid.Observability.shutdown(TIMEOUT_MS) }
            failure
        }
        return Outcome(expectations, error)
    }

    @Suppress("UNCHECKED_CAST")
    private fun steps(expectations: MutableMap<String, Any?>) {
        val references = expectations["references"] as MutableMap<String, Any?>
        val automatic = expectations["automatic"] as MutableMap<String, Any?>
        val stats = expectations["stats"] as MutableMap<String, Any?>
        val lifecycle = expectations["lifecycle"] as MutableMap<String, Any?>

        // (a) OTLP/JSON export to the receiver, with content rules and a device id.
        Xybrid.Observability.otlp(endpoint)
            .jsonProtocol()
            .service("xybrid-scenario-$BINDING", SERVICE_VERSION)
            .deviceId("scenario-device-$BINDING")
            .redactLiteral(SECRET, REPLACEMENT)
            .omitContent(OMITTED_ATTRIBUTE)
            .start()

        // (b) Parent-in: the workflow continues the caller's trace T under span P.
        val workflow = Xybrid.Observability.startWorkflow(
            WORKFLOW_NAME, observationOptions(traceparent = "00-$TRACE_ID-$PARENT_SPAN_ID-01"))
        references["workflow"] = reference(workflow.observationRef())
        val context = checkNotNull(workflow.context()) { "the workflow has no context although export is on" }
        val child = context.childOptions

        // (c) Manual children: a tool, a completed inference, a failed one and
        // one released unfinished.
        Xybrid.Observability.startTool(
            XybridToolCallInfo(TOOL_NAME, TOOL_CALL_ID, TOOL_TYPE, null, TOOL_ARGUMENTS_JSON), child,
        ).use { tool ->
            tool.recordResult(TOOL_RESULT_JSON)
            references["tool"] = reference(tool.observationRef())
            tool.complete()
        }

        Xybrid.Observability.startInference(INFERENCE_REQUEST, child).use { inference ->
            inference.recordSystemInstructions(SYSTEM_INSTRUCTIONS_JSON)
            inference.recordInputMessages(INPUT_MESSAGES_JSON)
            inference.recordOutputMessages(OUTPUT_MESSAGES_JSON)
            references["inference"] = reference(inference.observationRef())
            inference.complete()
        }

        Xybrid.Observability.startInference(INFERENCE_REQUEST, child).use { failed ->
            references["inference_failed"] = reference(failed.observationRef())
            failed.fail(FAILED_ERROR_TYPE, FAILED_MESSAGE)
        }

        val released = Xybrid.Observability.startInference(INFERENCE_REQUEST, child)
        released.recordInputMessages(RELEASED_INPUT_MESSAGES_JSON)
        references["inference_released"] = reference(released.observationRef())
        stats["before_release"] = snapshot()
        released.close() // the last owner, never ended: cancels, does not flush
        stats["after_release"] = snapshot()

        // (d) The automatic path: a weightless model loads lazily and fails to
        // run, both under the workflow with a conversation id.
        val modelDir = File(workDir, MODEL_DIRECTORY)
        modelDir.deleteRecursively()
        check(modelDir.mkdirs()) { "could not create $modelDir" }
        File(modelDir, "model_metadata.json").writeText(MODEL_METADATA_JSON)
        val model = try {
            XybridModel.load(
                XybridModelSource.Directory(modelDir.path),
                observationOptions(context.traceparent, context.tracestate, CONVERSATION_ID),
            )
        } catch (error: XybridError) {
            automatic["load"] = mapOf("outcome" to "error", "error" to describe(error))
            automatic["run"] = mapOf("outcome" to "skipped", "error" to null)
            null
        }
        model?.use {
            automatic["load"] = mapOf("outcome" to "ok", "error" to null)
            references["load"] = reference(it.loadObservationRef())
            val options = runOptions(
                traceparent = context.traceparent,
                tracestate = context.tracestate,
                conversationId = CONVERSATION_ID,
            )
            try {
                val result = it.run(Envelope.text(RUN_INPUT_TEXT), options)
                automatic["run"] = mapOf("outcome" to "ok", "error" to null)
                references["run"] = reference(result.observation)
            } catch (error: XybridError) {
                automatic["run"] = mapOf("outcome" to "error", "error" to describe(error))
            }
        }

        workflow.complete()
        workflow.close()

        // (e) Lifecycle: flush, stats, shutdown twice, then a no-op workflow.
        lifecycle["flush"] = outcome { Xybrid.Observability.flush(TIMEOUT_MS) }
        stats["after_flush"] = snapshot()
        lifecycle["shutdown"] = List(2) { outcome { Xybrid.Observability.shutdown(TIMEOUT_MS) } }
        stats["after_shutdown"] = snapshot()
        val after = Xybrid.Observability.startWorkflow(AFTER_SHUTDOWN_WORKFLOW_NAME)
        val afterRef = after.observationRef()
        val afterContext = after.context()
        after.complete()
        after.close()
        lifecycle["after_shutdown"] = linkedMapOf(
            "workflow_ref" to reference(afterRef),
            "workflow_context" to afterContext?.traceparent,
            "flush" to outcome { Xybrid.Observability.flush(TIMEOUT_MS) },
        )
    }

    companion object {
        const val BINDING = "kotlin"
        const val SCHEMA = "xybrid-observability-scenario/1"

        /** T and P (scenario.md, Identifiers): hex of the UTF-8 name, zero-padded. */
        val TRACE_ID: String = hexId("xy:$BINDING", 32)
        val PARENT_SPAN_ID: String = hexId(BINDING, 16)

        // The constants of tools/scripts/observability_capture/scenario.json.
        // check_capture.py reads that file, so a value that drifts here fails
        // the lane instead of passing silently.
        const val SERVICE_VERSION = "1.0.0"
        const val CONVERSATION_ID = "scenario-conversation-$BINDING"
        const val SECRET = "xybrid-scenario-secret-4c1d"
        const val REPLACEMENT = "[scenario-redacted]"
        const val OMITTED_ATTRIBUTE = "gen_ai.system_instructions"
        const val WORKFLOW_NAME = "scenario-workflow"
        const val AFTER_SHUTDOWN_WORKFLOW_NAME = "scenario-after-shutdown"
        const val TOOL_NAME = "scenario_lookup"
        const val TOOL_CALL_ID = "call-scenario-1"
        const val TOOL_TYPE = "function"
        const val TOOL_ARGUMENTS_JSON = """{"query":"weather","token":"xybrid-scenario-secret-4c1d"}"""
        const val TOOL_RESULT_JSON = """{"answer":"sunny","echo":"xybrid-scenario-secret-4c1d"}"""
        const val SYSTEM_INSTRUCTIONS_JSON = """[{"type":"text","content":"xybrid-scenario-omitted-9b2e"}]"""
        const val INPUT_MESSAGES_JSON =
            """[{"role":"user","parts":[{"type":"text","content":"my token is xybrid-scenario-secret-4c1d"}]}]"""
        const val OUTPUT_MESSAGES_JSON =
            """[{"role":"assistant","parts":[{"type":"text","content":"I will not repeat xybrid-scenario-secret-4c1d"}],"finish_reason":"stop"}]"""
        const val FAILED_ERROR_TYPE = "scenario_error"
        const val FAILED_MESSAGE = "upstream rejected xybrid-scenario-secret-4c1d"
        const val RELEASED_INPUT_MESSAGES_JSON =
            """[{"role":"user","parts":[{"type":"text","content":"abandoned request"}]}]"""
        const val MODEL_DIRECTORY = "scenario-weightless"
        const val MODEL_METADATA_JSON =
            """{"model_id":"scenario-weightless","version":"1.0","execution_template":{"type":"Onnx","model_file":"model.onnx"},"files":[]}"""
        const val RUN_INPUT_TEXT = "run input xybrid-scenario-secret-4c1d"
        const val TIMEOUT_MS = 5_000

        val INFERENCE_REQUEST = XybridInferenceRequest(
            operation = "chat",
            provider = "scenario-provider",
            model = "scenario-model",
            location = XybridExecutionLocation.LOCAL,
            runtime = "scenario-runtime",
            task = null,
            stream = false,
            maxTokens = null,
            temperature = null,
            topP = null,
            serverAddress = null,
            serverPort = null,
        )

        private val REFERENCE_NAMES = listOf(
            "workflow", "tool", "inference", "inference_failed", "inference_released", "load", "run")

        private fun hexId(text: String, digits: Int): String =
            text.toByteArray(Charsets.UTF_8).joinToString("") { "%02x".format(it) }.padEnd(digits, '0').take(digits)

        private fun reference(value: XybridObservationRef?): Map<String, String>? =
            value?.let { mapOf("trace_id" to it.traceId, "span_id" to it.spanId) }

        private fun describe(error: XybridError): String = "${error::class.simpleName}: ${error.displayMessage}"

        private fun outcome(call: () -> Unit): String = try {
            call()
            "ok"
        } catch (error: XybridError) {
            "error: ${describe(error)}"
        }

        private fun snapshot(): Map<String, Any?> {
            val stats = Xybrid.Observability.stats
            return linkedMapOf(
                "enabled" to stats.enabled,
                "traces" to counters(stats.traces),
                "logs" to counters(stats.logs),
                "metrics" to counters(stats.metrics),
            )
        }

        private fun counters(signal: XybridSignalStats): Map<String, Long> = linkedMapOf(
            "accepted" to signal.accepted.toLong(),
            "exported" to signal.exported.toLong(),
            "dropped" to signal.dropped.toLong(),
            "failed" to signal.failed.toLong(),
            "pending" to signal.pending.toLong(),
        )

        /** Minimal JSON for maps, lists, strings, numbers, booleans and null. */
        private fun writeJson(out: StringBuilder, value: Any?, indent: String) {
            when (value) {
                null -> out.append("null")
                is Boolean, is Int, is Long -> out.append(value.toString())
                is String -> writeString(out, value)
                is Map<*, *> -> writeEntries(out, indent, '{', '}', value.entries.toList()) { entry, inner ->
                    writeString(out, entry.key.toString())
                    out.append(": ")
                    writeJson(out, entry.value, inner)
                }
                is List<*> -> writeEntries(out, indent, '[', ']', value) { item, inner -> writeJson(out, item, inner) }
                else -> throw IllegalArgumentException("no JSON form for ${value::class}")
            }
        }

        private fun <T> writeEntries(out: StringBuilder, indent: String, open: Char, close: Char, items: List<T>,
                                     write: (T, String) -> Unit) {
            if (items.isEmpty()) {
                out.append(open).append(close)
                return
            }
            val inner = "$indent  "
            out.append(open)
            items.forEachIndexed { index, item ->
                out.append(if (index == 0) "\n" else ",\n").append(inner)
                write(item, inner)
            }
            out.append('\n').append(indent).append(close)
        }

        private fun writeString(out: StringBuilder, text: String) {
            out.append('"')
            for (char in text) {
                when {
                    char == '"' -> out.append("\\\"")
                    char == '\\' -> out.append("\\\\")
                    char == '\n' -> out.append("\\n")
                    char == '\r' -> out.append("\\r")
                    char == '\t' -> out.append("\\t")
                    char < ' ' -> out.append("\\u%04x".format(char.code))
                    else -> out.append(char)
                }
            }
            out.append('"')
        }
    }
}
