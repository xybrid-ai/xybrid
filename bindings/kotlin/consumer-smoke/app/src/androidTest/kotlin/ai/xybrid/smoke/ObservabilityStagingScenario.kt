package ai.xybrid.smoke

import ai.xybrid.Envelope
import ai.xybrid.Xybrid
import ai.xybrid.XybridError
import ai.xybrid.XybridGenerationConfig
import ai.xybrid.XybridModel
import ai.xybrid.XybridModelSource
import ai.xybrid.XybridObservationContext
import ai.xybrid.XybridObservationRef
import ai.xybrid.XybridResult
import ai.xybrid.XybridRunOptions
import ai.xybrid.XybridSignalStats
import ai.xybrid.XybridStreamEventKind
import ai.xybrid.deviceId
import ai.xybrid.displayMessage
import ai.xybrid.metadataOnly
import ai.xybrid.observationOptions
import ai.xybrid.redactLiteral
import ai.xybrid.resourceAttribute
import ai.xybrid.runOptions
import ai.xybrid.runStream
import ai.xybrid.service
import ai.xybrid.start
import ai.xybrid.text
import ai.xybrid.tracesEndpoint
import ai.xybrid.version
import org.json.JSONObject

/**
 * The step 6 staging scenario, performed through the public Kotlin API.
 *
 * Implements `tools/scripts/observability_staging/README.md` like the Python
 * reference (`bindings/python/tests/observability_staging.py`): two sessions in
 * one process, `direct` (the Xybrid preset, metadata only) and `collector`
 * (OTLP to the pinned Collector, content captured with one literal redaction),
 * each loading a real registry model under a remote parent, stopping one
 * stream after three chunks, streaming a second to the end, and recording
 * feedback and an evaluation on the run reference.
 *
 * [config] is the parsed `config.json`. It holds the staging key, which only
 * reaches the Xybrid preset and is never logged or recorded. [device]
 * describes the device for the report. Plain Kotlin and `org.json`, so the
 * instrumented [ObservabilityStagingScenarioTest] stays a thin shell around it.
 */
class ObservabilityStagingScenario(private val config: JSONObject, private val device: Map<String, Any?>) {
    private val constants = config.getJSONObject("constants")
    private val binding = config.getString("binding")
    private val lane = config.getString("lane")
    private val runId = config.getString("run")
    private val timeoutMs = constants.getInt("timeout_ms")

    /** How a stream ended, the token events received, and the result if it completed. */
    private class Streamed(val outcome: String, val chunks: Int, val result: XybridResult?, val finishReason: String?)

    /**
     * Runs both sessions. Never throws: a failed step is recorded as
     * `scenario_error`, export is shut down, and the error is returned with
     * whatever was recorded until then. The capture scenario's
     * [ObservabilityScenario.Outcome] carries and writes the document.
     */
    fun run(): ObservabilityScenario.Outcome {
        val sessions = linkedMapOf<String, Any?>()
        val expectations = linkedMapOf<String, Any?>(
            "schema" to SCHEMA,
            "binding" to binding,
            "lane" to lane,
            "run" to runId,
            "sdk_version" to version(),
            "model_id" to config.getString("model_id"),
            "device" to device,
            "sessions" to sessions,
            "scenario_error" to null,
        )
        val error = try {
            for (session in SESSIONS) {
                val record = linkedMapOf<String, Any?>()
                sessions[session] = record
                runSession(session, record)
            }
            null
        } catch (failure: Throwable) {
            expectations["scenario_error"] = failure.stackTraceToString().lineSequence().take(12).joinToString("\n")
            runCatching { Xybrid.Observability.shutdown(timeoutMs) }
            failure
        }
        return ObservabilityScenario.Outcome(expectations, error)
    }

    /** Step 1: the session's export, identity and content rules, then start. */
    private fun configure(session: String, values: JSONObject) {
        val observability = if (session == "direct") {
            Xybrid.Observability.xybrid(values.getString("api_key"))
                .tracesEndpoint(values.getString("traces_endpoint"))
                .metadataOnly()
        } else {
            Xybrid.Observability.otlp(values.getString("endpoint"))
                .redactLiteral(constants.getString("secret"), constants.getString("replacement"))
        }
        val perLane = constants.getJSONObject("per_lane")
        observability
            .service(perLane.getString("service_name").replace("{binding}", binding),
                     perLane.getString("service_version"))
            .deviceId(perLane.getString("device_id").replace("{lane}", lane))
        val acceptance = listOf("run" to runId, "lane" to lane, "commit" to config.getString("commit"),
                                "session" to session)
        for ((key, value) in acceptance) {
            observability.resourceAttribute("xybrid.acceptance.$key", value)
        }
        observability.start()
    }

    /** Steps 4 and 5: a pull stream, closed after [stopAfter] chunks or read to the end. */
    private fun stream(model: XybridModel, prompt: String, options: XybridRunOptions, stopAfter: Int?): Streamed {
        val streamId = model.runStream(Envelope.text(prompt), options)
        var open = true
        var chunks = 0
        var finish: String? = null
        try {
            while (true) {
                val event = model.streamNext(streamId)
                if (event.kind == XybridStreamEventKind.COMPLETE) {
                    val result = model.streamResult(streamId) // taking the result ends the session
                    open = false
                    return Streamed(if (stopAfter == null) "ok" else "completed", chunks, result, finish)
                }
                val token = event.token
                if (event.kind == XybridStreamEventKind.TOKEN && token != null) {
                    chunks += 1
                    finish = token.finishReason ?: finish
                    if (stopAfter != null && chunks >= stopAfter) {
                        // Closing before the terminal event cancels the run at
                        // its next token (a user cancellation, error.type cancelled).
                        open = false
                        model.streamClose(streamId)
                        return Streamed("cancelled", chunks, null, null)
                    }
                }
            }
        } finally {
            if (open) model.streamClose(streamId)
        }
    }

    private fun runSession(session: String, record: MutableMap<String, Any?>) {
        val values = config.getJSONObject("sessions").getJSONObject(session)
        val conversation = constants.getJSONObject("per_lane").getString("conversation_id")
            .replace("{run}", runId).replace("{session}", session)
        val references = linkedMapOf<String, Any?>("workflow" to null, "load" to null, "run" to null)
        record["trace_id"] = values.getString("trace_id")
        record["parent_span_id"] = values.getString("parent_span_id")
        record["references"] = references
        record["init"] = outcome { configure(session, values) }
        check(record["init"] == "ok") { "$session init failed: ${record["init"]}" }

        // Step 2: the workflow continues the driver's trace T under span P.
        val traceparent = "00-${values.getString("trace_id")}-${values.getString("parent_span_id")}-01"
        val workflow = Xybrid.Observability.startWorkflow(
            constants.getString("workflow_name"), observationOptions(traceparent = traceparent))
        workflow.use {
            references["workflow"] = reference(workflow.observationRef())
            val context = checkNotNull(workflow.context()) { "the workflow has no context although export is on" }

            // Step 3: a real registry model, loaded under the workflow.
            XybridModel.load(
                XybridModelSource.Registry(config.getString("model_id"), false),
                observationOptions(context.traceparent, context.tracestate, conversation),
            ).use { model ->
                references["load"] = reference(model.loadObservationRef())

                // Step 4: stopped after three chunks by closing the stream.
                val stopped = constants.getJSONObject("stopped")
                val halted = stream(model, stopped.getString("prompt"),
                                    streamOptions(stopped.getInt("max_tokens"), context, conversation),
                                    stopped.getInt("stop_after_chunks"))
                record["stopped"] = linkedMapOf(
                    "outcome" to halted.outcome, "chunks_consumed" to halted.chunks, "delivery" to "pull")

                // Step 5: streamed to the end; the result carries the run reference.
                val completed = constants.getJSONObject("completed")
                val streamed = stream(model, completed.getString("prompt"),
                                      streamOptions(completed.getInt("max_tokens"), context, conversation), null)
                val result = checkNotNull(streamed.result) { "the completed stream returned no result" }
                val text = result.text.orEmpty()
                record["completed"] = linkedMapOf(
                    "outcome" to streamed.outcome,
                    "chunks" to streamed.chunks,
                    "tokens_out" to result.metrics.tokensOut?.toLong(),
                    "text_chars" to text.codePointCount(0, text.length),
                    "finish_reason" to streamed.finishReason,
                )
                references["run"] = reference(result.observation)
                val runRef = checkNotNull(result.observation) { "the run has no reference although export is on" }

                // Step 6: quality results on the run reference.
                val feedback = constants.getJSONObject("feedback")
                val evaluation = constants.getJSONObject("evaluation")
                record["quality"] = linkedMapOf(
                    "feedback" to Xybrid.Observability.recordFeedback(
                        runRef, feedback.getDouble("score"), feedback.getString("label"),
                        feedback.getString("comment")),
                    "evaluation" to Xybrid.Observability.recordEvaluation(
                        runRef, evaluation.getString("name"), evaluation.getDouble("score_value"),
                        evaluation.getString("score_label"), evaluation.getString("explanation")),
                )
            }
            // Step 7: lifecycle. The model was released when its block ended;
            // the workflow completes, then is released when this one ends.
            workflow.complete()
        }
        val lifecycle = linkedMapOf<String, Any?>("flush" to outcome { Xybrid.Observability.flush(timeoutMs) })
        record["lifecycle"] = lifecycle
        val stats = linkedMapOf<String, Any?>("after_flush" to snapshot())
        record["stats"] = stats
        lifecycle["shutdown"] = outcome { Xybrid.Observability.shutdown(timeoutMs) }
        stats["after_shutdown"] = snapshot()
    }

    companion object {
        const val SCHEMA = "xybrid-staging-scenario/1"
        private val SESSIONS = listOf("direct", "collector")

        private fun generation(maxTokens: Int) = XybridGenerationConfig(
            maxTokens = maxTokens.toUInt(), temperature = null, topP = null, minP = null, topK = null,
            repetitionPenalty = null, stopSequences = emptyList(), grammar = null, tools = emptyList(),
        )

        private fun streamOptions(maxTokens: Int, context: XybridObservationContext, conversationId: String) =
            runOptions(
                generationConfig = generation(maxTokens),
                traceparent = context.traceparent,
                tracestate = context.tracestate,
                conversationId = conversationId,
            )

        private fun reference(value: XybridObservationRef?): Map<String, String>? =
            value?.let { mapOf("trace_id" to it.traceId, "span_id" to it.spanId) }

        private fun outcome(call: () -> Unit): String = try {
            call()
            "ok"
        } catch (error: XybridError) {
            "error: ${error::class.simpleName}: ${error.displayMessage}"
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
    }
}
