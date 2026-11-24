package ai.xybrid.reactnative

import ai.xybrid.*

/** Calls hold a registry lease; final disposal closes the generated handle
 * after those calls finish, cancelling unfinished work without flushing. */
internal class XybridObservationEntry private constructor(
  private val owner: AutoCloseable,
  private val getContext: () -> XybridObservationContext?,
  private val getReference: () -> XybridObservationRef?,
  val complete: () -> Unit,
  val fail: (String, String?) -> Unit,
  val cancel: () -> Unit,
  val isFinished: () -> Boolean,
  private val update: (String, JsObject) -> Unit,
) : AutoCloseable {
  override fun close() = owner.close()
  fun context(): JsObject? = getContext()?.let {
    mapOf("traceparent" to it.traceparent, "tracestate" to it.tracestate)
  }
  fun reference(): JsObject? = getReference()?.let(XybridCodec::encodeObservationRef)
  fun record(field: String, value: JsObject) {
    if (!isFinished()) update(field, value)
  }
  companion object {
    private const val what = "manual observation"
    private fun integer(o: JsObject, key: String): Double? {
      val value = XybridCodec.optionalDouble(o, key, what) ?: return null
      if (value < 0 || value > 9007199254740991.0 || value % 1.0 != 0.0) {
        throw BridgeException.InvalidArgument("observation integer must be a nonnegative safe integer")
      }
      return value
    }
    private fun location(o: JsObject): XybridExecutionLocation = when (
      XybridCodec.optionalString(o, "location", what) ?: "local"
    ) {
      "local" -> XybridExecutionLocation.LOCAL
      "remote" -> XybridExecutionLocation.REMOTE
      else -> throw BridgeException.InvalidArgument("unknown execution location")
    }
    private fun decodeToolCallInfo(o: JsObject): XybridToolCallInfo = XybridToolCallInfo(
      name = XybridCodec.string(o, "name", what),
      callId = XybridCodec.optionalString(o, "callId", what),
      toolType = XybridCodec.optionalString(o, "toolType", what),
      description = XybridCodec.optionalString(o, "description", what),
      argumentsJson = XybridCodec.optionalString(o, "argumentsJson", what)
    )
    private fun decodeInferenceRequest(o: JsObject): XybridInferenceRequest = XybridInferenceRequest(
      operation = XybridCodec.string(o, "operation", what),
      provider = XybridCodec.string(o, "provider", what),
      model = XybridCodec.optionalString(o, "model", what),
      location = location(o),
      runtime = XybridCodec.optionalString(o, "runtime", what),
      task = XybridCodec.optionalString(o, "task", what),
      stream = XybridCodec.optionalBoolean(o, "stream", what) ?: false,
      maxTokens = integer(o, "maxTokens")?.toLong(),
      temperature = XybridCodec.optionalDouble(o, "temperature", what),
      topP = XybridCodec.optionalDouble(o, "topP", what),
      serverAddress = XybridCodec.optionalString(o, "serverAddress", what),
      serverPort = integer(o, "serverPort")?.toLong()
    )
    private fun decodeModelLoadInfo(o: JsObject): XybridModelLoadInfo = XybridModelLoadInfo(
      model = XybridCodec.string(o, "model", what),
      provider = XybridCodec.optionalString(o, "provider", what),
      runtime = XybridCodec.optionalString(o, "runtime", what),
      format = XybridCodec.optionalString(o, "format", what),
      sizeBytes = integer(o, "sizeBytes")?.toULong()
    )
    private fun decodeUsage(o: JsObject): XybridUsage = XybridUsage(
      inputTokens = integer(o, "inputTokens")?.toULong(),
      outputTokens = integer(o, "outputTokens")?.toULong(),
      cacheReadInputTokens = integer(o, "cacheReadInputTokens")?.toULong(),
      cacheWriteInputTokens = integer(o, "cacheWriteInputTokens")?.toULong(),
      reasoningOutputTokens = integer(o, "reasoningOutputTokens")?.toULong()
    )

    fun start(kind: String, o: JsObject, options: XybridObservationOptions?): XybridObservationEntry = when (kind) {
      "workflow" -> {
        val op = XybridWorkflowObservation.new(XybridCodec.string(o, "name", what), options)
        XybridObservationEntry(op, op::context, op::observationRef, op::complete, op::fail, op::cancel, op::isFinished) { field, value ->
          when (field) {
            else -> throw BridgeException.InvalidArgument("unsupported observation field")
          }
        }
      }
      "tool" -> {
        val op = XybridToolObservation.new(decodeToolCallInfo(o), options)
        XybridObservationEntry(op, op::context, op::observationRef, op::complete, op::fail, op::cancel, op::isFinished) { field, value ->
          when (field) {
            "result" -> op.recordResult(XybridCodec.string(value, "json", what))
            else -> throw BridgeException.InvalidArgument("unsupported observation field")
          }
        }
      }
      "inference" -> {
        val op = XybridInferenceObservation.new(decodeInferenceRequest(o), options)
        XybridObservationEntry(op, op::context, op::observationRef, op::complete, op::fail, op::cancel, op::isFinished) { field, value ->
          when (field) {
            "systemInstructions" -> op.recordSystemInstructions(XybridCodec.string(value, "json", what))
            "inputMessages" -> op.recordInputMessages(XybridCodec.string(value, "json", what))
            "outputMessages" -> op.recordOutputMessages(XybridCodec.string(value, "json", what))
            "toolDefinitions" -> op.recordToolDefinitions(XybridCodec.string(value, "json", what))
            "chunk" -> op.recordChunk()
            "firstToken" -> op.recordFirstToken(integer(value, "atEpochMs")?.toULong())
            "usage" -> op.recordUsage(decodeUsage(value))
            "finishReasons" -> op.recordFinishReasons(XybridCodec.stringList(value, "reasons", what))
            "response" -> op.recordResponse(XybridCodec.optionalString(value, "responseId", what), XybridCodec.optionalString(value, "model", what))
            else -> throw BridgeException.InvalidArgument("unsupported observation field")
          }
        }
      }
      "modelLoad" -> {
        val op = XybridModelLoadObservation.new(decodeModelLoadInfo(o), options)
        XybridObservationEntry(op, op::context, op::observationRef, op::complete, op::fail, op::cancel, op::isFinished) { field, value ->
          when (field) {
            "cacheHit" -> op.recordCacheHit(XybridCodec.optionalBoolean(value, "hit", what) ?: false)
            else -> throw BridgeException.InvalidArgument("unsupported observation field")
          }
        }
      }
      else -> throw BridgeException.InvalidArgument("unknown observation kind")
    }
  }
}
