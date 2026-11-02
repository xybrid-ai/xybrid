// Xybrid SDK - Inference Result
// Wrapper for the output of model inference.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace Xybrid
{
    /// <summary>
    /// Per-stage latency entry for pipeline runs. One entry per executed
    /// stage; <see cref="StageId"/> matches the stage name in the pipeline
    /// definition.
    /// </summary>
    public sealed class StageLatency
    {
        public string StageId { get; }
        public uint LatencyMs { get; }

        internal StageLatency(string stageId, uint latencyMs)
        {
            StageId = stageId;
            LatencyMs = latencyMs;
        }
    }

    /// <summary>
    /// When one phoneme is spoken in a TTS result's audio. Drive lip sync or
    /// word highlighting from <see cref="InferenceResult.PhonemeTimings"/>.
    /// </summary>
    /// <remarks>
    /// <see cref="Start"/> and <see cref="End"/> are seconds from the first
    /// sample of <see cref="InferenceResult.AudioBytes"/>, the audio exactly as
    /// returned (after silence trimming and chunk crossfades).
    /// </remarks>
    public sealed class PhonemeTiming
    {
        /// <summary>
        /// The phoneme as the model's phonemizer wrote it (IPA-style, for
        /// example <c>ə</c>). It can be more than one character, and it can be
        /// punctuation or a space.
        /// </summary>
        public string Phoneme { get; }

        /// <summary>Start of the phoneme, in seconds from the first audio sample.</summary>
        public float Start { get; }

        /// <summary>End of the phoneme, in seconds from the first audio sample.</summary>
        public float End { get; }

        internal PhonemeTiming(string phoneme, float start, float end)
        {
            Phoneme = phoneme;
            Start = start;
            End = end;
        }
    }

    /// <summary>
    /// Typed inference metrics surfaced on every <see cref="InferenceResult"/>.
    /// </summary>
    /// <remarks>
    /// LLM-specific fields (<see cref="TtftMs"/>, <see cref="TokensPerSecond"/>,
    /// <see cref="PrefillTps"/>, <see cref="DecodeTps"/>, <see cref="TokensOut"/>)
    /// are <c>null</c> for ASR/TTS/embedding runs. <see cref="StageLatenciesMs"/>
    /// is empty for <c>model.Run()</c> and populated for pipeline runs.
    /// </remarks>
    public sealed class InferenceMetrics
    {
        public uint TotalMs { get; }
        public uint? TtftMs { get; }
        public float? TokensPerSecond { get; }
        public float? PrefillTps { get; }
        public float? DecodeTps { get; }
        public uint? TokensOut { get; }
        public IReadOnlyList<StageLatency> StageLatenciesMs { get; }

        internal InferenceMetrics(
            uint totalMs,
            uint? ttftMs,
            float? tokensPerSecond,
            float? prefillTps,
            float? decodeTps,
            uint? tokensOut,
            IReadOnlyList<StageLatency> stageLatenciesMs)
        {
            TotalMs = totalMs;
            TtftMs = ttftMs;
            TokensPerSecond = tokensPerSecond;
            PrefillTps = prefillTps;
            DecodeTps = decodeTps;
            TokensOut = tokensOut;
            StageLatenciesMs = stageLatenciesMs;
        }
    }

    /// <summary>
    /// Represents the result of model inference.
    /// </summary>
    public sealed class InferenceResult : IDisposable
    {
        /// <summary>Gets whether this result has been disposed.</summary>
        public bool IsDisposed { get; private set; }

        /// <summary>Gets whether the inference was successful.</summary>
        public bool Success { get; }

        /// <summary>Kitten batch outcome, when present in the native result.</summary>
        public TtsStatus? SpeechStatus { get; private set; }
        /// <summary>Number of Kitten chunks that reached a speech-token limit.</summary>
        public uint? LimitedChunks { get; private set; }

        /// <summary>Gets the error message if inference failed, or null if successful.</summary>
        public string Error { get; }

        /// <summary>Gets the text output (for ASR or LLM models), or null if not applicable.</summary>
        public string Text { get; }

        /// <summary>Gets model reasoning separately from the final answer text.</summary>
        public string ReasoningContent { get; }

        /// <summary>Gets the inference latency in milliseconds.</summary>
        public uint LatencyMs { get; }

        /// <summary>Gets the type of output produced by inference.</summary>
        public OutputType OutputType { get; }

        /// <summary>
        /// Gets the raw audio bytes (for TTS models), or null if not applicable.
        /// Audio format is raw PCM 16-bit signed little-endian, typically 24kHz mono.
        /// </summary>
        public byte[] AudioBytes { get; }

        /// <summary>Gets the embedding vector (for embedding models), or null if not applicable.</summary>
        public float[] Embedding { get; }

        /// <summary>Gets whether this result contains audio data.</summary>
        public bool HasAudio => AudioBytes != null && AudioBytes.Length > 0;

        /// <summary>Gets whether this result contains an embedding.</summary>
        public bool HasEmbedding => Embedding != null && Embedding.Length > 0;

        /// <summary>Gets the typed inference metrics (TTFT, tok/s, per-stage latencies).</summary>
        public InferenceMetrics Metrics { get; }

        /// <summary>
        /// Gets whether this answer came from the device or the cloud gateway.
        /// Cloud fallback keeps the model id identical on both legs, so this is
        /// the only way to tell them apart. Defaults to
        /// <see cref="XybridBolt.XybridExecutionTarget.Local"/> for synthesized
        /// failures, which never reached the cloud.
        /// </summary>
        public XybridBolt.XybridExecutionTarget ExecutionTarget { get; }

        /// <summary>
        /// Tool calls the model asked for this turn.
        /// </summary>
        /// <remarks>
        /// Empty unless the request offered tools via
        /// <see cref="GenerationConfig.AddTool"/>. Run each call yourself, then
        /// feed the outcomes back with <see cref="Envelope.ToolResults"/>. The
        /// raw tool-call block stays in <see cref="Text"/> untouched, and
        /// malformed model output yields an empty list rather than an error.
        /// </remarks>
        public IReadOnlyList<XybridBolt.XybridToolCall> ToolCalls { get; }

        /// <summary>Gets whether the model asked to call at least one tool.</summary>
        public bool HasToolCalls => ToolCalls != null && ToolCalls.Count > 0;

        /// <summary>
        /// When each phoneme is spoken in <see cref="AudioBytes"/>, in order.
        /// Never null.
        /// </summary>
        /// <remarks>
        /// Filled for ONNX TTS models that expose a per-token duration output
        /// (for example KittenTTS 0.8, or a Kokoro export that adds one). Empty
        /// for every other model, for streaming TTS, for cloud results and for
        /// failed results. Entries never overlap and stay inside the audio.
        /// Padding and silence tokens are not listed, so there can be gaps
        /// between entries. A malformed timing payload yields an empty list
        /// rather than an error.
        /// </remarks>
        public IReadOnlyList<PhonemeTiming> PhonemeTimings { get; }

        /// <summary>Gets whether this result carries per-phoneme timings.</summary>
        public bool HasPhonemeTimings => PhonemeTimings != null && PhonemeTimings.Count > 0;

        private InferenceResult(
            bool success,
            string error,
            string text,
            uint latencyMs,
            OutputType outputType,
            byte[] audioBytes,
            float[] embedding,
            InferenceMetrics metrics,
            string reasoningContent,
            XybridBolt.XybridExecutionTarget executionTarget =
                XybridBolt.XybridExecutionTarget.Local,
            IReadOnlyList<XybridBolt.XybridToolCall> toolCalls = null,
            IReadOnlyList<PhonemeTiming> phonemeTimings = null)
        {
            Success = success;
            Error = error;
            Text = text;
            LatencyMs = latencyMs;
            OutputType = outputType;
            AudioBytes = audioBytes;
            Embedding = embedding;
            Metrics = metrics;
            ReasoningContent = reasoningContent;
            ExecutionTarget = executionTarget;
            ToolCalls = toolCalls ?? System.Array.Empty<XybridBolt.XybridToolCall>();
            PhonemeTimings = phonemeTimings ?? Array.Empty<PhonemeTiming>();
        }

        /// <summary>Decode a successful bolt result into the public shape.</summary>
        internal static InferenceResult FromBolt(XybridBolt.XybridResult result)
        {
            DecodePayload(result.Envelope, out string text, out byte[] audio, out float[] embedding);

            var speech = TtsStreamResult.FromBolt(result);
            bool hasSpeechStatus = false;
            string phonemeTimingsJson = null;
            foreach (var entry in result.Envelope.Metadata)
            {
                if (entry.Key == "tts_status")
                    hasSpeechStatus = true;
                else if (entry.Key == PhonemeTimingParser.MetadataKey && phonemeTimingsJson == null)
                    phonemeTimingsJson = entry.Value;
            }
            bool partial = hasSpeechStatus && speech.Status != TtsStatus.Completed;
            return new InferenceResult(
                success: !partial,
                error: partial ? "TTS " + speech.Status.ToString().ToLowerInvariant() + "; partial audio retained" : null,
                text: text,
                latencyMs: result.LatencyMs,
                outputType: MapOutputType(result.OutputType),
                audioBytes: audio,
                embedding: embedding,
                metrics: MapMetrics(result.Metrics),
                reasoningContent: result.ReasoningContent,
                executionTarget: result.ExecutionTarget,
                toolCalls: result.ToolCalls ?? System.Array.Empty<XybridBolt.XybridToolCall>(),
                phonemeTimings: PhonemeTimingParser.Parse(phonemeTimingsJson))
            {
                SpeechStatus = hasSpeechStatus ? speech.Status : (TtsStatus?)null,
                LimitedChunks = hasSpeechStatus ? speech.LimitedChunks : (uint?)null,
            };
        }

        /// <summary>
        /// A failed result. Bolt inference throws on error; <see cref="Model"/>
        /// catches inference failures and synthesizes this so callers that
        /// inspect <see cref="Success"/> keep their contract.
        /// </summary>
        internal static InferenceResult Failed(string error) =>
            new InferenceResult(
                success: false,
                error: error,
                text: null,
                latencyMs: 0,
                outputType: OutputType.Unknown,
                audioBytes: null,
                embedding: null,
                metrics: new InferenceMetrics(0, null, null, null, null, null, Array.Empty<StageLatency>()),
                reasoningContent: null);

        /// <summary>
        /// Split an envelope into the flat text / audio / embedding fields the
        /// public result types expose. Every field but the matching one is null.
        /// </summary>
        internal static void DecodePayload(
            XybridBolt.XybridEnvelope envelope,
            out string text,
            out byte[] audio,
            out float[] embedding)
        {
            text = null;
            audio = null;
            embedding = null;
            switch (envelope.Kind)
            {
                case XybridBolt.XybridEnvelopeKind.Text t:
                    text = t.Value;
                    break;
                case XybridBolt.XybridEnvelopeKind.Audio a:
                    audio = a.Bytes;
                    break;
                case XybridBolt.XybridEnvelopeKind.Embedding e:
                    embedding = e.Values;
                    break;
            }
        }

        internal static OutputType MapOutputType(XybridBolt.XybridOutputType outputType)
        {
            switch (outputType)
            {
                case XybridBolt.XybridOutputType.Text:
                    return OutputType.Text;
                case XybridBolt.XybridOutputType.Audio:
                    return OutputType.Audio;
                case XybridBolt.XybridOutputType.Embedding:
                    return OutputType.Embedding;
                default:
                    return OutputType.Unknown;
            }
        }

        internal static InferenceMetrics MapMetrics(XybridBolt.XybridInferenceMetrics metrics)
        {
            var stages = new List<StageLatency>(metrics.StageLatenciesMs.Length);
            foreach (XybridBolt.XybridStageLatency stage in metrics.StageLatenciesMs)
            {
                stages.Add(new StageLatency(stage.StageId, stage.LatencyMs));
            }

            return new InferenceMetrics(
                totalMs: metrics.TotalMs,
                ttftMs: metrics.TtftMs,
                tokensPerSecond: metrics.TokensPerSecond,
                prefillTps: metrics.PrefillTps,
                decodeTps: metrics.DecodeTps,
                tokensOut: metrics.TokensOut,
                stageLatenciesMs: stages);
        }

        /// <summary>
        /// Throws an InferenceException if the result indicates failure.
        /// </summary>
        /// <exception cref="InferenceException">Thrown if Success is false.</exception>
        public void ThrowIfFailed()
        {
            if (!Success)
            {
                throw new InferenceException(Error ?? "Unknown inference error");
            }
        }

        /// <summary>
        /// No-op: the result holds no native resources. Retained so existing
        /// <c>using</c> call sites keep compiling.
        /// </summary>
        public void Dispose()
        {
            IsDisposed = true;
        }

        /// <summary>Returns a string representation of the result.</summary>
        public override string ToString()
        {
            return Success
                ? $"InferenceResult(Success, OutputType={OutputType}, LatencyMs={LatencyMs}, " +
                  $"Text=\"{Text ?? "null"}\", AudioBytes={AudioBytes?.Length ?? 0})"
                : $"InferenceResult(Failed, Error=\"{Error}\")";
        }
    }

    /// <summary>
    /// Strict reader for the <c>phoneme_timings</c> metadata entry: a JSON
    /// array of <c>{"phoneme": string, "start": number, "end": number}</c>
    /// objects, as the native SDK writes it.
    /// </summary>
    /// <remarks>
    /// Unknown keys are skipped. Anything else that is not exactly that shape
    /// (invalid JSON, a missing, repeated or mistyped field, a time that is
    /// not a finite number, trailing text) makes the whole list empty: a
    /// partial list would put the mouth out of sync with the audio. Numbers
    /// always use the invariant culture, so a comma-decimal locale reads
    /// <c>0.5</c> as one half. <see cref="Parse"/> never throws.
    /// </remarks>
    internal sealed class PhonemeTimingParser
    {
        /// <summary>Metadata key the native SDK stores the timings under.</summary>
        internal const string MetadataKey = "phoneme_timings";

        // Values under unknown keys are skipped recursively; cap their nesting
        // so a hostile payload cannot exhaust the stack.
        private const int MaxSkipDepth = 32;

        private readonly string _json;
        private int _pos;

        private PhonemeTimingParser(string json)
        {
            _json = json;
        }

        /// <summary>
        /// Parse a <c>phoneme_timings</c> value. Returns an empty list when
        /// <paramref name="json"/> is null, empty or malformed.
        /// </summary>
        internal static IReadOnlyList<PhonemeTiming> Parse(string json)
        {
            if (string.IsNullOrEmpty(json))
                return Array.Empty<PhonemeTiming>();
            var parser = new PhonemeTimingParser(json);
            return parser.TryReadDocument(out PhonemeTiming[] timings)
                ? timings
                : Array.Empty<PhonemeTiming>();
        }

        private bool TryReadDocument(out PhonemeTiming[] timings)
        {
            timings = null;
            var list = new List<PhonemeTiming>();
            SkipWhitespace();
            if (!Consume('['))
                return false;
            SkipWhitespace();
            if (!Consume(']'))
            {
                while (true)
                {
                    if (!TryReadTiming(out PhonemeTiming timing))
                        return false;
                    list.Add(timing);
                    SkipWhitespace();
                    if (Consume(']'))
                        break;
                    if (!Consume(','))
                        return false;
                    SkipWhitespace();
                }
            }
            SkipWhitespace();
            if (_pos != _json.Length)
                return false;
            timings = list.ToArray();
            return true;
        }

        private bool TryReadTiming(out PhonemeTiming timing)
        {
            timing = null;
            string phoneme = null;
            float start = 0f;
            float end = 0f;
            bool hasStart = false;
            bool hasEnd = false;
            if (!Consume('{'))
                return false;
            SkipWhitespace();
            if (!Consume('}'))
            {
                while (true)
                {
                    if (!TryReadString(out string key))
                        return false;
                    SkipWhitespace();
                    if (!Consume(':'))
                        return false;
                    SkipWhitespace();
                    switch (key)
                    {
                        case "phoneme":
                            if (phoneme != null || !TryReadString(out phoneme))
                                return false;
                            break;
                        case "start":
                            if (hasStart || !TryReadTime(out start))
                                return false;
                            hasStart = true;
                            break;
                        case "end":
                            if (hasEnd || !TryReadTime(out end))
                                return false;
                            hasEnd = true;
                            break;
                        default:
                            if (!TrySkipValue(0))
                                return false;
                            break;
                    }
                    SkipWhitespace();
                    if (Consume('}'))
                        break;
                    if (!Consume(','))
                        return false;
                    SkipWhitespace();
                }
            }
            if (phoneme == null || !hasStart || !hasEnd)
                return false;
            timing = new PhonemeTiming(phoneme, start, end);
            return true;
        }

        private bool TryReadTime(out float value)
        {
            value = 0f;
            int begin = _pos;
            if (!TryScanNumber())
                return false;
            // The token already matches the JSON number grammar, so Float plus
            // the invariant culture accepts exactly it, whatever the locale.
            return float.TryParse(
                       _json.Substring(begin, _pos - begin),
                       NumberStyles.Float,
                       CultureInfo.InvariantCulture,
                       out value)
                   && !float.IsNaN(value)
                   && !float.IsInfinity(value);
        }

        // JSON number: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
        private bool TryScanNumber()
        {
            Consume('-');
            if (!Consume('0'))
            {
                if (!IsDigit(Peek()))
                    return false;
                SkipDigits();
            }
            if (Consume('.') && SkipDigits() == 0)
                return false;
            if (Consume('e') || Consume('E'))
            {
                if (!Consume('+'))
                    Consume('-');
                if (SkipDigits() == 0)
                    return false;
            }
            return true;
        }

        private bool TryReadString(out string value)
        {
            value = null;
            if (!Consume('"'))
                return false;
            var builder = new StringBuilder();
            while (_pos < _json.Length)
            {
                char c = _json[_pos++];
                if (c == '"')
                {
                    value = builder.ToString();
                    return true;
                }
                if (c < ' ')
                    return false; // control characters must be escaped
                if (c != '\\')
                {
                    builder.Append(c);
                    continue;
                }
                if (_pos >= _json.Length)
                    return false;
                char escape = _json[_pos++];
                switch (escape)
                {
                    case '"': builder.Append('"'); break;
                    case '\\': builder.Append('\\'); break;
                    case '/': builder.Append('/'); break;
                    case 'b': builder.Append('\b'); break;
                    case 'f': builder.Append('\f'); break;
                    case 'n': builder.Append('\n'); break;
                    case 'r': builder.Append('\r'); break;
                    case 't': builder.Append('\t'); break;
                    case 'u':
                        if (!TryReadHex4(out char unit))
                            return false;
                        if (char.IsHighSurrogate(unit))
                        {
                            // Only a \u-escaped low surrogate may complete the pair.
                            if (!Consume('\\') || !Consume('u')
                                || !TryReadHex4(out char low) || !char.IsLowSurrogate(low))
                                return false;
                            builder.Append(unit).Append(low);
                        }
                        else if (char.IsLowSurrogate(unit))
                        {
                            return false;
                        }
                        else
                        {
                            builder.Append(unit);
                        }
                        break;
                    default:
                        return false;
                }
            }
            return false; // unterminated string
        }

        private bool TryReadHex4(out char unit)
        {
            unit = '\0';
            if (_json.Length - _pos < 4)
                return false;
            int code = 0;
            for (int i = 0; i < 4; i++)
            {
                int digit = HexValue(_json[_pos + i]);
                if (digit < 0)
                    return false;
                code = (code << 4) | digit;
            }
            _pos += 4;
            unit = (char)code;
            return true;
        }

        private bool TrySkipValue(int depth)
        {
            if (depth > MaxSkipDepth)
                return false;
            switch (Peek())
            {
                case '"':
                    return TryReadString(out _);
                case '{':
                    _pos++;
                    SkipWhitespace();
                    if (Consume('}'))
                        return true;
                    while (true)
                    {
                        if (!TryReadString(out _))
                            return false;
                        SkipWhitespace();
                        if (!Consume(':'))
                            return false;
                        SkipWhitespace();
                        if (!TrySkipValue(depth + 1))
                            return false;
                        SkipWhitespace();
                        if (Consume('}'))
                            return true;
                        if (!Consume(','))
                            return false;
                        SkipWhitespace();
                    }
                case '[':
                    _pos++;
                    SkipWhitespace();
                    if (Consume(']'))
                        return true;
                    while (true)
                    {
                        if (!TrySkipValue(depth + 1))
                            return false;
                        SkipWhitespace();
                        if (Consume(']'))
                            return true;
                        if (!Consume(','))
                            return false;
                        SkipWhitespace();
                    }
                case 't':
                    return ConsumeLiteral("true");
                case 'f':
                    return ConsumeLiteral("false");
                case 'n':
                    return ConsumeLiteral("null");
                default:
                    return TryScanNumber();
            }
        }

        private bool ConsumeLiteral(string literal)
        {
            if (_json.Length - _pos < literal.Length
                || string.CompareOrdinal(_json, _pos, literal, 0, literal.Length) != 0)
                return false;
            _pos += literal.Length;
            return true;
        }

        private int Peek() => _pos < _json.Length ? _json[_pos] : -1;

        private bool Consume(char expected)
        {
            if (_pos >= _json.Length || _json[_pos] != expected)
                return false;
            _pos++;
            return true;
        }

        private void SkipWhitespace()
        {
            while (_pos < _json.Length)
            {
                char c = _json[_pos];
                if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
                    return;
                _pos++;
            }
        }

        private int SkipDigits()
        {
            int begin = _pos;
            while (IsDigit(Peek()))
                _pos++;
            return _pos - begin;
        }

        private static bool IsDigit(int c) => c >= '0' && c <= '9';

        private static int HexValue(char c)
        {
            if (c >= '0' && c <= '9')
                return c - '0';
            if (c >= 'a' && c <= 'f')
                return c - 'a' + 10;
            if (c >= 'A' && c <= 'F')
                return c - 'A' + 10;
            return -1;
        }
    }
}
