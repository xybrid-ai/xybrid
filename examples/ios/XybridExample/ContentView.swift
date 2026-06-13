//
//  ContentView.swift
//  XybridExample
//
//  Demonstrates Xybrid SDK usage for iOS.
//  Requires the XCFramework to be built first:
//    cargo xtask build-xcframework
//  Then add the Xybrid Swift Package dependency (../../bindings/apple) in Xcode.
//

import SwiftUI
import AVFoundation
import Xybrid

// MARK: - App State

enum AppState {
    case notInitialized
    case initializing
    case ready
    case error(String)
}

enum InferenceState {
    case idle
    case loading
    case running
    case completed(XybridResult)
    case error(String)
}

// MARK: - Main Content View

struct ContentView: View {
    @State private var appState: AppState = .notInitialized

    var body: some View {
        switch appState {
        case .notInitialized:
            navigation { WelcomeView(onInitialize: initializeSDK) }
        case .initializing:
            navigation { LoadingView(message: "Initializing Xybrid SDK...") }
        case .ready:
            TabView {
                navigation { InferenceView() }
                    .tabItem { Label("Speech", systemImage: "waveform") }
                navigation { LiveVisionView() }
                    .tabItem { Label("Vision", systemImage: "camera.viewfinder") }
            }
        case .error(let message):
            navigation { ErrorView(message: message, onRetry: initializeSDK) }
        }
    }

    /// Wraps a screen in the example's standard stacked-navigation chrome.
    @ViewBuilder
    private func navigation<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        NavigationView {
            content()
                .navigationBarTitleDisplayMode(.inline)
        }
        .navigationViewStyle(.stack)
    }

    private func initializeSDK() {
        appState = .initializing
        Task {
            // Initialize the SDK cache directory
            let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)
                .first!.appendingPathComponent("xybrid").path
            initSdkCacheDir(cacheDir: cacheDir)

            // The key and platform URL come from the XYBRID_API_KEY and
            // XYBRID_PLATFORM_URL scheme environment variables (Product →
            // Scheme → Edit Scheme → Run → Arguments), so they never land in
            // the repo. Empty/unset resolves to anonymous, local-only init
            // against the default platform. Get a free key at
            // dashboard.xybrid.dev. See README.
            let env = ProcessInfo.processInfo.environment
            let apiKey = env["XYBRID_API_KEY"]
            let platformUrl = env["XYBRID_PLATFORM_URL"]
            Xybrid.initialize(
                apiKey: (apiKey ?? "").isEmpty ? nil : apiKey,
                ingestUrl: (platformUrl ?? "").isEmpty ? nil : platformUrl
            )

            await MainActor.run {
                appState = .ready
            }
        }
    }
}

// MARK: - Welcome View

struct WelcomeView: View {
    let onInitialize: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "waveform.circle.fill")
                .font(.system(size: 72))
                .foregroundColor(.blue)

            Text("Xybrid SDK Example")
                .font(.largeTitle)
                .fontWeight(.bold)

            Text("iOS Reference Implementation")
                .font(.subheadline)
                .foregroundColor(.secondary)

            Spacer()

            Button(action: onInitialize) {
                Label("Initialize SDK", systemImage: "play.circle.fill")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal, 24)
            .padding(.bottom, 40)
        }
    }
}

// MARK: - Loading View

struct LoadingView: View {
    let message: String

    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.5)
            Text(message)
                .font(.headline)
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Error View

struct ErrorView: View {
    let message: String
    let onRetry: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 48))
                .foregroundColor(.orange)

            Text("Initialization Failed")
                .font(.headline)

            Text(message)
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            Button(action: onRetry) {
                Label("Retry", systemImage: "arrow.clockwise")
                    .font(.headline)
            }
            .buttonStyle(.borderedProminent)
        }
    }
}

// MARK: - Inference View

struct InferenceView: View {
    @State private var inputText: String = "Hello, welcome to Xybrid!"
    @State private var modelId: String = "kokoro-82m"
    @State private var voiceId: String = "af"
    @State private var inferenceState: InferenceState = .idle
    @State private var model: XybridModel? = nil
    @State private var voices: [XybridVoiceInfo]? = nil
    @State private var audioPlayer: AVAudioPlayer? = nil

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Header
                VStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 36))
                        .foregroundColor(.green)

                    Text("SDK Ready")
                        .font(.headline)
                        .foregroundColor(.green)
                }
                .padding(.top, 20)

                // Model Loading Section
                VStack(alignment: .leading, spacing: 12) {
                    Text("Model Configuration")
                        .font(.headline)

                    HStack {
                        TextField("Model ID", text: $modelId)
                            .textFieldStyle(.roundedBorder)
                            .autocapitalization(.none)
                            .disableAutocorrection(true)

                        if model != nil {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                        }
                    }

                    Button(action: loadModel) {
                        HStack {
                            if case .loading = inferenceState {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.8)
                            }
                            Text(model != nil ? "Model Loaded" : "Load Model")
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isLoadingOrRunning || model != nil)
                }
                .padding(.horizontal)

                // Voice Picker (shown after model loads with voice support)
                if let voices = voices, !voices.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Voice")
                            .font(.subheadline)
                            .foregroundColor(.secondary)

                        Picker("Voice", selection: $voiceId) {
                            ForEach(voices, id: \.id) { voice in
                                Text("\(voice.name) (\(voice.id))")
                                    .tag(voice.id)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                    .padding(.horizontal)
                }

                Divider()
                    .padding(.horizontal)

                // Inference Section
                VStack(alignment: .leading, spacing: 12) {
                    Text("Text-to-Speech Inference")
                        .font(.headline)

                    Text("Input Text")
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    TextEditor(text: $inputText)
                        .frame(minHeight: 80, maxHeight: 120)
                        .padding(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                        )

                    Button(action: runInference) {
                        HStack {
                            if case .running = inferenceState {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.8)
                            }
                            Text(isRunningInference ? "Running..." : "Run Inference")
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model == nil || isLoadingOrRunning || inputText.isEmpty)
                }
                .padding(.horizontal)

                // Results Section
                if case .completed(let result) = inferenceState {
                    Divider()
                        .padding(.horizontal)

                    ResultView(result: result, onPlay: playAudio)
                        .padding(.horizontal)
                }

                // Error Section
                if case .error(let message) = inferenceState {
                    Divider()
                        .padding(.horizontal)

                    VStack(alignment: .leading, spacing: 8) {
                        Label("Error", systemImage: "exclamationmark.triangle.fill")
                            .font(.headline)
                            .foregroundColor(.red)

                        Text(message)
                            .font(.body)
                            .foregroundColor(.secondary)
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.red.opacity(0.1))
                            .cornerRadius(8)
                    }
                    .padding(.horizontal)
                }

                Spacer(minLength: 40)
            }
        }
        .navigationTitle("Inference Demo")
    }

    // MARK: - Computed Properties

    private var isLoadingOrRunning: Bool {
        switch inferenceState {
        case .loading, .running:
            return true
        default:
            return false
        }
    }

    private var isRunningInference: Bool {
        if case .running = inferenceState {
            return true
        }
        return false
    }

    // MARK: - Actions

    private func loadModel() {
        inferenceState = .loading

        // Capture the @State input on the main actor before detaching.
        let modelId = self.modelId

        // `Task.detached`, NOT `Task {}`: this method runs on the main
        // actor (SwiftUI View), and a plain `Task` inherits that
        // executor — the synchronous, blocking `XybridModel(fromRegistry:)`
        // (model resolve + download + load) would run on the main thread
        // and freeze the UI. Detaching runs it on a background executor.
        Task.detached {
            do {
                let loadedModel = try XybridModel(fromRegistry: modelId)
                let modelVoices = loadedModel.voices()
                let defaultVoice = loadedModel.defaultVoice()?.id

                await MainActor.run {
                    self.model = loadedModel
                    self.voices = modelVoices.isEmpty ? nil : modelVoices
                    if let defaultVoice = defaultVoice {
                        self.voiceId = defaultVoice
                    }
                    inferenceState = .idle
                }
            } catch {
                await MainActor.run {
                    inferenceState = .error("Failed to load model: \(error.localizedDescription)")
                }
            }
        }
    }

    private func runInference() {
        guard let model = model else { return }

        inferenceState = .running

        // Capture the @State inputs on the main actor before detaching.
        let inputText = self.inputText
        let voiceId = self.voiceId

        // `Task.detached` for the same reason as `loadModel`: bolt's
        // `run` is synchronous + blocking, and a plain `Task` from this
        // main-actor method would run it on the main thread and freeze
        // the UI for the duration of inference.
        Task.detached {
            do {
                let envelope = XybridEnvelope.text(
                    text: inputText,
                    voiceId: voiceId,
                    speed: 1.0
                )
                let result = try model.run(envelope: envelope)

                await MainActor.run {
                    inferenceState = .completed(result)
                }
            } catch {
                await MainActor.run {
                    inferenceState = .error("Inference failed: \(error.localizedDescription)")
                }
            }
        }
    }

    private func playAudio() {
        guard case .completed(let result) = inferenceState,
              let audioBytes = result.audioBytes else { return }

        // Build a WAV header for raw PCM data (24000 Hz, 16-bit mono)
        let wavData = buildWavData(
            pcm: Data(audioBytes),
            sampleRate: 24000,
            channels: 1,
            bitsPerSample: 16
        )

        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            audioPlayer = try AVAudioPlayer(data: wavData)
            audioPlayer?.play()
        } catch {
            print("Audio playback error: \(error)")
        }
    }

    private func buildWavData(pcm: Data, sampleRate: UInt32, channels: UInt16, bitsPerSample: UInt16) -> Data {
        var data = Data()
        let dataSize = UInt32(pcm.count)
        let byteRate = sampleRate * UInt32(channels) * UInt32(bitsPerSample / 8)
        let blockAlign = channels * (bitsPerSample / 8)

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        data.append(withUnsafeBytes(of: (36 + dataSize).littleEndian) { Data($0) })
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        data.append(withUnsafeBytes(of: UInt32(16).littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: UInt16(1).littleEndian) { Data($0) }) // PCM
        data.append(withUnsafeBytes(of: channels.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: sampleRate.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: byteRate.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: blockAlign.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: bitsPerSample.littleEndian) { Data($0) })

        // data chunk
        data.append(contentsOf: "data".utf8)
        data.append(withUnsafeBytes(of: dataSize.littleEndian) { Data($0) })
        data.append(pcm)

        return data
    }
}

// MARK: - Result View

struct ResultView: View {
    let result: XybridResult
    let onPlay: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Result", systemImage: "checkmark.circle.fill")
                .font(.headline)
                .foregroundColor(.green)

            VStack(alignment: .leading, spacing: 8) {
                // Status
                HStack {
                    Text("Status:")
                        .fontWeight(.medium)
                    Text(result.success ? "Success" : "Failed")
                        .foregroundColor(result.success ? .green : .red)
                }

                // Latency
                HStack {
                    Text("Latency:")
                        .fontWeight(.medium)
                    Text("\(result.latencyMs) ms")
                        .foregroundColor(.blue)
                }

                // Output text
                if let text = result.text {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Output:")
                            .fontWeight(.medium)
                        Text(text)
                            .foregroundColor(.secondary)
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.secondary.opacity(0.1))
                            .cornerRadius(4)
                    }
                }

                // Audio size
                if let audioBytes = result.audioBytes {
                    HStack {
                        Text("Audio:")
                            .fontWeight(.medium)
                        Text("\(audioBytes.count / 1024) KB")
                            .foregroundColor(.secondary)
                    }
                }

                // Typed metrics section — populated from result.metrics.
                // LLM-specific fields are nil for TTS/ASR; stage latencies
                // are empty for single-model runs.
                MetricsSection(metrics: result.metrics)

                // Play Button
                if result.success && result.audioBytes != nil {
                    Button(action: onPlay) {
                        Label("Play Audio", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                    }
                    .buttonStyle(.bordered)
                    .padding(.top, 8)
                }
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.green.opacity(0.1))
            .cornerRadius(8)
        }
    }
}

// MARK: - Metrics Section

struct MetricsSection: View {
    let metrics: XybridInferenceMetrics

    var rows: [(String, String)] {
        var out: [(String, String)] = []
        if let ttft = metrics.ttftMs { out.append(("TTFT", "\(ttft) ms")) }
        if let tps = metrics.tokensPerSecond {
            out.append(("Throughput", String(format: "%.1f tok/s", tps)))
        }
        if let p = metrics.prefillTps {
            out.append(("Prefill", String(format: "%.1f tok/s", p)))
        }
        if let d = metrics.decodeTps {
            out.append(("Decode", String(format: "%.1f tok/s", d)))
        }
        if let t = metrics.tokensOut { out.append(("Tokens out", "\(t)")) }
        if !metrics.stageLatenciesMs.isEmpty {
            let s = metrics.stageLatenciesMs.map { "\($0.stageId)=\($0.latencyMs)ms" }.joined(separator: ", ")
            out.append(("Stages", s))
        }
        return out
    }

    var body: some View {
        if !rows.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                Text("Metrics")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .padding(.top, 8)
                ForEach(rows, id: \.0) { row in
                    HStack {
                        Text(row.0)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(row.1)
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                }
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.secondary.opacity(0.1))
            .cornerRadius(4)
        }
    }
}

// MARK: - Preview

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
