//
//  ContentView.swift
//  XybridExample
//
//  Demonstrates Xybrid SDK usage for iOS.
//
//  NOTE: The Xybrid import requires the XCFramework to be built first.
//  Run `cargo xtask build-xcframework` from the xybrid repo root to build it.
//  Then uncomment `import Xybrid` below to enable SDK functionality.
//

import SwiftUI
// import Xybrid  // Uncomment after building XCFramework

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
    case completed(InferenceResult)
    case error(String)
}

struct InferenceResult {
    let text: String?
    let latencyMs: Int
    let success: Bool
}

// MARK: - Main Content View

struct ContentView: View {
    @State private var appState: AppState = .notInitialized

    var body: some View {
        NavigationView {
            Group {
                switch appState {
                case .notInitialized:
                    WelcomeView(onInitialize: initializeSDK)
                case .initializing:
                    LoadingView(message: "Initializing Xybrid SDK...")
                case .ready:
                    InferenceView()
                case .error(let message):
                    ErrorView(message: message, onRetry: initializeSDK)
                }
            }
            .navigationBarTitleDisplayMode(.inline)
        }
        .navigationViewStyle(.stack)
    }

    private func initializeSDK() {
        appState = .initializing

        // NOTE: Replace with actual Xybrid SDK initialization after XCFramework is built
        // Example with real SDK:
        // Task {
        //     do {
        //         try await Xybrid.initialize()
        //         await MainActor.run {
        //             appState = .ready
        //         }
        //     } catch {
        //         await MainActor.run {
        //             appState = .error(error.localizedDescription)
        //         }
        //     }
        // }

        // Simulated initialization for demo purposes
        Task {
            try? await Task.sleep(nanoseconds: 800_000_000) // 0.8 seconds
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
    @State private var inferenceState: InferenceState = .idle
    @State private var modelLoaded: Bool = false

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

                        if modelLoaded {
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
                            Text(modelLoaded ? "Model Loaded" : "Load Model")
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isLoadingOrRunning || modelLoaded)
                }
                .padding(.horizontal)

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
                    .disabled(!modelLoaded || isLoadingOrRunning || inputText.isEmpty)
                }
                .padding(.horizontal)

                // Results Section
                if let result = inferenceResult {
                    Divider()
                        .padding(.horizontal)

                    ResultView(result: result)
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

    private var inferenceResult: InferenceResult? {
        if case .completed(let result) = inferenceState {
            return result
        }
        return nil
    }

    // MARK: - Actions

    private func loadModel() {
        inferenceState = .loading

        // NOTE: Replace with actual model loading after XCFramework is built
        // Example with real SDK:
        // Task {
        //     do {
        //         let loader = XybridModelLoader.fromRegistry(modelId: modelId)
        //         let model = try await loader.load()
        //         await MainActor.run {
        //             self.model = model
        //             modelLoaded = true
        //             inferenceState = .idle
        //         }
        //     } catch {
        //         await MainActor.run {
        //             inferenceState = .error("Failed to load model: \(error.localizedDescription)")
        //         }
        //     }
        // }

        // Simulated model loading for demo purposes
        Task {
            try? await Task.sleep(nanoseconds: 1_200_000_000) // 1.2 seconds
            await MainActor.run {
                modelLoaded = true
                inferenceState = .idle
            }
        }
    }

    private func runInference() {
        guard modelLoaded else { return }

        inferenceState = .running
        let startTime = CFAbsoluteTimeGetCurrent()

        // NOTE: Replace with actual inference after XCFramework is built
        // Example with real SDK:
        // Task {
        //     do {
        //         let envelope = XybridEnvelope.text(
        //             text: inputText,
        //             voiceId: "af",
        //             speed: 1.0
        //         )
        //         let result = try await model.run(envelope: envelope)
        //         let latencyMs = Int((CFAbsoluteTimeGetCurrent() - startTime) * 1000)
        //
        //         await MainActor.run {
        //             inferenceState = .completed(InferenceResult(
        //                 text: result.text,
        //                 latencyMs: latencyMs,
        //                 success: result.success
        //             ))
        //         }
        //     } catch {
        //         await MainActor.run {
        //             inferenceState = .error("Inference failed: \(error.localizedDescription)")
        //         }
        //     }
        // }

        // Simulated inference for demo purposes
        Task {
            // Simulate processing time
            let processingTime = UInt64.random(in: 300_000_000...800_000_000)
            try? await Task.sleep(nanoseconds: processingTime)

            let latencyMs = Int((CFAbsoluteTimeGetCurrent() - startTime) * 1000)

            await MainActor.run {
                inferenceState = .completed(InferenceResult(
                    text: "Audio generated (\(inputText.count) characters)",
                    latencyMs: latencyMs,
                    success: true
                ))
            }
        }
    }
}

// MARK: - Result View

struct ResultView: View {
    let result: InferenceResult

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

                // Output
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

                // Play Button (placeholder for audio playback)
                if result.success {
                    Button(action: playAudio) {
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

    private func playAudio() {
        // NOTE: Implement audio playback after XCFramework is built
        // Example:
        // if let audioData = result.audioBytes {
        //     let player = try? AVAudioPlayer(data: audioData)
        //     player?.play()
        // }
        print("Play audio tapped - requires XCFramework for actual playback")
    }
}

// MARK: - Preview

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
