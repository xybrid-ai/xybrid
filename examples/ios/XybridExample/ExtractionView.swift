//
//  ExtractionView.swift
//  XybridExample
//
//  On-device structured data extraction with LFM2.5-230M and GBNF
//  constrained decoding. A JSON Schema is converted to a GBNF grammar
//  (`jsonSchemaToGbnf`) and attached to `XybridGenerationConfig.grammar`,
//  which guarantees the model can only emit schema-valid JSON — fully
//  offline, no API key, no post-hoc parsing or retries.
//
//  Flip "Constrain to schema" off to see what the raw 230M model does with
//  the same prompt: it typically wraps output in markdown fences and invents
//  its own keys, which fails JSON parsing. The grammar is the difference
//  between a demo and something you can ship.
//

import SwiftUI
import Xybrid

// MARK: - Presets

struct ExtractionPreset: Identifiable {
    let id: String
    let name: String
    let input: String
    let schemaJson: String
}

private let extractionPresets: [ExtractionPreset] = [
    ExtractionPreset(
        id: "receipt",
        name: "Receipt",
        input: """
        STARBUCKS STORE #1123
        2x Latte         9.00
        1x Croissant     3.50
        TOTAL           12.50 USD
        03/15/2026
        """,
        schemaJson: """
        {
          "type": "object",
          "properties": {
            "merchant": { "type": "string" },
            "total":    { "type": "number" },
            "currency": { "enum": ["USD", "EUR", "GBP"] },
            "date":     { "type": "string" },
            "items":    { "type": "array", "items": { "type": "string" } }
          }
        }
        """
    ),
    ExtractionPreset(
        id: "contact",
        name: "Signature",
        input: """
        Thanks so much, talk soon!

        --
        Dr. Maria N. Alvarez
        Head of Radiology, St. Vincent Hospital
        maria.alvarez@stvincent.org | +33 6 12 34 56 78
        Paris, France
        """,
        schemaJson: """
        {
          "type": "object",
          "properties": {
            "name":    { "type": "string" },
            "title":   { "type": "string" },
            "company": { "type": "string" },
            "email":   { "type": "string" },
            "phone":   { "type": "string" },
            "city":    { "type": "string" }
          }
        }
        """
    ),
]

// MARK: - Extraction View

struct ExtractionView: View {
    private let modelId = "lfm2.5-230m"

    @State private var presetId: String = extractionPresets[0].id
    @State private var inputText: String = extractionPresets[0].input
    @State private var constrained: Bool = true
    @State private var model: XybridModel? = nil
    @State private var inferenceState: InferenceState = .idle

    private var preset: ExtractionPreset {
        extractionPresets.first { $0.id == presetId } ?? extractionPresets[0]
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Model loading
                VStack(alignment: .leading, spacing: 12) {
                    Text("Model")
                        .font(.headline)

                    HStack {
                        Text(modelId)
                            .font(.system(.body, design: .monospaced))
                        Spacer()
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
                            Text(model != nil ? "Model Loaded" : "Load Model (~146 MB)")
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isLoadingOrRunning || model != nil)
                }

                Divider()

                // Input
                VStack(alignment: .leading, spacing: 12) {
                    Text("Input")
                        .font(.headline)

                    Picker("Preset", selection: $presetId) {
                        ForEach(extractionPresets) { p in
                            Text(p.name).tag(p.id)
                        }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: presetId) { newValue in
                        if let p = extractionPresets.first(where: { $0.id == newValue }) {
                            inputText = p.input
                        }
                    }

                    TextEditor(text: $inputText)
                        .font(.system(.footnote, design: .monospaced))
                        .frame(minHeight: 110, maxHeight: 150)
                        .padding(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                        )

                    DisclosureGroup("JSON Schema") {
                        Text(preset.schemaJson)
                            .font(.system(.caption2, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(8)
                            .background(Color.secondary.opacity(0.1))
                            .cornerRadius(4)
                    }
                    .font(.subheadline)

                    Toggle(isOn: $constrained) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Constrain to schema (GBNF)")
                                .font(.subheadline)
                            Text(constrained
                                ? "Output is guaranteed schema-valid JSON"
                                : "Raw model output — expect parse failures")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    Button(action: runExtraction) {
                        HStack {
                            if case .running = inferenceState {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.8)
                            }
                            Text(isRunning ? "Extracting…" : "Extract")
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model == nil || isLoadingOrRunning || inputText.isEmpty)
                }

                // Result
                if case .completed(let result) = inferenceState {
                    Divider()
                    ExtractionResultView(result: result)
                }

                // Error
                if case .error(let message) = inferenceState {
                    Divider()
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
                }

                Spacer(minLength: 40)
            }
            .padding(.horizontal)
            .padding(.top, 16)
        }
        .navigationTitle("Structured Extraction")
    }

    // MARK: - Computed

    private var isLoadingOrRunning: Bool {
        switch inferenceState {
        case .loading, .running:
            return true
        default:
            return false
        }
    }

    private var isRunning: Bool {
        if case .running = inferenceState { return true }
        return false
    }

    // MARK: - Actions

    private func loadModel() {
        inferenceState = .loading
        let modelId = self.modelId

        // The loader is cheap to create; its async load performs resolution,
        // download, disk access, and runtime initialization off the main actor.
        Task {
            do {
                let loadedModel = try await Xybrid.model(modelId).load()
                await MainActor.run {
                    self.model = loadedModel
                    inferenceState = .idle
                }
            } catch {
                await MainActor.run {
                    inferenceState = .error("Failed to load model: \(error.localizedDescription)")
                }
            }
        }
    }

    private func runExtraction() {
        guard let model = model else { return }
        inferenceState = .running

        // Capture @State inputs on the main actor before detaching.
        let inputText = self.inputText
        let schemaJson = preset.schemaJson
        let constrained = self.constrained

        Task.detached {
            do {
                // JSON Schema → GBNF. This is the whole feature: with the
                // grammar attached, sampling masks every token that would
                // take the output off a schema-valid path.
                let grammar: String? = constrained
                    ? try jsonSchemaToGbnf(schemaJson: schemaJson)
                    : nil

                // Greedy decoding (temperature 0) — extraction wants the
                // most likely tokens, reproducibly, not creative sampling.
                let config = XybridGenerationConfig.greedy(
                    maxTokens: 200,
                    grammar: grammar
                )
                let options = XybridRunOptions(
                    generationConfig: config,
                    abortOn: [],
                    fallbackToCloud: false,
                    maxGraceTokens: 0,
                    correlationId: nil
                )

                let envelope = XybridEnvelope(
                    kind: .text(text: "Extract the fields from this text:\n" + inputText),
                    metadata: [
                        XybridMetadataEntry(
                            key: "system_prompt",
                            value: "You extract structured data. Respond with a single JSON object, nothing else."
                        )
                    ]
                )

                let result = try model.run(envelope: envelope, options: options)
                await MainActor.run {
                    inferenceState = .completed(result)
                }
            } catch {
                await MainActor.run {
                    inferenceState = .error("Extraction failed: \(error.localizedDescription)")
                }
            }
        }
    }
}

// MARK: - Result View

struct ExtractionResultView: View {
    let result: XybridResult

    /// Parsed key/value pairs when the output is a valid JSON object.
    /// Stored, not computed: `body` reads this more than once per render,
    /// and a computed property would re-run JSONSerialization each time.
    private let parsedFields: [(String, String)]?

    init(result: XybridResult) {
        self.result = result
        if let text = result.text,
           let data = text.trimmingCharacters(in: .whitespacesAndNewlines).data(using: .utf8),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            self.parsedFields = object
                .map { key, value in (key, Self.stringify(value)) }
                .sorted { $0.0 < $1.0 }
        } else {
            self.parsedFields = nil
        }
    }

    private static func stringify(_ value: Any) -> String {
        if let array = value as? [Any] {
            return array.map { "\($0)" }.joined(separator: ", ")
        }
        return "\(value)"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Result", systemImage: "doc.text.magnifyingglass")
                    .font(.headline)

                Spacer()

                // The badge the demo hinges on: schema-constrained runs are
                // always valid; unconstrained runs usually aren't.
                if parsedFields != nil {
                    Label("Valid JSON", systemImage: "checkmark.seal.fill")
                        .font(.caption)
                        .foregroundColor(.green)
                } else {
                    Label("Not valid JSON", systemImage: "xmark.seal.fill")
                        .font(.caption)
                        .foregroundColor(.red)
                }
            }

            if let fields = parsedFields {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(fields, id: \.0) { field in
                        HStack(alignment: .top) {
                            Text(field.0)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.secondary)
                                .frame(width: 90, alignment: .leading)
                            Text(field.1)
                                .font(.system(.caption, design: .monospaced))
                                .fontWeight(.medium)
                        }
                    }
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.green.opacity(0.1))
                .cornerRadius(8)
            }

            if let text = result.text {
                DisclosureGroup("Raw output") {
                    Text(text)
                        .font(.system(.caption2, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                        .background(Color.secondary.opacity(0.1))
                        .cornerRadius(4)
                }
                .font(.subheadline)
            }

            MetricsSection(metrics: result.metrics)
        }
    }
}

// MARK: - Preview

struct ExtractionView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView { ExtractionView() }
    }
}
