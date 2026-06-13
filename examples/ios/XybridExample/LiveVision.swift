//
//  LiveVision.swift
//  XybridExample
//
//  Live-camera visual Q&A demo. AVFoundation captures frames; a cheap luma
//  change-gate wakes the VLM only when the scene changes; each gated frame is
//  sent as a single *batch* multimodal turn:
//
//      XybridEnvelope.userMessage(text: prompt, images: [.image(jpeg, "jpeg")])
//          → model.run(envelope:config:)  →  caption
//
//  Responsiveness model — drop-if-busy (not cancel-and-replace):
//  The Swift / UniFFI bindings currently expose only the batch `run`. Streaming
//  tokens, cancel-and-replace, and raw-frame (`imageRaw`) envelopes — the
//  realtime-vision primitives already in the Flutter SDK — are not yet bound for
//  Swift. So while one frame is being answered, newly gated frames are *dropped*
//  rather than preempting the in-flight run. When the streaming/cancellation
//  surface lands for Swift this loop can move to latest-frame-wins.
//

import AVFoundation
import CoreImage
import SwiftUI
import UIKit
import Xybrid

// MARK: - Frame

struct LiveVisionFrame {
    let jpegData: Data
    let capturedAt: Date
    /// Downsampled 16×12 luma grid — a cheap signature for frame-difference gating.
    let lumaSignature: [UInt8]
}

// MARK: - Camera service

/// Thin AVFoundation wrapper: configures a back-camera BGRA feed and hands each
/// frame (as JPEG + a luma signature) to a handler. All capture work runs off
/// the main thread; the handler is invoked on the video queue.
final class LiveCameraService: NSObject, @unchecked Sendable {
    enum AuthorizationState { case notDetermined, authorized, denied }

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "dev.xybrid.example.camera.session")
    private let videoQueue = DispatchQueue(label: "dev.xybrid.example.camera.video")
    private var isConfigured = false
    private var frameHandler: ((LiveVisionFrame) -> Void)?

    func authorizationState() -> AuthorizationState {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return .authorized
        case .notDetermined: return .notDetermined
        default: return .denied
        }
    }

    func requestAccess() async -> AuthorizationState {
        switch authorizationState() {
        case .authorized: return .authorized
        case .denied: return .denied
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            return granted ? .authorized : .denied
        }
    }

    func setFrameHandler(_ handler: @escaping (LiveVisionFrame) -> Void) {
        sessionQueue.async { self.frameHandler = handler }
    }

    func start() async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            sessionQueue.async {
                do {
                    try self.configureIfNeeded()
                    if !self.session.isRunning { self.session.startRunning() }
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func stop() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            sessionQueue.async {
                if self.session.isRunning { self.session.stopRunning() }
                continuation.resume()
            }
        }
    }

    private func configureIfNeeded() throws {
        guard !isConfigured else { return }

        session.beginConfiguration()
        session.sessionPreset = .vga640x480
        defer { session.commitConfiguration() }

        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
                ?? AVCaptureDevice.default(for: .video),
            let input = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input)
        else {
            throw NSError(
                domain: "LiveCameraService", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No usable camera on this device."]
            )
        }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(output) else {
            throw NSError(
                domain: "LiveCameraService", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Could not add the camera output."]
            )
        }
        session.addOutput(output)

        if let connection = output.connection(with: .video), connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }
        isConfigured = true
    }

    private static let ciContext = CIContext()
}

extension LiveCameraService: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard
            let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
            let jpeg = Self.jpegData(from: pixelBuffer)
        else { return }

        frameHandler?(
            LiveVisionFrame(
                jpegData: jpeg,
                capturedAt: Date(),
                lumaSignature: Self.lumaSignature(from: pixelBuffer)
            )
        )
    }

    private static func jpegData(from pixelBuffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        return UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.7)
    }

    private static func lumaSignature(from pixelBuffer: CVPixelBuffer) -> [UInt8] {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return [] }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let ptr = base.assumingMemoryBound(to: UInt8.self)
        let columns = 16, rows = 12

        return (0..<rows).flatMap { row -> [UInt8] in
            (0..<columns).map { column in
                let x = min(width - 1, max(0, ((column * width) + (width / 2)) / columns))
                let y = min(height - 1, max(0, ((row * height) + (height / 2)) / rows))
                let offset = (y * bytesPerRow) + (x * 4)   // BGRA
                let b = UInt16(ptr[offset]), g = UInt16(ptr[offset + 1]), r = UInt16(ptr[offset + 2])
                return UInt8((54 * r + 183 * g + 19 * b) >> 8)
            }
        }
    }
}

// MARK: - Camera preview

/// SwiftUI bridge for the live viewfinder.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

// MARK: - History

struct LiveVisionHistoryItem: Identifiable {
    let id = UUID()
    let question: String
    let answer: String
    let latencyMs: UInt32?
    let isError: Bool
}

// MARK: - View model

@MainActor
final class LiveVisionViewModel: ObservableObject {
    /// Any vision-capable (VLM / mtmd) model id resolvable from the registry.
    @Published var modelId = "lfm2-vl-450m"
    @Published var question = "What do you see? Describe the scene briefly."
    @Published private(set) var isCameraRunning = false
    @Published private(set) var isBusy = false
    @Published private(set) var status = "Idle"
    @Published private(set) var latestAnswer = "No vision output yet."
    @Published private(set) var errorMessage: String?
    @Published private(set) var acceptedFrames = 0
    @Published private(set) var droppedBusyFrames = 0
    @Published private(set) var droppedGateFrames = 0
    @Published private(set) var history: [LiveVisionHistoryItem] = []

    let camera = LiveCameraService()

    // Gate tuning: fire at most once per ~2.5 s, and only when the scene changed
    // enough (mean per-cell luma delta over the 16×12 grid).
    private let minInferenceInterval: TimeInterval = 2.5
    private let changeThreshold = 7.5
    private let maxHistory = 8

    private var model: XybridModel?
    private var loadedModelId: String?
    private var lastAcceptedAt: Date?
    private var lastSignature: [UInt8]?
    /// Bumped on stop/reset so an in-flight result that lands afterwards is dropped.
    private var generation = 0

    init() {
        camera.setFrameHandler { [weak self] frame in
            Task { @MainActor in self?.handle(frame: frame) }
        }
    }

    func start() async {
        guard !isCameraRunning else { return }
        errorMessage = nil
        status = "Requesting camera…"
        guard await camera.requestAccess() == .authorized else {
            status = "Camera permission required"
            latestAnswer = "Enable camera access in Settings to use live vision."
            return
        }
        do {
            try await camera.start()
            generation += 1
            isCameraRunning = true
            status = "Watching for scene changes"
        } catch {
            errorMessage = error.localizedDescription
            status = "Camera failed"
        }
    }

    func stop() async {
        guard isCameraRunning else { return }
        generation += 1
        isCameraRunning = false
        status = "Camera stopped"
        await camera.stop()
    }

    func reset() {
        acceptedFrames = 0
        droppedBusyFrames = 0
        droppedGateFrames = 0
        history.removeAll()
        lastAcceptedAt = nil
        lastSignature = nil
        latestAnswer = "No vision output yet."
        errorMessage = nil
        status = isCameraRunning ? "Watching for scene changes" : "Idle"
    }

    private func handle(frame: LiveVisionFrame) {
        guard isCameraRunning else { return }

        let prompt = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            status = "Enter a question first"
            droppedGateFrames += 1
            return
        }
        // Drop-if-busy (no streaming/cancel binding for Swift yet — see header).
        guard !isBusy else { droppedBusyFrames += 1; return }

        if let last = lastAcceptedAt, frame.capturedAt.timeIntervalSince(last) < minInferenceInterval {
            droppedGateFrames += 1
            return
        }
        if let prev = lastSignature, !sceneChanged(from: prev, to: frame.lumaSignature) {
            droppedGateFrames += 1
            return
        }

        lastAcceptedAt = frame.capturedAt
        lastSignature = frame.lumaSignature
        acceptedFrames += 1
        runInference(frame: frame, prompt: prompt, generation: generation)
    }

    private func runInference(frame: LiveVisionFrame, prompt: String, generation: Int) {
        isBusy = true
        status = "Running local VLM…"
        latestAnswer = "Thinking…"
        errorMessage = nil

        Task {
            let item = await infer(frame: frame, prompt: prompt)
            isBusy = false
            // Camera stopped / reset while running — drop the stale result.
            guard generation == self.generation else {
                status = isCameraRunning ? "Watching for scene changes" : "Camera stopped"
                return
            }
            latestAnswer = item.answer
            errorMessage = item.isError ? item.answer : nil
            history.insert(item, at: 0)
            if history.count > maxHistory { history.removeLast(history.count - maxHistory) }
            status = isCameraRunning ? "Watching for scene changes" : "Camera stopped"
        }
    }

    private func infer(frame: LiveVisionFrame, prompt: String) async -> LiveVisionHistoryItem {
        do {
            let model = try await ensureModel()
            let image = XybridEnvelope.image(bytes: frame.jpegData, format: "jpeg")
            let envelope = XybridEnvelope.userMessage(text: prompt, images: [image])
            let config = XybridGenerationConfig(
                maxTokens: 96,
                temperature: 0.0,
                topP: 0.9,
                minP: 0.05,
                topK: 40,
                repetitionPenalty: 1.05,
                stopSequences: []
            )
            let result = try await model.run(envelope: envelope, config: config)
            let text = result.text?.trimmingCharacters(in: .whitespacesAndNewlines)
            return LiveVisionHistoryItem(
                question: prompt,
                answer: (text?.isEmpty == false) ? text! : (result.success ? "(no text returned)" : "Inference failed"),
                latencyMs: result.latencyMs,
                isError: !result.success
            )
        } catch {
            return LiveVisionHistoryItem(
                question: prompt,
                answer: error.localizedDescription,
                latencyMs: nil,
                isError: true
            )
        }
    }

    private func ensureModel() async throws -> XybridModel {
        if let model, loadedModelId == modelId { return model }
        status = "Loading \(modelId)…"
        let loaded = try await XybridModelLoader.fromRegistry(modelId: modelId).load()
        model = loaded
        loadedModelId = modelId
        return loaded
    }

    private func sceneChanged(from old: [UInt8], to new: [UInt8]) -> Bool {
        guard old.count == new.count, !old.isEmpty else { return true }
        let total = zip(old, new).reduce(0) { $0 + abs(Int($1.0) - Int($1.1)) }
        return Double(total) / Double(old.count) >= changeThreshold
    }
}

// MARK: - View

struct LiveVisionView: View {
    @StateObject private var vm = LiveVisionViewModel()

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                viewfinder
                configFields
                controls
                statusAndAnswer
                historyList
                Spacer(minLength: 32)
            }
            .padding(.top, 12)
        }
        .navigationTitle("Live Vision")
        .onDisappear { Task { await vm.stop() } }
    }

    private var viewfinder: some View {
        ZStack {
            if vm.isCameraRunning {
                CameraPreview(session: vm.camera.session)
                    .aspectRatio(3.0 / 4.0, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.secondary.opacity(0.15))
                    .aspectRatio(3.0 / 4.0, contentMode: .fit)
                    .overlay(
                        VStack(spacing: 8) {
                            Image(systemName: "camera.viewfinder")
                                .font(.system(size: 44))
                                .foregroundColor(.secondary)
                            Text("Camera off").foregroundColor(.secondary)
                        }
                    )
            }
        }
        .padding(.horizontal)
    }

    private var configFields: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Vision model")
                .font(.subheadline)
                .foregroundColor(.secondary)
            TextField("Vision model id", text: $vm.modelId)
                .textFieldStyle(.roundedBorder)
                .autocapitalization(.none)
                .disableAutocorrection(true)
                .disabled(vm.isCameraRunning)

            Text("Question")
                .font(.subheadline)
                .foregroundColor(.secondary)
            TextField("Ask about the scene", text: $vm.question)
                .textFieldStyle(.roundedBorder)
        }
        .padding(.horizontal)
    }

    private var controls: some View {
        HStack {
            Button {
                Task {
                    if vm.isCameraRunning { await vm.stop() } else { await vm.start() }
                }
            } label: {
                Label(
                    vm.isCameraRunning ? "Stop" : "Start live",
                    systemImage: vm.isCameraRunning ? "stop.fill" : "camera.fill"
                )
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent)

            Button(action: vm.reset) {
                Label("Reset", systemImage: "arrow.clockwise")
                    .padding(.vertical, 8)
            }
            .buttonStyle(.bordered)
            .disabled(vm.history.isEmpty && vm.acceptedFrames == 0)
        }
        .padding(.horizontal)
    }

    private var statusAndAnswer: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                if vm.isBusy {
                    ProgressView().scaleEffect(0.8)
                }
                Text(vm.status)
                    .font(.footnote)
                    .foregroundColor(.secondary)
            }

            Text(vm.latestAnswer)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(Color.secondary.opacity(0.1))
                .cornerRadius(8)

            if let err = vm.errorMessage {
                Text(err)
                    .font(.footnote)
                    .foregroundColor(.red)
            }

            Text("accepted \(vm.acceptedFrames) · dropped busy \(vm.droppedBusyFrames) · dropped gate \(vm.droppedGateFrames)")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal)
    }

    @ViewBuilder
    private var historyList: some View {
        if !vm.history.isEmpty {
            Divider().padding(.horizontal)
            VStack(alignment: .leading, spacing: 8) {
                Text("History").font(.headline)
                ForEach(vm.history) { item in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(item.question)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(item.answer)
                            .font(.subheadline)
                            .foregroundColor(item.isError ? .red : .primary)
                        if let ms = item.latencyMs {
                            Text("\(ms) ms")
                                .font(.caption2)
                                .foregroundColor(.blue)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background(Color.secondary.opacity(0.08))
                    .cornerRadius(6)
                }
            }
            .padding(.horizontal)
        }
    }
}
