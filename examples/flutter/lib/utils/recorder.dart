// Xybrid Flutter SDK - Audio Recorder with Voice Activity Detection
//
// Provides integrated microphone recording with:
// - Silence detection and auto-stop
// - Voice Activity Detection (VAD)
// - Permission handling helpers
// - Direct integration with XybridPipeline

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:record/record.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:path_provider/path_provider.dart';

/// Recording configuration for XybridRecorder.
class RecordingConfig {
  /// Sample rate in Hz (default: 16000 for ASR)
  final int sampleRate;

  /// Number of channels (default: 1 for mono)
  final int channels;

  /// Bit rate for encoding (default: 128000)
  final int bitRate;

  /// Maximum recording duration in milliseconds (default: 60000 = 1 minute)
  final int maxDurationMs;

  /// Enable auto-stop on silence detection
  final bool autoStopOnSilence;

  /// Silence threshold in dB (default: -40 dB)
  final double silenceThresholdDb;

  /// Duration of silence before auto-stop in milliseconds (default: 2000 = 2 seconds)
  final int silenceDurationMs;

  /// Enable Voice Activity Detection
  final bool enableVad;

  /// VAD sensitivity (0.0 = less sensitive, 1.0 = more sensitive)
  final double vadSensitivity;

  const RecordingConfig({
    this.sampleRate = 16000,
    this.channels = 1,
    this.bitRate = 128000,
    this.maxDurationMs = 60000,
    this.autoStopOnSilence = false,
    this.silenceThresholdDb = -40.0,
    this.silenceDurationMs = 2000,
    this.enableVad = false,
    this.vadSensitivity = 0.5,
  });

  /// Default config for ASR (speech recognition)
  static const asr = RecordingConfig(
    sampleRate: 16000,
    channels: 1,
    autoStopOnSilence: true,
    silenceDurationMs: 1500,
  );

  /// Config with VAD enabled for automatic segmentation
  static const withVad = RecordingConfig(
    sampleRate: 16000,
    channels: 1,
    enableVad: true,
    autoStopOnSilence: true,
    silenceDurationMs: 1000,
  );
}

/// Recording state
enum RecordingState {
  /// Not recording
  idle,

  /// Recording in progress
  recording,

  /// Voice detected (when VAD is enabled)
  voiceDetected,

  /// Silence detected (may auto-stop soon)
  silenceDetected,

  /// Recording stopped
  stopped,
}

/// Result of a recording session
class RecordingResult {
  /// Path to the recorded audio file
  final String path;

  /// Duration of recording in milliseconds
  final int durationMs;

  /// File size in bytes
  final int sizeBytes;

  /// Whether recording was auto-stopped due to silence
  final bool autoStopped;

  /// Voice segments detected (when VAD is enabled)
  final List<VoiceSegment> voiceSegments;

  const RecordingResult({
    required this.path,
    required this.durationMs,
    required this.sizeBytes,
    this.autoStopped = false,
    this.voiceSegments = const [],
  });
}

/// A detected voice segment (when VAD is enabled)
class VoiceSegment {
  /// Start time in milliseconds from recording start
  final int startMs;

  /// End time in milliseconds from recording start
  final int endMs;

  const VoiceSegment({required this.startMs, required this.endMs});

  /// Duration of this segment in milliseconds
  int get durationMs => endMs - startMs;
}

/// Permission status for audio recording
enum PermissionStatus {
  /// Permission granted
  granted,

  /// Permission denied
  denied,

  /// Permission not yet requested
  undetermined,

  /// Permission permanently denied (user must enable in settings)
  permanentlyDenied,
}

/// Xybrid Audio Recorder with VAD and silence detection.
///
/// Provides high-level audio recording with:
/// - Automatic silence detection and auto-stop
/// - Voice Activity Detection (VAD) for segmentation
/// - Permission handling helpers
/// - Direct integration with XybridPipeline
///
/// Example:
/// ```dart
/// final recorder = XybridRecorder();
///
/// // Check and request permissions
/// if (await recorder.requestPermission() == PermissionStatus.granted) {
///   // Start recording with auto-stop on silence
///   await recorder.start(config: RecordingConfig.asr);
///
///   // Wait for recording to complete (auto-stops on silence)
///   final result = await recorder.waitForCompletion();
///
///   // Read audio bytes for inference
///   final bytes = await recorder.readAudioBytes(result.path);
/// }
///
/// recorder.dispose();
/// ```
class XybridRecorder {
  final AudioRecorder _recorder = AudioRecorder();
  AudioPlayer? _player;

  RecordingState _state = RecordingState.idle;
  String? _currentPath;
  String? _lastRecordingPath;
  RecordingConfig _config = const RecordingConfig();

  // Amplitude monitoring for silence detection
  StreamSubscription<Amplitude>? _amplitudeSubscription;
  Timer? _silenceTimer;
  Timer? _maxDurationTimer;
  DateTime? _recordingStartTime;
  DateTime? _silenceStartTime;

  // VAD state
  final List<VoiceSegment> _voiceSegments = [];
  DateTime? _voiceStartTime;
  bool _voiceActive = false;

  // Completion handling
  Completer<RecordingResult>? _completionCompleter;
  bool _autoStopped = false;

  // Playback
  bool _isPlaying = false;
  StreamSubscription? _playerCompleteSubscription;

  // Streaming mode
  bool _isStreaming = false;
  StreamSubscription<Uint8List>? _streamSubscription;
  void Function(List<double> samples)? _onStreamSamples;

  // Callbacks
  void Function(RecordingState state)? onStateChanged;
  void Function(double amplitudeDb)? onAmplitude;
  void Function()? onVoiceStart;
  void Function()? onVoiceEnd;
  void Function()? onPlaybackComplete;

  /// Current recording state
  RecordingState get state => _state;

  /// Whether currently recording
  bool get isRecording => _state == RecordingState.recording ||
      _state == RecordingState.voiceDetected ||
      _state == RecordingState.silenceDetected;

  /// Whether audio is playing
  bool get isPlaying => _isPlaying;

  /// Whether streaming mode is active
  bool get isStreaming => _isStreaming;

  /// Path of the last recording
  String? get lastRecordingPath => _lastRecordingPath;

  /// Check if recording permission is granted
  Future<bool> hasPermission() async {
    return await _recorder.hasPermission();
  }

  /// Request recording permission
  ///
  /// Returns the permission status after requesting.
  Future<PermissionStatus> requestPermission() async {
    final hasIt = await _recorder.hasPermission();
    if (hasIt) {
      return PermissionStatus.granted;
    }

    // The record package auto-requests when we check
    // If still no permission, it was denied
    final stillHas = await _recorder.hasPermission();
    return stillHas ? PermissionStatus.granted : PermissionStatus.denied;
  }

  /// Start recording audio
  ///
  /// [config] - Recording configuration (default: RecordingConfig())
  ///
  /// Throws if permission is not granted or already recording.
  Future<void> start({RecordingConfig config = const RecordingConfig()}) async {
    if (isRecording) {
      throw StateError('Already recording');
    }

    if (!await hasPermission()) {
      throw StateError('Microphone permission not granted');
    }

    _config = config;
    _autoStopped = false;
    _voiceSegments.clear();
    _voiceActive = false;
    _voiceStartTime = null;

    try {
      // Get app documents directory for proper sandboxed path
      final directory = await getApplicationDocumentsDirectory();
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      _currentPath = '${directory.path}/xybrid_recording_$timestamp.wav';

      // Configure recording settings
      final recordConfig = RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: config.sampleRate,
        numChannels: config.channels,
        bitRate: config.bitRate,
      );

      await _recorder.start(recordConfig, path: _currentPath!);
      _recordingStartTime = DateTime.now();
      _updateState(RecordingState.recording);

      // Start amplitude monitoring for silence detection / VAD
      if (config.autoStopOnSilence || config.enableVad) {
        _startAmplitudeMonitoring();
      }

      // Start max duration timer
      if (config.maxDurationMs > 0) {
        _maxDurationTimer = Timer(
          Duration(milliseconds: config.maxDurationMs),
              () => _handleMaxDuration(),
        );
      }

      // Create completion completer
      _completionCompleter = Completer<RecordingResult>();
    } catch (e) {
      _currentPath = null;
      _updateState(RecordingState.idle);
      throw StateError('Failed to start recording: $e');
    }
  }

  /// Stop recording and return the result
  Future<RecordingResult> stop() async {
    if (!isRecording) {
      throw StateError('Not recording');
    }

    return await _stopInternal();
  }

  /// Cancel recording (discard audio)
  Future<void> cancel() async {
    if (!isRecording) return;

    _stopMonitoring();
    await _recorder.stop();
    _updateState(RecordingState.idle);
    _currentPath = null;
    _completionCompleter?.completeError(StateError('Recording cancelled'));
    _completionCompleter = null;
  }

  /// Wait for recording to complete (via auto-stop or manual stop)
  ///
  /// Returns when the recording is stopped, either manually or via auto-stop.
  Future<RecordingResult> waitForCompletion() async {
    if (_completionCompleter == null) {
      throw StateError('No recording in progress');
    }
    return _completionCompleter!.future;
  }

  /// Read audio file as bytes
  Future<Uint8List> readAudioBytes(String path) async {
    final file = File(path);
    if (!await file.exists()) {
      throw StateError('Audio file not found: $path');
    }
    return await file.readAsBytes();
  }

  /// Read the last recording as bytes
  Future<Uint8List> readLastRecordingBytes() async {
    if (_lastRecordingPath == null) {
      throw StateError('No recording available');
    }
    return await readAudioBytes(_lastRecordingPath!);
  }

  /// Play the last recorded audio
  Future<void> playLastRecording() async {
    if (_lastRecordingPath == null) {
      throw StateError('No recording to play');
    }
    await playAudioFile(_lastRecordingPath!);
  }

  /// Play an audio file by path
  Future<void> playAudioFile(String path) async {
    final file = File(path);
    if (!await file.exists()) {
      throw StateError('Audio file not found: $path');
    }

    if (_isPlaying) {
      await stopPlayback();
    }

    try {
      await _playerCompleteSubscription?.cancel();
      await _player?.dispose();
      _player = AudioPlayer();

      _playerCompleteSubscription = _player!.onPlayerComplete.listen((_) {
        _isPlaying = false;
        onPlaybackComplete?.call();
      });

      _player!.onPlayerStateChanged.listen((state) {
        if (state == PlayerState.completed || state == PlayerState.stopped) {
          _isPlaying = false;
        }
      });

      _isPlaying = true;
      await _player!.play(DeviceFileSource(path));
    } catch (e) {
      _isPlaying = false;
      throw StateError('Failed to play audio: $e');
    }
  }

  /// Stop audio playback
  Future<void> stopPlayback() async {
    if (_isPlaying) {
      await _player?.stop();
      _isPlaying = false;
    }
  }

  /// Get temp directory for audio files
  Future<String> getTempDirectory() async {
    final directory = await getTemporaryDirectory();
    return directory.path;
  }

  /// Write audio bytes to a file
  Future<void> writeAudioFile(String path, Uint8List bytes) async {
    final file = File(path);
    await file.writeAsBytes(bytes);
  }

  /// Start streaming audio samples in real-time.
  ///
  /// This records audio and provides raw PCM samples to the callback
  /// in real-time, suitable for streaming ASR.
  ///
  /// [onSamples] - Callback that receives float32 audio samples (16kHz mono).
  ///               Samples are normalized to range -1.0 to 1.0.
  ///
  /// Note: This does NOT save to a file. Use [start] for file-based recording.
  Future<void> startStreaming({
    required void Function(List<double> samples) onSamples,
    int sampleRate = 16000,
  }) async {
    if (_isStreaming) {
      throw StateError('Already streaming');
    }

    if (isRecording) {
      throw StateError('Already recording');
    }

    if (!await hasPermission()) {
      throw StateError('Microphone permission not granted');
    }

    _onStreamSamples = onSamples;
    _isStreaming = true;
    _recordingStartTime = DateTime.now();
    _updateState(RecordingState.recording);

    try {
      // Use PCM stream for raw audio data
      final recordConfig = RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: sampleRate,
        numChannels: 1,
      );

      final stream = await _recorder.startStream(recordConfig);

      _streamSubscription = stream.listen(
            (Uint8List pcmBytes) {
          // Convert PCM 16-bit to float32 samples
          final samples = _pcm16ToFloat32(pcmBytes);
          _onStreamSamples?.call(samples);
        },
        onError: (error) {
          _isStreaming = false;
          _updateState(RecordingState.idle);
        },
        onDone: () {
          _isStreaming = false;
          _updateState(RecordingState.stopped);
        },
      );
    } catch (e) {
      _isStreaming = false;
      _onStreamSamples = null;
      _updateState(RecordingState.idle);
      throw StateError('Failed to start streaming: $e');
    }
  }

  /// Stop streaming and return duration info.
  Future<int> stopStreaming() async {
    if (!_isStreaming) {
      throw StateError('Not streaming');
    }

    await _streamSubscription?.cancel();
    _streamSubscription = null;
    await _recorder.stop();

    _isStreaming = false;
    _onStreamSamples = null;
    _updateState(RecordingState.stopped);

    final durationMs = _recordingStartTime != null
        ? DateTime.now().difference(_recordingStartTime!).inMilliseconds
        : 0;

    return durationMs;
  }

  /// Convert PCM 16-bit little-endian bytes to float32 samples.
  List<double> _pcm16ToFloat32(Uint8List pcmBytes) {
    final samples = <double>[];
    for (int i = 0; i < pcmBytes.length - 1; i += 2) {
      final sample = pcmBytes[i] | (pcmBytes[i + 1] << 8);
      // Convert from unsigned to signed
      final signed = sample > 32767 ? sample - 65536 : sample;
      // Normalize to -1.0 to 1.0
      samples.add(signed / 32768.0);
    }
    return samples;
  }

  /// Dispose all resources
  Future<void> dispose() async {
    _stopMonitoring();
    if (_isStreaming) {
      await _streamSubscription?.cancel();
      _streamSubscription = null;
      _isStreaming = false;
    }
    if (isRecording) {
      await _recorder.stop();
    }
    await _playerCompleteSubscription?.cancel();
    await _recorder.dispose();
    await _player?.dispose();
  }

  // Private methods

  void _updateState(RecordingState newState) {
    _state = newState;
    onStateChanged?.call(newState);
  }

  void _startAmplitudeMonitoring() {
    // Monitor amplitude every 100ms
    _amplitudeSubscription = _recorder
        .onAmplitudeChanged(const Duration(milliseconds: 100))
        .listen(_handleAmplitude);
  }

  void _handleAmplitude(Amplitude amplitude) {
    final db = amplitude.current;
    onAmplitude?.call(db);

    // Check for voice activity
    if (_config.enableVad) {
      _handleVad(db);
    }

    // Check for silence
    if (_config.autoStopOnSilence) {
      _handleSilenceDetection(db);
    }
  }

  void _handleVad(double db) {
    // Simple VAD: voice is active when amplitude is above threshold
    // Threshold adjusted by sensitivity (higher sensitivity = lower threshold)
    final threshold =
        _config.silenceThresholdDb + (1.0 - _config.vadSensitivity) * 20;
    final isVoice = db > threshold;

    if (isVoice && !_voiceActive) {
      // Voice started
      _voiceActive = true;
      _voiceStartTime = DateTime.now();
      _updateState(RecordingState.voiceDetected);
      onVoiceStart?.call();
    } else if (!isVoice && _voiceActive) {
      // Voice ended
      _voiceActive = false;
      if (_voiceStartTime != null) {
        final startMs =
            _voiceStartTime!.difference(_recordingStartTime!).inMilliseconds;
        final endMs =
            DateTime.now().difference(_recordingStartTime!).inMilliseconds;
        _voiceSegments.add(VoiceSegment(startMs: startMs, endMs: endMs));
      }
      _updateState(RecordingState.recording);
      onVoiceEnd?.call();
    }
  }

  void _handleSilenceDetection(double db) {
    final isSilent = db <= _config.silenceThresholdDb;

    if (isSilent) {
      if (_silenceStartTime == null) {
        _silenceStartTime = DateTime.now();
        _updateState(RecordingState.silenceDetected);
      } else {
        // Check if silence duration exceeded
        final silenceDuration =
            DateTime.now().difference(_silenceStartTime!).inMilliseconds;
        if (silenceDuration >= _config.silenceDurationMs) {
          _handleAutoStop();
        }
      }
    } else {
      // Voice detected, reset silence timer
      _silenceStartTime = null;
      if (_state == RecordingState.silenceDetected) {
        _updateState(
            _voiceActive ? RecordingState.voiceDetected : RecordingState.recording);
      }
    }
  }

  void _handleAutoStop() {
    _autoStopped = true;
    _stopInternal();
  }

  void _handleMaxDuration() {
    if (isRecording) {
      _stopInternal();
    }
  }

  Future<RecordingResult> _stopInternal() async {
    _stopMonitoring();

    final path = await _recorder.stop();
    final recordingPath = path ?? _currentPath!;
    _lastRecordingPath = recordingPath;
    _currentPath = null;
    _updateState(RecordingState.stopped);

    // Get file info
    final file = File(recordingPath);
    final sizeBytes = await file.exists() ? await file.length() : 0;
    final durationMs = _recordingStartTime != null
        ? DateTime.now().difference(_recordingStartTime!).inMilliseconds
        : 0;

    final result = RecordingResult(
      path: recordingPath,
      durationMs: durationMs,
      sizeBytes: sizeBytes,
      autoStopped: _autoStopped,
      voiceSegments: List.from(_voiceSegments),
    );

    _completionCompleter?.complete(result);
    _completionCompleter = null;

    return result;
  }

  void _stopMonitoring() {
    _amplitudeSubscription?.cancel();
    _amplitudeSubscription = null;
    _silenceTimer?.cancel();
    _silenceTimer = null;
    _maxDurationTimer?.cancel();
    _maxDurationTimer = null;
    _silenceStartTime = null;
  }
}
