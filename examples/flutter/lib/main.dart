import 'package:flutter/material.dart';
import 'package:xybrid_example/ui.dart';
import 'package:xybrid_flutter/xybrid.dart';

import 'screens/asr_demo_screen.dart';
import 'screens/errors_demo_screen.dart';
import 'screens/hub_screen.dart';
import 'screens/llm_demo_screen.dart';
import 'screens/model_loading_screen.dart';
import 'screens/pipeline_demo_screen.dart';
import 'screens/tts_demo_screen.dart';

void main() {
  runApp(const XybridExampleApp());
}

/// The main application widget for the Xybrid SDK example.
class XybridExampleApp extends StatelessWidget {
  const XybridExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Xybrid SDK Example',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const SdkInitializationScreen(),
      routes: {
        '/demos': (context) => const DemoHubScreen(),
        '/demos/model-loading': (context) => const ModelLoadingScreen(),
        '/demos/text-to-speech': (context) => const TextToSpeechScreen(),
        '/demos/speech-to-text': (context) => const SpeechToTextScreen(),
        '/demos/pipelines': (context) => const PipelineScreen(),
        '/demos/llm': (context) => const LlmDemoScreen(),
        '/demos/error-handling': (context) => const ErrorHandlingScreen(),
      },
    );
  }
}

/// Screen that handles SDK initialization and displays status.
class SdkInitializationScreen extends StatefulWidget {
  const SdkInitializationScreen({super.key});

  @override
  State<SdkInitializationScreen> createState() =>
      _SdkInitializationScreenState();
}

class _SdkInitializationScreenState extends State<SdkInitializationScreen> {
  /// Current initialization state.
  _InitState _initState = _InitState.loading;

  /// Error message if initialization failed.
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _initializeSdk();
  }

  /// Initialize the Xybrid SDK.
  Future<void> _initializeSdk() async {
    try {
      await Xybrid.init();
      if (mounted) {
        setState(() {
          _initState = _InitState.ready;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _initState = _InitState.error;
          _errorMessage = e.toString();
        });
      }
    }
  }

  /// Retry SDK initialization after failure.
  void _retryInitialization() {
    setState(() {
      _initState = _InitState.loading;
      _errorMessage = null;
    });
    _initializeSdk();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Xybrid SDK Example'),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: switch (_initState) {
            _InitState.loading => const _LoadingView(),
            _InitState.ready => const _ReadyView(),
            _InitState.error => ErrorView(
              errorMessage: _errorMessage ?? 'Unknown error',
              onRetry: _retryInitialization,
            ),
          },
        ),
      ),
    );
  }
}

/// Enum representing SDK initialization states.
enum _InitState { loading, ready, error }

/// Loading view shown during SDK initialization.
class _LoadingView extends StatelessWidget {
  const _LoadingView();

  @override
  Widget build(BuildContext context) {
    return const Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        CircularProgressIndicator(),
        SizedBox(height: 24),
        Text('Initializing Xybrid SDK...', style: TextStyle(fontSize: 18)),
      ],
    );
  }
}

/// Ready view shown when SDK is initialized successfully.
class _ReadyView extends StatelessWidget {
  const _ReadyView();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.check_circle, size: 72, color: theme.colorScheme.primary),
        const SizedBox(height: 24),
        Text(
          'SDK Ready',
          style: theme.textTheme.headlineMedium?.copyWith(
            color: theme.colorScheme.primary,
          ),
        ),
        const SizedBox(height: 16),
        Text(
          'Xybrid SDK has been initialized successfully.\n'
          'You can now load models and run inference.',
          textAlign: TextAlign.center,
          style: theme.textTheme.bodyLarge,
        ),
        const SizedBox(height: 32),
        FilledButton.icon(
          onPressed: () => Navigator.pushReplacementNamed(context, '/demos'),
          icon: const Icon(Icons.arrow_forward),
          label: const Text('Continue to Demos'),
        ),
      ],
    );
  }
}
