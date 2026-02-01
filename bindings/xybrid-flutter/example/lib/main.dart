import 'package:flutter/material.dart';
import 'package:flutter_piper/flutter_piper.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

import 'logger.dart';
import 'widgets/device_capabilities_tooltip.dart';
import 'widgets/execution_section.dart';
import 'widgets/input_section.dart';
import 'widgets/selection_section.dart';
import 'widgets/status_section.dart';
import 'widgets/voice_selector.dart';
import 'xybrid_view_model.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Xybrid.init();

  // Local development configuration
  // const localBackendUrl = 'http://127.0.0.1:8000';
  final logStream = XybridLogger.init(level: LogLevel.trace);

  logStream.listen(
    (entry) => logger.d(entry.formatWithTimestamp()),
    onError: (error, stackTrace) => logger.e(error),
  );

  XybridLogger.test();

  // Configure API key for LLM gateway and telemetry
  // Xybrid.setApiKey('sk_test_LkQZgc5DhTZHBLHaoll0IAe94ZorWp5K');
  // Xybrid.setApiKey('sk_test_XBUKnwXxn2IR4GaTjdf1hebIzSY4Bnbz');
  Xybrid.setApiKey('sk_test_Mggzz3yb0BzHB2X1jrryD01McVdP61uR');

  // Set gateway URL for LLM calls (affects pipelines with cloud integration stages)
  // Xybrid.setGatewayUrl(localBackendUrl);

  // Initialize telemetry to send events to local backend
  Xybrid.initTelemetry();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ViewModelScope(
      create: [() => XybridViewModel(audioRecorder: XybridRecorder())],
      child: MaterialApp(
        title: 'Xybrid',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.indigo,
            brightness: Brightness.light,
          ),
          useMaterial3: true,
        ),
        home: const XybridDemo(),
      ),
    );
  }
}

class XybridDemo extends StatefulWidget {
  const XybridDemo({super.key});

  @override
  State<XybridDemo> createState() => _XybridDemoState();
}

class _XybridDemoState extends State<XybridDemo> {
  final TextEditingController _textController = TextEditingController();

  @override
  void initState() {
    super.initState();
    // Load available items when the widget is initialized
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.vm<XybridViewModel>().loadAvailableItems();
    });
  }

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final vm = context.vm<XybridViewModel>();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Xybrid'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [DeviceCapabilitiesTooltip()],
      ),
      body: vm.availableItems.build(
        (state) => switch (state) {
          AsyncEmpty() => const Center(child: Text('No items loaded')),
          AsyncLoading() => const Center(child: CircularProgressIndicator()),
          AsyncError(:final message) => Center(child: Text('Error: $message')),
          AsyncData(:final data) => SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                SelectionSection(vm: vm, items: data),
                const SizedBox(height: 16),
                vm.selectedItem.build((selected) {
                  if (selected == null) return const SizedBox.shrink();
                  return StatusSection(vm: vm);
                }),
                const SizedBox(height: 16),
                // Voice selector for TTS models (only shows when voices available)
                VoiceSelector(vm: vm),
                const SizedBox(height: 16),
                vm.loadingState.build((loadState) {
                  if (loadState != LoadingState.loaded) {
                    return const SizedBox.shrink();
                  }
                  return InputSection(vm: vm, textController: _textController);
                }),
                const SizedBox(height: 16),
                vm.loadingState.build((loadState) {
                  if (loadState != LoadingState.loaded) {
                    return const SizedBox.shrink();
                  }
                  return ExecutionSection(vm: vm);
                }),
              ],
            ),
          ),
        },
      ),
    );
  }
}
