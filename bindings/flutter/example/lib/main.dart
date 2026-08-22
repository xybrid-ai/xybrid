import 'package:flutter/material.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart';

void main() {
  runApp(const XybridDemo());
}

class XybridDemo extends StatelessWidget {
  const XybridDemo({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "Xybrid Demo App",
      home: const DemoHomePage(),
    );
  }
}

class DemoHomePage extends StatefulWidget {
  const DemoHomePage({super.key});

  @override
  State<DemoHomePage> createState() => DemoHomePageState();
}

class DemoHomePageState extends State<DemoHomePage> {
  String status = "Initializing Xybrid...";

  @override
  void initState() {
    super.initState();
    initAndLoad();
  }

  Future<void> initAndLoad() async {
    try {
      await Xybrid.init();

      if (!mounted) return;
      setState(() => status = "Loading whisper tiny");
      await XybridModelLoader.fromRegistry('whisper-tiny-ggml').load();

      if (!mounted) return;
      setState(() => status = "Whisper loaded");
    } catch (e) {
      if (!mounted) return;
      final message = e is XybridException ? e.message : e.toString();
      setState(() => status = "Error: $message");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Xybrid Demo App")),
      body: Center(child: Text(status)),
    );
  }
}

