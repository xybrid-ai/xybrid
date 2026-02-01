import 'package:flutter/material.dart';

/// Demo hub screen showing available SDK demos.
class DemoHubScreen extends StatelessWidget {
  const DemoHubScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        backgroundColor: theme.colorScheme.inversePrimary,
        title: const Text('Xybrid SDK Demos'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _DemoTile(
            icon: Icons.download,
            title: 'Model Loading',
            subtitle: 'Load models from the registry',
            onTap: () => Navigator.pushNamed(context, '/demos/model-loading'),
          ),
          _DemoTile(
            icon: Icons.record_voice_over,
            title: 'Text-to-Speech',
            subtitle: 'Convert text to audio',
            onTap: () => Navigator.pushNamed(context, '/demos/text-to-speech'),
          ),
          _DemoTile(
            icon: Icons.mic,
            title: 'Speech-to-Text',
            subtitle: 'Transcribe audio to text',
            onTap: () => Navigator.pushNamed(context, '/demos/speech-to-text'),
          ),
          _DemoTile(
            icon: Icons.account_tree,
            title: 'Pipelines',
            subtitle: 'Multi-stage inference workflows',
            onTap: () => Navigator.pushNamed(context, '/demos/pipelines'),
          ),
          _DemoTile(
            icon: Icons.warning,
            title: 'Error Handling',
            subtitle: 'Error patterns showcase',
            onTap: () => Navigator.pushNamed(context, '/demos/error-handling'),
          ),
        ],
      ),
    );
  }
}

/// A tile representing a demo in the hub.
class _DemoTile extends StatelessWidget {
  const _DemoTile({
    required this.icon,
    required this.title,
    required this.subtitle,
    this.onTap,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      child: ListTile(
        leading: Icon(icon, color: theme.colorScheme.primary),
        title: Text(title),
        subtitle: Text(subtitle),
        trailing: const Icon(Icons.chevron_right),
        onTap: onTap,
      ),
    );
  }
}
