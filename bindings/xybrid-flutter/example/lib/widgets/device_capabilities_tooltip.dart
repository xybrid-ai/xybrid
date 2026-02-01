import 'package:flutter/material.dart';
import 'package:xybrid_flutter/xybrid_flutter.dart' show getDeviceCapabilities;

class DeviceCapabilitiesTooltip extends StatelessWidget {
  const DeviceCapabilitiesTooltip({super.key});

  void _showDeviceCapabilities(BuildContext context) {
    /*TODO
    final caps = getDeviceCapabilities();

    showModalBottomSheet(
      context: context,
      builder: (context) => Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Device Capabilities',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            _buildRow('Platform', caps.platform),
            _buildRow('GPU', caps.hasGpu ? caps.gpuType : 'None'),
            _buildRow('Metal', caps.hasMetal ? 'Yes' : 'No'),
            _buildRow('NPU', caps.hasNpu ? caps.npuType : 'None'),
            _buildRow(
                'Memory', '${caps.memoryAvailableMb}/${caps.memoryTotalMb} MB'),
            _buildRow('CPU', '${caps.cpuCores} cores'),
            _buildRow('Throttle', caps.shouldThrottle ? 'Yes' : 'No'),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  Widget _buildRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontWeight: FontWeight.w500)),
          Text(value),
        ],
      ),
    );*/
  }

  @override
  Widget build(BuildContext context) {
    return IconButton(
      icon: const Icon(Icons.memory),
      onPressed: () => _showDeviceCapabilities(context),
      tooltip: 'Device Capabilities',
    );
  }
}
