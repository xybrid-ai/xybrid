import 'package:flutter/material.dart';
import 'package:flutter_piper/flutter_piper.dart';
import 'package:xybrid_flutter_example/xybrid_view_model.dart';

class SelectionSection extends StatelessWidget {
  final XybridViewModel vm;
  final List<SelectionItem> items;

  const SelectionSection({super.key, required this.vm, required this.items});

  DropdownMenuItem<SelectionItem> _buildDropdownItem(SelectionItem item) {
    return DropdownMenuItem(
      value: item,
      child: Row(
        children: [
          Icon(
            item.type == SelectionType.pipeline
                ? Icons.account_tree
                : Icons.smart_toy,
            size: 18,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(item.displayName),
                /*if (item.description != null)
                  Text(
                    item.description!,
                    style: TextStyle(fontSize: 11, color: Colors.grey[600]),
                  ),*/
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Select Pipeline or Model',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            vm.selectedItem.build((selected) {
              return DropdownButtonFormField<SelectionItem>(
                initialValue: selected,
                decoration: InputDecoration(
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 8,
                  ),
                ),
                hint: const Text('Choose a pipeline or model...'),
                isExpanded: true,
                items: [
                  // Pipelines section
                  if (items.any((i) => i.type == SelectionType.pipeline)) ...[
                    DropdownMenuItem<SelectionItem>(
                      enabled: false,
                      child: Text(
                        'PIPELINES',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey[600],
                        ),
                      ),
                    ),
                    ...items
                        .where((i) => i.type == SelectionType.pipeline)
                        .map((item) => _buildDropdownItem(item)),
                  ],
                  // Models section
                  if (items.any((i) => i.type == SelectionType.model)) ...[
                    DropdownMenuItem<SelectionItem>(
                      enabled: false,
                      child: Text(
                        'MODELS',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey[600],
                        ),
                      ),
                    ),
                    ...items
                        .where((i) => i.type == SelectionType.model)
                        .map((item) => _buildDropdownItem(item)),
                  ],
                ],
                onChanged: (item) => vm.onSelectionChanged(item),
              );
            }),
          ],
        ),
      ),
    );
  }
}
