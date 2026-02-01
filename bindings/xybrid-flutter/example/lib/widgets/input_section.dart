import 'package:flutter/material.dart';
import 'package:flutter_piper/flutter_piper.dart';
import 'package:xybrid_flutter_example/xybrid_view_model.dart';

import 'audio_input.dart';
import 'text_input.dart';

class InputSection extends StatelessWidget {
  final XybridViewModel vm;
  final TextEditingController textController;

  const InputSection({
    super.key,
    required this.vm,
    required this.textController,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            vm.inputType.build(
              (type) => Text(
                type == XybridInputType.audio ? 'Audio Input' : 'Text Input',
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const SizedBox(height: 12),
            vm.inputType.build((type) {
              if (type == XybridInputType.audio) {
                return AudioInput(vm: vm);
              } else {
                return vm.outputType.build((outputType) {
                  final hintText = outputType == XybridOutputType.text
                      ? 'Enter your prompt...'
                      : 'Enter text to synthesize...';
                  return TextInputWidget(
                    controller: textController,
                    onChanged: (text) => vm.textInput.value = text,
                    hintText: hintText,
                  );
                });
              }
            }),
          ],
        ),
      ),
    );
  }
}
