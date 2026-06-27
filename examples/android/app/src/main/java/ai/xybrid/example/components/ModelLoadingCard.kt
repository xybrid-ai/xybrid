package ai.xybrid.example.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import ai.xybrid.XybridModel
import ai.xybrid.XybridVoiceInfo
import ai.xybrid.example.data.CatalogModel
import ai.xybrid.example.data.ModelTask
import ai.xybrid.example.data.MODEL_CATALOG
import ai.xybrid.example.state.ModelState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelLoadingCard(
    modelState: ModelState,
    selectedModel: CatalogModel?,
    onModelSelected: (CatalogModel) -> Unit,
    onLoadModel: () -> Unit,
    onUnloadModel: () -> Unit,
    onRetry: () -> Unit
) {
    var dropdownExpanded by remember { mutableStateOf(false) }
    val isLocked = modelState is ModelState.Loading || modelState is ModelState.Loaded

    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Step 1: Select & Load Model",
                style = MaterialTheme.typography.titleMedium
            )

            ExposedDropdownMenuBox(
                expanded = dropdownExpanded,
                onExpandedChange = { if (!isLocked) dropdownExpanded = it }
            ) {
                OutlinedTextField(
                    value = selectedModel?.displayName ?: "",
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Choose a model") },
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = dropdownExpanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor(),
                    enabled = !isLocked
                )
                ExposedDropdownMenu(
                    expanded = dropdownExpanded,
                    onDismissRequest = { dropdownExpanded = false }
                ) {
                    MODEL_CATALOG.forEach { model ->
                        DropdownMenuItem(
                            text = {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Column(modifier = Modifier.weight(1f)) {
                                        Text(model.displayName)
                                        Text(
                                            text = model.description,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                    Spacer(modifier = Modifier.width(8.dp))
                                    TaskBadge(model.task)
                                }
                            },
                            onClick = {
                                onModelSelected(model)
                                dropdownExpanded = false
                            }
                        )
                    }
                }
            }

            // Model info surface (before loading)
            if (selectedModel != null && modelState !is ModelState.Loaded) {
                ModelInfoSurface(selectedModel)
            }

            when (modelState) {
                is ModelState.NotLoaded -> {
                    Button(
                        onClick = onLoadModel,
                        enabled = selectedModel != null,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Download & Load")
                    }
                }
                is ModelState.Loading -> {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                        Text(
                            text = "Downloading & loading ${selectedModel?.displayName ?: "model"}...",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Text(
                            text = "This may take a moment on first run",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
                is ModelState.Loaded -> {
                    LoadedModelInfo(
                        model = modelState.model,
                        catalogModel = selectedModel
                    )
                    OutlinedButton(
                        onClick = onUnloadModel,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Unload Model")
                    }
                }
                is ModelState.Error -> {
                    Text(
                        text = "Error: ${modelState.message}",
                        color = MaterialTheme.colorScheme.error
                    )
                    Button(
                        onClick = onRetry,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Retry")
                    }
                }
            }
        }
    }
}

@Composable
private fun ModelInfoSurface(model: CatalogModel) {
    Surface(
        color = MaterialTheme.colorScheme.secondaryContainer,
        shape = MaterialTheme.shapes.small,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                TaskBadge(model.task)
                model.parameterCount?.let {
                    Text(
                        text = it,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                }
            }
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = model.description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSecondaryContainer
            )
        }
    }
}

@Composable
private fun LoadedModelInfo(model: XybridModel, catalogModel: CatalogModel?) {
    Surface(
        color = MaterialTheme.colorScheme.primaryContainer,
        shape = MaterialTheme.shapes.small,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = catalogModel?.displayName ?: "Model",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
                catalogModel?.let { TaskBadge(it.task) }
            }

            catalogModel?.parameterCount?.let {
                Text(
                    text = "Parameters: $it",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
            }

            // Show voice info for TTS models
            if (model.hasVoices()) {
                // Bolt's `voices()` returns non-optional List<XybridVoiceInfo>
                // (empty for non-TTS models) and `defaultVoice()` returns the
                // full VoiceInfo (we extract `.id` for the "default: …" label).
                val voices: List<XybridVoiceInfo> = model.voices()
                val defaultVoice = model.defaultVoice()?.id
                if (voices.isNotEmpty()) {
                    Text(
                        text = "${voices.size} voice${if (voices.size > 1) "s" else ""} available" +
                                (defaultVoice?.let { " (default: $it)" } ?: ""),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                }
            }

            Text(
                text = "Ready for inference",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.primary
            )
        }
    }
}

@Composable
fun TaskBadge(task: ModelTask) {
    val color = when (task) {
        ModelTask.TTS -> MaterialTheme.colorScheme.tertiary
        ModelTask.ASR -> MaterialTheme.colorScheme.secondary
        ModelTask.LLM -> MaterialTheme.colorScheme.primary
    }
    Surface(
        color = color.copy(alpha = 0.15f),
        shape = MaterialTheme.shapes.extraSmall
    ) {
        Text(
            text = task.label,
            style = MaterialTheme.typography.labelSmall,
            color = color,
            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
        )
    }
}
