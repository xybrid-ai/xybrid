package ai.xybrid.example.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ai.xybrid.example.data.CatalogModel
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
                                Column {
                                    Text(model.displayName)
                                    Text(
                                        text = model.task,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
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

            if (selectedModel != null && modelState !is ModelState.Loaded) {
                Surface(
                    color = MaterialTheme.colorScheme.secondaryContainer,
                    shape = MaterialTheme.shapes.small,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(
                            text = selectedModel.task,
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                        Text(
                            text = selectedModel.description,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                    }
                }
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
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        CircularProgressIndicator(modifier = Modifier.size(24.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Downloading & loading model...")
                    }
                }
                is ModelState.Loaded -> {
                    Text(
                       // TODO text = "Model loaded: ${modelState.model.id}",
                        text = "Model loaded",
                        color = MaterialTheme.colorScheme.primary
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
