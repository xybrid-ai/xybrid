package ai.xybrid.example.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun AboutCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                text = "About This Example",
                style = MaterialTheme.typography.titleSmall
            )
            Text(
                text = "This app demonstrates the Xybrid SDK for Android. " +
                       "Select a model from the registry, download and load it, then run inference.",
                style = MaterialTheme.typography.bodySmall
            )
            Text(
                text = "Models are fetched from the Xybrid registry. An internet connection is required for the initial download.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.primary
            )
        }
    }
}
