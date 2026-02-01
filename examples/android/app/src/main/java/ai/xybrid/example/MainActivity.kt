package ai.xybrid.example

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import ai.xybrid.example.ui.theme.XybridExampleTheme

/**
 * Main Activity for the Xybrid SDK Example App.
 *
 * This app demonstrates how to integrate the Xybrid SDK for:
 * - SDK initialization
 * - Model loading from registry
 * - Running TTS (Text-to-Speech) inference
 * - Proper error handling patterns
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            XybridExampleTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    XybridExampleApp()
                }
            }
        }
    }
}
