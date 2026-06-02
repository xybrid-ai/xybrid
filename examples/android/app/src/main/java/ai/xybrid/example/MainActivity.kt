package ai.xybrid.example

import ai.xybrid.Xybrid
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
        // The key and platform URL come from BuildConfig, wired in
        // app/build.gradle.kts from local.properties (xybrid.apiKey /
        // xybrid.platformUrl), -PXYBRID_API_KEY / -PXYBRID_PLATFORM_URL, or the
        // matching env vars — never committed. Blank resolves to anonymous,
        // local-only init against the default platform. See README.
        Xybrid.init(
            this,
            apiKey = BuildConfig.XYBRID_API_KEY.ifBlank { null },
            ingestUrl = BuildConfig.XYBRID_PLATFORM_URL.ifBlank { null },
        )
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
