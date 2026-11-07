package ai.xybrid.example

import ai.xybrid.Xybrid
import ai.xybrid.XybridError
import ai.xybrid.displayMessage
import ai.xybrid.service
import ai.xybrid.start
import ai.xybrid.tracesEndpoint
import android.os.Bundle
import android.util.Log
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
        // The key and trace endpoint come from BuildConfig, wired in
        // app/build.gradle.kts from local.properties (xybrid.apiKey /
        // xybrid.tracesEndpoint), -PXYBRID_API_KEY / -PXYBRID_TRACES_ENDPOINT,
        // or the matching env vars — never committed. Blank resolves to
        // anonymous, local-only init. See README.
        val apiKey = BuildConfig.XYBRID_API_KEY.ifBlank { null }
        Xybrid.init(this, apiKey = apiKey)

        // The API key alone exports nothing: opt in to dashboard traces.
        if (apiKey != null) {
            val observability = Xybrid.Observability.xybrid(apiKey)
                .service("XybridExample", version = BuildConfig.VERSION_NAME)
            BuildConfig.XYBRID_TRACES_ENDPOINT.ifBlank { null }?.let {
                observability.tracesEndpoint(it)
            }
            try {
                observability.start()
            } catch (e: XybridError) {
                Log.w("XybridExample", "Observability disabled: ${e.displayMessage}")
            }
        }
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
