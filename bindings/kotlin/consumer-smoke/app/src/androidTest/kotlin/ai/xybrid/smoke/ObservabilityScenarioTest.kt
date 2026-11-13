package ai.xybrid.smoke

import ai.xybrid.Xybrid
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The shared observability scenario on Android, through the shipped AAR.
 *
 * Skipped unless the runner gets `-e xybridScenarioEndpoint <url>`, the capture
 * receiver as the device reaches it (`http://10.0.2.2:<port>` from an
 * emulator), so the dlopen gates' `connectedDebugAndroidTest` never runs it.
 * `observability-scenario.sh` passes it, then copies
 * `files/observability-scenario/expectations.json` off the device with
 * `run-as` for `check_capture.py`. The file is written even when a step
 * failed; the test then fails too.
 */
@RunWith(AndroidJUnit4::class)
class ObservabilityScenarioTest {
    @Test
    fun exportsTheSharedScenario() {
        val endpoint = InstrumentationRegistry.getArguments().getString(ENDPOINT_ARGUMENT)
        assumeTrue("pass -e $ENDPOINT_ARGUMENT http://10.0.2.2:<port>", !endpoint.isNullOrEmpty())
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        // Registers the binding the exported resource names (xybrid.sdk.binding = kotlin).
        Xybrid.init(context)

        val outDir = File(context.filesDir, "observability-scenario")
        outDir.deleteRecursively()
        check(outDir.mkdirs()) { "could not create $outDir" }
        val outcome = ObservabilityScenario(endpoint!!, File(context.cacheDir, "observability-scenario")).run()

        val expectations = File(outDir, "expectations.json")
        val partial = File(outDir, "expectations.json.tmp")
        partial.writeText(outcome.toJson())
        check(partial.renameTo(expectations)) { "could not write $expectations" }
        Log.i(TAG, "wrote ${expectations.path}; scenario_error=${outcome.expectations["scenario_error"]}")

        outcome.error?.let { throw AssertionError("the observability scenario failed; see $expectations", it) }
    }

    private companion object {
        const val ENDPOINT_ARGUMENT = "xybridScenarioEndpoint"
        const val TAG = "XybridScenario"
    }
}
