package ai.xybrid.smoke

import ai.xybrid.Xybrid
import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.json.JSONException
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The step 6 staging scenario on a device, through the shipped AAR.
 *
 * Skipped unless the app's private `files/xybrid-staging/config.json` exists,
 * so the dlopen gates' `connectedDebugAndroidTest` never runs it.
 * `observability-scenario.sh staging` streams the config there with `run-as`,
 * then copies `files/xybrid-staging/expectations.json` off the device for
 * `run_staging.py`. The config holds the staging key: it is deleted as soon as
 * it is read, and never logged. The expectations file is written even when a
 * step failed; the test then fails too.
 */
@RunWith(AndroidJUnit4::class)
class ObservabilityStagingScenarioTest {
    @Test(timeout = TIMEOUT_MS)
    fun exportsTheStagingScenario() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(context.filesDir, DIRECTORY)
        val configFile = File(dir, "config.json")
        assumeTrue("no $configFile; observability-scenario.sh staging copies it", configFile.isFile)
        val text = configFile.readText()
        check(configFile.delete()) { "could not delete $configFile" }
        val config = try {
            JSONObject(text)
        } catch (malformed: JSONException) {
            // Not chained: the parser's message quotes the document, key included.
            throw IllegalArgumentException("$configFile is not a JSON object")
        }
        // Registers the binding the exported resource names (xybrid.sdk.binding = kotlin).
        Xybrid.init(context)

        val outcome = ObservabilityStagingScenario(config, device()).run()
        config.remove("sessions") // it holds the staging key; nothing reads it again

        val expectations = File(dir, "expectations.json")
        val partial = File(dir, "expectations.json.tmp")
        partial.writeText(outcome.toJson())
        check(partial.renameTo(expectations)) { "could not write $expectations" }
        Log.i(TAG, "wrote ${expectations.path}; scenario_error=${outcome.expectations["scenario_error"]}")

        outcome.error?.let { throw AssertionError("the observability staging scenario failed; see $expectations", it) }
    }

    private fun device(): Map<String, Any?> = linkedMapOf(
        "system" to "Android",
        "release" to Build.VERSION.RELEASE,
        "api_level" to Build.VERSION.SDK_INT,
        "model" to Build.MODEL,
        "machine" to Build.SUPPORTED_ABIS.firstOrNull(),
    )

    private companion object {
        const val DIRECTORY = "xybrid-staging"
        const val TAG = "XybridStaging"

        /**
         * Generous: the first launch downloads the model (about 150 MB) in the
         * direct session's load. Below run_staging.py's 30-minute lane limit,
         * so a hang fails here and the script still reports it.
         */
        const val TIMEOUT_MS = 20 * 60 * 1000L
    }
}
