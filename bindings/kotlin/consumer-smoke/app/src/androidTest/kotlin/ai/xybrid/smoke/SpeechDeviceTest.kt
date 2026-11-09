package ai.xybrid.smoke

import ai.xybrid.Envelope
import ai.xybrid.Xybrid
import ai.xybrid.XybridCancellationToken
import ai.xybrid.XybridModel
import ai.xybrid.audioBytes
import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.RandomAccessFile
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Opt-in on-device speech check for the arm64-v8a AAR (Phonon and Kitten TTS 2).
 *
 * Skipped unless the runner gets `-e xybridSpeech true`, so the emulator gate
 * never runs it. Phonon downloads `phonon-31m-en` from the registry (network
 * needed). Kitten loads a local bundle (its `model_metadata.json` plus the four
 * files) from `-e kittenDir <dir>`, default `<external files>/kitten`, and is
 * skipped when that directory has no metadata. Each test writes a WAV into the
 * app's external files directory for listening.
 */
@RunWith(AndroidJUnit4::class)
class SpeechDeviceTest {
    private val arguments = InstrumentationRegistry.getArguments()
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Before
    fun optIn() {
        assumeTrue("pass -e xybridSpeech true", arguments.getString("xybridSpeech") == "true")
        // Speech ships only in the arm64-v8a slice.
        assumeTrue("arm64-v8a only", Build.SUPPORTED_ABIS.firstOrNull() == "arm64-v8a")
        Xybrid.init(context)
    }

    @Test
    fun phononFromRegistry() {
        val model = XybridModel.fromRegistry("phonon-31m-en")
        speak(model, "Hello from Xybrid. Phonon is speaking on an Android phone.", "phonon.wav")
    }

    @Test
    fun kittenFromLocalBundle() {
        val dir = arguments.getString("kittenDir")
            ?: File(context.getExternalFilesDir(null), "kitten").path
        assumeTrue("no Kitten bundle at $dir", File(dir, "model_metadata.json").isFile)
        val model = XybridModel.fromDirectory(dir)
        speak(model, "Hello from Xybrid. Kitten is speaking on an Android phone.", "kitten.wav")
    }

    private fun speak(model: XybridModel, text: String, wavName: String) {
        val result = XybridCancellationToken().use { cancel ->
            model.run(Envelope.text(text), null, cancel)
        }
        val pcm = result.audioBytes
        assertNotNull("no audio output", pcm)
        val meta = result.envelope.metadata.associate { it.key to it.value }
        assertEquals("completed", meta["tts_status"])
        val sampleRate = meta.getValue("sample_rate").toInt()
        val channels = meta.getValue("channels").toInt()
        val samples = meta.getValue("samples").toLong()
        assertTrue("too little audio: $samples samples", samples > sampleRate / 2)
        val bytesPerSample = (pcm!!.size / (samples * channels)).toInt()
        assertTrue("unexpected sample width $bytesPerSample", bytesPerSample == 2 || bytesPerSample == 4)
        val wav = File(context.getExternalFilesDir(null), wavName)
        writeWav(wav, pcm, sampleRate, channels, bytesPerSample)
        println("xybrid-speech: ${result.modelId} -> ${wav.path} ($samples samples @ $sampleRate Hz, ${result.latencyMs} ms)")
        model.unload()
    }

    private fun writeWav(file: File, pcm: ByteArray, rate: Int, channels: Int, width: Int) {
        val format = if (width == 4) 3 else 1 // IEEE float or integer PCM
        RandomAccessFile(file, "rw").use { out ->
            out.setLength(0)
            fun le32(v: Int) = out.write(byteArrayOf(v.toByte(), (v shr 8).toByte(), (v shr 16).toByte(), (v shr 24).toByte()))
            fun le16(v: Int) = out.write(byteArrayOf(v.toByte(), (v shr 8).toByte()))
            out.writeBytes("RIFF"); le32(36 + pcm.size); out.writeBytes("WAVE")
            out.writeBytes("fmt "); le32(16); le16(format); le16(channels); le32(rate)
            le32(rate * channels * width); le16(channels * width); le16(width * 8)
            out.writeBytes("data"); le32(pcm.size); out.write(pcm)
        }
    }
}
