package ai.xybrid

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test

class ModelLoaderTest {
    @Test
    fun registryShortcutCreatesAnUnloadedRegistryReference() {
        val loader = Xybrid.model("kokoro-82m")

        assertEquals(ModelSource.Registry("kokoro-82m"), loader.source)
    }

    @Test
    fun typedSourceCreatesAnUnloadedBundleReference() {
        val source = ModelSource.Bundle("/models/kokoro.xyb")

        val loader = Xybrid.model(source)

        assertEquals(source, loader.source)
    }

    @Test
    fun sourceFactoriesDescribeEverySupportedModelLocation() {
        assertEquals(ModelSource.Registry("model"), ModelSource.registry("model"))
        assertEquals(ModelSource.Bundle("model.xyb"), ModelSource.bundle("model.xyb"))
        assertEquals(ModelSource.Directory("models/model"), ModelSource.directory("models/model"))
        assertEquals(
            ModelSource.HuggingFace("org/model"),
            ModelSource.huggingFace("org/model"),
        )
        assertEquals(ModelSource.ModelFile("model.gguf"), ModelSource.modelFile("model.gguf"))
    }

    @Test
    fun modelFileLoaderIsUnloadedAndHasNothingToDownload() {
        val loader = XybridModelLoader.fromModelFile("/data/models/qwen/model.gguf")

        assertEquals(ModelSource.ModelFile("/data/models/qwen/model.gguf"), loader.source)
        assertNull(loader.download())
        assertFalse(loader.willSpeculate)
    }
}
