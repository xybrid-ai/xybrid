@file:OptIn(kotlin.ExperimentalUnsignedTypes::class)

package ai.xybrid

private object Utf8Codec {
    fun maxBytes(value: String): Int = value.length * 3
}

private val <First, Second> Pair<First, Second>.field0: First get() = first
private val <First, Second> Pair<First, Second>.field1: Second get() = second
private val <First, Second, Third> Triple<First, Second, Third>.field0: First get() = first
private val <First, Second, Third> Triple<First, Second, Third>.field1: Second get() = second
private val <First, Second, Third> Triple<First, Second, Third>.field2: Third get() = third

class FfiException(message: String) : RuntimeException(message)

internal class BoltFfiErrorBufferException(val bytes: ByteArray) : RuntimeException("BoltFFI call failed")

private object DirectVectorCodec {
    fun readBooleanArray(bytes: ByteArray): BooleanArray =
        BooleanArray(bytes.size) { index -> bytes[index] != 0.toByte() }

    fun readByteArray(bytes: ByteArray): ByteArray = bytes

    fun writeBooleanArray(values: BooleanArray): ByteArray =
        ByteArray(values.size) { index -> if (values[index]) 1.toByte() else 0.toByte() }

    fun writeByteArray(values: ByteArray): ByteArray = values

    fun readShortArray(bytes: ByteArray): ShortArray {
        val values = ShortArray(elementCount(bytes, 2))
        nativeBuffer(bytes).asShortBuffer().get(values)
        return values
    }

    fun readUShortArray(bytes: ByteArray): UShortArray =
        readShortArray(bytes).toUShortArray()

    fun writeShortArray(values: ShortArray): ByteArray {
        val bytes = ByteArray(values.size * 2)
        nativeBuffer(bytes).asShortBuffer().put(values)
        return bytes
    }

    fun writeUShortArray(values: UShortArray): ByteArray =
        writeShortArray(values.asShortArray())

    fun readIntArray(bytes: ByteArray): IntArray {
        val values = IntArray(elementCount(bytes, 4))
        nativeBuffer(bytes).asIntBuffer().get(values)
        return values
    }

    fun readUIntArray(bytes: ByteArray): UIntArray =
        readIntArray(bytes).toUIntArray()

    fun writeIntArray(values: IntArray): ByteArray {
        val bytes = ByteArray(values.size * 4)
        nativeBuffer(bytes).asIntBuffer().put(values)
        return bytes
    }

    fun writeUIntArray(values: UIntArray): ByteArray =
        writeIntArray(values.asIntArray())

    fun readLongArray(bytes: ByteArray): LongArray {
        val values = LongArray(elementCount(bytes, 8))
        nativeBuffer(bytes).asLongBuffer().get(values)
        return values
    }

    fun readULongArray(bytes: ByteArray): ULongArray =
        readLongArray(bytes).toULongArray()

    fun writeLongArray(values: LongArray): ByteArray {
        val bytes = ByteArray(values.size * 8)
        nativeBuffer(bytes).asLongBuffer().put(values)
        return bytes
    }

    fun writeULongArray(values: ULongArray): ByteArray =
        writeLongArray(values.asLongArray())

    fun readFloatArray(bytes: ByteArray): FloatArray {
        val values = FloatArray(elementCount(bytes, 4))
        nativeBuffer(bytes).asFloatBuffer().get(values)
        return values
    }

    fun writeFloatArray(values: FloatArray): ByteArray {
        val bytes = ByteArray(values.size * 4)
        nativeBuffer(bytes).asFloatBuffer().put(values)
        return bytes
    }

    fun readDoubleArray(bytes: ByteArray): DoubleArray {
        val values = DoubleArray(elementCount(bytes, 8))
        nativeBuffer(bytes).asDoubleBuffer().get(values)
        return values
    }

    fun writeDoubleArray(values: DoubleArray): ByteArray {
        val bytes = ByteArray(values.size * 8)
        nativeBuffer(bytes).asDoubleBuffer().put(values)
        return bytes
    }

    private fun nativeBuffer(bytes: ByteArray): java.nio.ByteBuffer =
        java.nio.ByteBuffer
            .wrap(bytes)
            .order(java.nio.ByteOrder.nativeOrder())

    private fun elementCount(bytes: ByteArray, width: Int): Int {
        require(bytes.size % width == 0)
        return bytes.size / width
    }
}

internal class WireReader(private val bytes: ByteArray) {
    private var position = 0

    fun hasRemaining(): Boolean = position < bytes.size

    fun readBool(): Boolean = readI8() != 0.toByte()

    fun readI8(): Byte {
        val value = bytes[position]
        position += 1
        return value
    }

    fun readU8(): UByte = readI8().toUByte()

    fun readI16(): Short {
        val value =
            (bytes[position].toInt() and 0xff) or
                ((bytes[position + 1].toInt() and 0xff) shl 8)
        position += 2
        return value.toShort()
    }

    fun readU16(): UShort = readI16().toUShort()

    fun readI32(): Int {
        val value =
            (bytes[position].toInt() and 0xff) or
                ((bytes[position + 1].toInt() and 0xff) shl 8) or
                ((bytes[position + 2].toInt() and 0xff) shl 16) or
                ((bytes[position + 3].toInt() and 0xff) shl 24)
        position += 4
        return value
    }

    fun readU32(): UInt = readI32().toUInt()

    fun readI64(): Long {
        val low = readI32().toLong() and 0xffffffffL
        val high = readI32().toLong() and 0xffffffffL
        return low or (high shl 32)
    }

    fun readU64(): ULong = readI64().toULong()

    fun readF32(): Float = java.lang.Float.intBitsToFloat(readI32())

    fun readF64(): Double = java.lang.Double.longBitsToDouble(readI64())

    fun readOptionalBool(): Boolean? = readOptional { it.readBool() }

    fun readOptionalI8(): Byte? = readOptional { it.readI8() }

    fun readOptionalU8(): UByte? = readOptional { it.readU8() }

    fun readOptionalI16(): Short? = readOptional { it.readI16() }

    fun readOptionalU16(): UShort? = readOptional { it.readU16() }

    fun readOptionalI32(): Int? = readOptional { it.readI32() }

    fun readOptionalU32(): UInt? = readOptional { it.readU32() }

    fun readOptionalI64(): Long? = readOptional { it.readI64() }

    fun readOptionalU64(): ULong? = readOptional { it.readU64() }

    fun readOptionalF32(): Float? = readOptional { it.readF32() }

    fun readOptionalF64(): Double? = readOptional { it.readF64() }

    fun readString(): String {
        val length = readU32().toInt()
        val value = String(bytes, position, length, Charsets.UTF_8)
        position += length
        return value
    }

    fun readBytes(): ByteArray {
        val length = readU32().toInt()
        val value = bytes.copyOfRange(position, position + length)
        position += length
        return value
    }

    fun readBooleanArray(): BooleanArray {
        val length = readU32().toInt()
        return BooleanArray(length) { readBool() }
    }

    fun readByteArray(): ByteArray = readBytes()

    fun readShortArray(): ShortArray {
        val length = readU32().toInt()
        val byteCount = length * 2
        val values = ShortArray(length)
        java.nio.ByteBuffer
            .wrap(bytes, position, byteCount)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .asShortBuffer()
            .get(values)
        position += byteCount
        return values
    }

    fun readUShortArray(): UShortArray =
        readShortArray().toUShortArray()

    fun readIntArray(): IntArray {
        val length = readU32().toInt()
        val byteCount = length * 4
        val values = IntArray(length)
        java.nio.ByteBuffer
            .wrap(bytes, position, byteCount)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .asIntBuffer()
            .get(values)
        position += byteCount
        return values
    }

    fun readUIntArray(): UIntArray =
        readIntArray().toUIntArray()

    fun readLongArray(): LongArray {
        val length = readU32().toInt()
        val byteCount = length * 8
        val values = LongArray(length)
        java.nio.ByteBuffer
            .wrap(bytes, position, byteCount)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .asLongBuffer()
            .get(values)
        position += byteCount
        return values
    }

    fun readULongArray(): ULongArray =
        readLongArray().toULongArray()

    fun readFloatArray(): FloatArray {
        val length = readU32().toInt()
        val byteCount = length * 4
        val values = FloatArray(length)
        java.nio.ByteBuffer
            .wrap(bytes, position, byteCount)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer()
            .get(values)
        position += byteCount
        return values
    }

    fun readDoubleArray(): DoubleArray {
        val length = readU32().toInt()
        val byteCount = length * 8
        val values = DoubleArray(length)
        java.nio.ByteBuffer
            .wrap(bytes, position, byteCount)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .asDoubleBuffer()
            .get(values)
        position += byteCount
        return values
    }

    fun <T> readOptionalValue(read: (WireReader) -> T): T? = readOptional(read)

    fun <T> readSequence(read: (WireReader) -> T): List<T> {
        val length = readU32().toInt()
        return List(length) { read(this) }
    }

    fun <K, V> readMap(readKey: (WireReader) -> K, readValue: (WireReader) -> V): Map<K, V> {
        val length = readU32().toInt()
        val values = LinkedHashMap<K, V>(length)
        repeat(length) {
            val key = readKey(this)
            if (values.containsKey(key)) {
                throw IllegalArgumentException("duplicate map key")
            }
            values[key] = readValue(this)
        }
        return values
    }

    private inline fun <T> readOptional(read: (WireReader) -> T): T? {
        return when (readU8()) {
            0.toUByte() -> null
            1.toUByte() -> read(this)
            else -> throw IllegalArgumentException("invalid optional wire tag")
        }
    }
}

internal class WireWriter(initialCapacity: Int) {
    private var buffer = java.nio.ByteBuffer
        .allocateDirect(initialCapacity)
        .order(java.nio.ByteOrder.LITTLE_ENDIAN)
    private var position = 0

    fun reset(requiredCapacity: Int) {
        if (buffer.capacity() < requiredCapacity) {
            buffer = java.nio.ByteBuffer
                .allocateDirect(requiredCapacity)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN)
        }
        position = 0
    }

    fun toByteArray(): ByteArray {
        val bytes = ByteArray(position)
        val view = buffer.duplicate()
        view.position(0)
        view.get(bytes, 0, position)
        return bytes
    }

    fun directBuffer(): java.nio.ByteBuffer = buffer

    fun size(): Int = position

    fun writeBool(value: Boolean) {
        ensureCapacity(1)
        buffer.put(position, if (value) 1.toByte() else 0.toByte())
        position += 1
    }

    fun writeI8(value: Byte) {
        ensureCapacity(1)
        buffer.put(position, value)
        position += 1
    }

    fun writeU8(value: UByte) {
        writeI8(value.toByte())
    }

    fun writeI16(value: Short) {
        ensureCapacity(2)
        buffer.putShort(position, value)
        position += 2
    }

    fun writeU16(value: UShort) {
        writeI16(value.toShort())
    }

    fun writeI32(value: Int) {
        ensureCapacity(4)
        buffer.putInt(position, value)
        position += 4
    }

    fun writeU32(value: UInt) {
        writeI32(value.toInt())
    }

    fun writeI64(value: Long) {
        ensureCapacity(8)
        buffer.putLong(position, value)
        position += 8
    }

    fun writeU64(value: ULong) {
        writeI64(value.toLong())
    }

    fun writeF32(value: Float) {
        writeI32(java.lang.Float.floatToRawIntBits(value))
    }

    fun writeF64(value: Double) {
        writeI64(java.lang.Double.doubleToRawLongBits(value))
    }

    fun writeOptionalBool(value: Boolean?) = writeOptional(value) { writer, present ->
        writer.writeBool(present)
    }

    fun writeOptionalI8(value: Byte?) = writeOptional(value) { writer, present ->
        writer.writeI8(present)
    }

    fun writeOptionalU8(value: UByte?) = writeOptional(value) { writer, present ->
        writer.writeU8(present)
    }

    fun writeOptionalI16(value: Short?) = writeOptional(value) { writer, present ->
        writer.writeI16(present)
    }

    fun writeOptionalU16(value: UShort?) = writeOptional(value) { writer, present ->
        writer.writeU16(present)
    }

    fun writeOptionalI32(value: Int?) = writeOptional(value) { writer, present ->
        writer.writeI32(present)
    }

    fun writeOptionalU32(value: UInt?) = writeOptional(value) { writer, present ->
        writer.writeU32(present)
    }

    fun writeOptionalI64(value: Long?) = writeOptional(value) { writer, present ->
        writer.writeI64(present)
    }

    fun writeOptionalU64(value: ULong?) = writeOptional(value) { writer, present ->
        writer.writeU64(present)
    }

    fun writeOptionalF32(value: Float?) = writeOptional(value) { writer, present ->
        writer.writeF32(present)
    }

    fun writeOptionalF64(value: Double?) = writeOptional(value) { writer, present ->
        writer.writeF64(present)
    }

    fun writeString(value: String) {
        val bytes = value.toByteArray(Charsets.UTF_8)
        writeU32(bytes.size.toUInt())
        writeBytesRaw(bytes)
    }

    fun writeBytes(value: ByteArray) {
        writeU32(value.size.toUInt())
        writeBytesRaw(value)
    }

    fun writeBooleanArray(values: BooleanArray) {
        writeU32(values.size.toUInt())
        values.forEach { writeBool(it) }
    }

    fun writeByteArray(values: ByteArray) = writeBytes(values)

    fun writeShortArray(values: ShortArray) {
        writeU32(values.size.toUInt())
        val byteCount = values.size * 2
        ensureCapacity(byteCount)
        val view = buffer.duplicate().order(java.nio.ByteOrder.LITTLE_ENDIAN)
        view.position(position)
        view.asShortBuffer().put(values)
        position += byteCount
    }

    fun writeUShortArray(values: UShortArray) =
        writeShortArray(values.asShortArray())

    fun writeIntArray(values: IntArray) {
        writeU32(values.size.toUInt())
        val byteCount = values.size * 4
        ensureCapacity(byteCount)
        val view = buffer.duplicate().order(java.nio.ByteOrder.LITTLE_ENDIAN)
        view.position(position)
        view.asIntBuffer().put(values)
        position += byteCount
    }

    fun writeUIntArray(values: UIntArray) =
        writeIntArray(values.asIntArray())

    fun writeLongArray(values: LongArray) {
        writeU32(values.size.toUInt())
        val byteCount = values.size * 8
        ensureCapacity(byteCount)
        val view = buffer.duplicate().order(java.nio.ByteOrder.LITTLE_ENDIAN)
        view.position(position)
        view.asLongBuffer().put(values)
        position += byteCount
    }

    fun writeULongArray(values: ULongArray) =
        writeLongArray(values.asLongArray())

    fun writeFloatArray(values: FloatArray) {
        writeU32(values.size.toUInt())
        val byteCount = values.size * 4
        ensureCapacity(byteCount)
        val view = buffer.duplicate().order(java.nio.ByteOrder.LITTLE_ENDIAN)
        view.position(position)
        view.asFloatBuffer().put(values)
        position += byteCount
    }

    fun writeDoubleArray(values: DoubleArray) {
        writeU32(values.size.toUInt())
        val byteCount = values.size * 8
        ensureCapacity(byteCount)
        val view = buffer.duplicate().order(java.nio.ByteOrder.LITTLE_ENDIAN)
        view.position(position)
        view.asDoubleBuffer().put(values)
        position += byteCount
    }

    fun <T> writeOptionalValue(value: T?, write: (WireWriter, T) -> Unit) {
        writeOptional(value, write)
    }

    fun <T> writeSequence(value: Iterable<T>, count: Int, write: (WireWriter, T) -> Unit) {
        writeU32(count.toUInt())
        value.forEach { item -> write(this, item) }
    }

    fun <K, V> writeMap(
        value: Map<K, V>,
        writeKey: (WireWriter, K) -> Unit,
        writeValue: (WireWriter, V) -> Unit,
    ) {
        writeU32(value.size.toUInt())
        value.entries.forEach { entry ->
            writeKey(this, entry.key)
            writeValue(this, entry.value)
        }
    }

    private fun writeBytesRaw(bytes: ByteArray) {
        ensureCapacity(bytes.size)
        val view = buffer.duplicate().order(java.nio.ByteOrder.LITTLE_ENDIAN)
        view.position(position)
        view.put(bytes)
        position += bytes.size
    }

    private fun ensureCapacity(needed: Int) {
        val required = position + needed
        if (required <= buffer.capacity()) {
            return
        }
        val nextCapacity = maxOf(buffer.capacity() * 2, required)
        val next = java.nio.ByteBuffer
            .allocateDirect(nextCapacity)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
        val source = buffer.duplicate().order(java.nio.ByteOrder.LITTLE_ENDIAN)
        source.limit(position)
        source.position(0)
        next.put(source)
        buffer = next
    }

    private inline fun <T> writeOptional(value: T?, write: (WireWriter, T) -> Unit) {
        if (value == null) {
            writeU8(0.toUByte())
            return
        }
        writeU8(1.toUByte())
        write(this, value)
    }
}

private const val MAX_CACHED_WIRE_WRITER_BYTES: Int = 1024 * 1024

private class WireWriterPoolState(private val cacheSize: Int = 4) {
    private val cachedWriters: Array<WireWriter?> = arrayOfNulls(cacheSize)
    private var depth = 0

    fun acquire(requiredCapacity: Int): BorrowedWireWriter {
        val slot = depth
        depth = slot + 1
        val shouldCache = requiredCapacity <= MAX_CACHED_WIRE_WRITER_BYTES && slot < cacheSize
        val writer = if (shouldCache) {
            cachedWriters[slot] ?: WireWriter(requiredCapacity).also { cachedWriters[slot] = it }
        } else {
            WireWriter(requiredCapacity)
        }

        writer.reset(requiredCapacity)
        return BorrowedWireWriter(this, writer)
    }

    fun release() {
        depth -= 1
    }
}

private class BorrowedWireWriter(
    private val state: WireWriterPoolState,
    val writer: WireWriter,
) : AutoCloseable {
    fun bytes(): ByteArray = writer.toByteArray()

    fun directBuffer(): java.nio.ByteBuffer = writer.directBuffer()

    fun size(): Int = writer.size()

    override fun close() {
        state.release()
    }
}

private object WireWriterPool {
    private val state: ThreadLocal<WireWriterPoolState> =
        ThreadLocal.withInitial { WireWriterPoolState() }

    fun acquire(requiredCapacity: Int): BorrowedWireWriter {
        val poolState = state.get() ?: WireWriterPoolState().also { state.set(it) }
        return poolState.acquire(requiredCapacity)
    }
}

private inline fun <K, V> Map<K, V>.wireSize(
    keySize: (K) -> Int,
    valueSize: (V) -> Int,
): Int = 4 + entries.sumOf { entry -> keySize(entry.key) + valueSize(entry.value) }

@Suppress("FunctionName")
private object Native {
    init {
        val androidLibrary = "xybrid_bolt"
        val desktopPreferredLibrary = "xybrid_bolt_jni"
        val desktopFallbackLibrary = "xybrid_bolt"
        val vmName = System.getProperty("java.vm.name").orEmpty()
        val isAndroidRuntime =
            vmName.contains("dalvik", ignoreCase = true) ||
            vmName.contains("art", ignoreCase = true)
        if (isAndroidRuntime) {
            System.loadLibrary(androidLibrary)
        } else {
            loadDesktopLibraries(desktopPreferredLibrary, desktopFallbackLibrary)
        }
    }

    @Volatile
    private var bundledLibraryDirectory: java.io.File? = null

    private fun loadDesktopLibraries(preferredLibrary: String, fallbackLibrary: String) {
        var preferredFailure = tryLoadDesktopLibrary(preferredLibrary)
        if (preferredFailure == null) {
            return
        }

        if (tryLoadOptionalDesktopLibrary(fallbackLibrary)) {
            preferredFailure = tryLoadDesktopLibrary(preferredLibrary)
            if (preferredFailure == null) {
                return
            }
        }

        throw preferredFailure
    }

    private fun tryLoadDesktopLibrary(libraryName: String): UnsatisfiedLinkError? {
        try {
            if (loadBundledLibraryIfPresent(libraryName) || loadExternalLibraryIfPresent(libraryName)) {
                return null
            }
            return UnsatisfiedLinkError("Could not load native library '$libraryName'")
        } catch (error: UnsatisfiedLinkError) {
            return error
        }
    }

    private fun tryLoadOptionalDesktopLibrary(libraryName: String): Boolean {
        return try {
            loadBundledLibraryIfPresent(libraryName) || loadExternalLibraryIfPresent(libraryName)
        } catch (_: UnsatisfiedLinkError) {
            false
        }
    }

    private fun loadExternalLibraryIfPresent(libraryName: String): Boolean {
        return try {
            System.loadLibrary(libraryName)
            true
        } catch (_: UnsatisfiedLinkError) {
            false
        }
    }

    private fun loadBundledLibraryIfPresent(libraryName: String): Boolean {
        val mappedName = System.mapLibraryName(libraryName)
        for (resourcePath in bundledLibraryResourceCandidates(mappedName)) {
            Native::class.java.getResourceAsStream(resourcePath)?.use { input ->
                val extracted = extractBundledLibrary(resourcePath, input)
                System.load(extracted.absolutePath)
                return true
            }
        }
        return false
    }

    private fun extractBundledLibrary(
        resourcePath: String,
        input: java.io.InputStream,
    ): java.io.File {
        val fileName = resourcePath.substringAfterLast('/')
        val extracted = java.io.File(bundledLibraryDirectory(), fileName)
        if (!extracted.isFile) {
            java.io.FileOutputStream(extracted).use { output ->
                input.copyTo(output)
            }
            extracted.deleteOnExit()
        }
        return extracted
    }

    private fun bundledLibraryDirectory(): java.io.File {
        bundledLibraryDirectory?.let { return it }
        synchronized(this) {
            bundledLibraryDirectory?.let { return it }
            val created = java.io.File.createTempFile("boltffi-native-", "")
            if (!created.delete() || !created.mkdir()) {
                throw java.io.IOException("failed to create temp directory for bundled native extraction")
            }
            created.deleteOnExit()
            bundledLibraryDirectory = created
            return created
        }
    }

    private fun bundledLibraryResourceCandidates(mappedName: String): List<String> {
        val candidates = mutableListOf<String>()
        for (directory in desktopNativeDirectories()) {
            candidates += "/$directory/$mappedName"
            candidates += "/native/$directory/$mappedName"
        }
        candidates += "/$mappedName"
        return candidates
    }

    private fun desktopNativeDirectories(): List<String> {
        val osName = System.getProperty("os.name").orEmpty().lowercase()
        val osArch = System.getProperty("os.arch").orEmpty().lowercase()
        return when {
            (osName.contains("mac") || osName.contains("darwin")) &&
                (osArch == "aarch64" || osArch == "arm64") ->
                listOf("darwin-arm64", "darwin-aarch64")
            (osName.contains("mac") || osName.contains("darwin")) &&
                (osArch == "x86_64" || osArch == "amd64") ->
                listOf("darwin-x86_64", "darwin-x86-64")
            (osName.contains("linux")) &&
                (osArch == "x86_64" || osArch == "amd64") ->
                listOf("linux-x86_64", "linux-x86-64")
            (osName.contains("linux")) &&
                (osArch == "aarch64" || osArch == "arm64") ->
                listOf("linux-aarch64", "linux-arm64")
            (osName.contains("windows")) &&
                (osArch == "x86_64" || osArch == "amd64") ->
                listOf("windows-x86_64", "windows-x86-64", "win32-x86_64")
            (osName.contains("windows")) &&
                (osArch == "aarch64" || osArch == "arm64") ->
                listOf("windows-aarch64", "windows-arm64", "win32-arm64")
            else -> emptyList()
        }
    }
    @JvmStatic external fun boltffi_release_class_xybrid_bolt_xybrid_model(handle: Long): Unit
    @JvmStatic external fun boltffi_init_class_xybrid_bolt_xybrid_model_from_registry(id: java.nio.ByteBuffer, __boltffi_id_len: Int): Long
    @JvmStatic external fun boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative(id: java.nio.ByteBuffer, __boltffi_id_len: Int): Long
    @JvmStatic external fun boltffi_init_class_xybrid_bolt_xybrid_model_from_directory(path: java.nio.ByteBuffer, __boltffi_path_len: Int): Long
    @JvmStatic external fun boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle(path: java.nio.ByteBuffer, __boltffi_path_len: Int): Long
    @JvmStatic external fun boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface(repo: java.nio.ByteBuffer, __boltffi_repo_len: Int): Long
    @JvmStatic external fun boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision(repo: java.nio.ByteBuffer, __boltffi_repo_len: Int, revision: java.nio.ByteBuffer, __boltffi_revision_len: Int): Long
    @JvmStatic external fun boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file(path: java.nio.ByteBuffer, __boltffi_path_len: Int): Long
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_model_id(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_version(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_output_type(`receiver`: Long): Int
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded(`receiver`: Long): Boolean
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving(`receiver`: Long): Boolean
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_download_status(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_await_download(`receiver`: Long, timeout_ms: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming(`receiver`: Long): Boolean
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming(`receiver`: Long): Boolean
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_is_llm(`receiver`: Long): Boolean
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_has_voices(`receiver`: Long): Boolean
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_voices(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_default_voice(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_voice(`receiver`: Long, voice_id: java.nio.ByteBuffer, __boltffi_voice_id_len: Int): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_run(`receiver`: Long, envelope: java.nio.ByteBuffer, __boltffi_envelope_len: Int, options: java.nio.ByteBuffer, __boltffi_options_len: Int): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_run_stream(`receiver`: Long, envelope: java.nio.ByteBuffer, __boltffi_envelope_len: Int, options: java.nio.ByteBuffer, __boltffi_options_len: Int): Long
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_stream_next(`receiver`: Long, stream_id: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_stream_result(`receiver`: Long, stream_id: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_stream_close(`receiver`: Long, stream_id: Long): Unit
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context(`receiver`: Long, envelope: java.nio.ByteBuffer, __boltffi_envelope_len: Int, context: Long, options: java.nio.ByteBuffer, __boltffi_options_len: Int): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context(`receiver`: Long, envelope: java.nio.ByteBuffer, __boltffi_envelope_len: Int, context: Long, options: java.nio.ByteBuffer, __boltffi_options_len: Int): Long
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_warmup(`receiver`: Long): Unit
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_model_unload(`receiver`: Long): Unit
    @JvmStatic external fun boltffi_release_class_xybrid_bolt_xybrid_conversation_context(handle: Long): Unit
    @JvmStatic external fun boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new(): Long
    @JvmStatic external fun boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id(id: java.nio.ByteBuffer, __boltffi_id_len: Int): Long
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push(`receiver`: Long, envelope: java.nio.ByteBuffer, __boltffi_envelope_len: Int): Unit
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system(`receiver`: Long, envelope: java.nio.ByteBuffer, __boltffi_envelope_len: Int): Unit
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear(`receiver`: Long): Unit
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len(`receiver`: Long): Int
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system(`receiver`: Long): Boolean
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len(`receiver`: Long, len: Int): Unit
    @JvmStatic external fun boltffi_release_class_xybrid_bolt_xybrid_telemetry_config(handle: Long): Unit
    @JvmStatic external fun boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new(api_key: java.nio.ByteBuffer, __boltffi_api_key_len: Int): Long
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint(`receiver`: Long, endpoint: java.nio.ByteBuffer, __boltffi_endpoint_len: Int): Unit
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version(`receiver`: Long, version: java.nio.ByteBuffer, __boltffi_version_len: Int): Unit
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label(`receiver`: Long, label: java.nio.ByteBuffer, __boltffi_label_len: Int): Unit
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute(`receiver`: Long, key: java.nio.ByteBuffer, __boltffi_key_len: Int, value: java.nio.ByteBuffer, __boltffi_value_len: Int): Unit
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size(`receiver`: Long, batch_size: Int): Unit
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs(`receiver`: Long, secs: Int): Unit
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init(`receiver`: Long): Unit
    @JvmStatic external fun boltffi_release_class_xybrid_bolt_xybrid_bundle(handle: Long): Unit
    @JvmStatic external fun boltffi_init_class_xybrid_bolt_xybrid_bundle_open(path: java.nio.ByteBuffer, __boltffi_path_len: Int): Long
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_bundle_version(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_bundle_target(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_bundle_hash(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata(`receiver`: Long): Boolean
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count(`receiver`: Long): Int
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name(`receiver`: Long, index: Int): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json(`receiver`: Long): ByteArray?
    @JvmStatic external fun boltffi_method_class_xybrid_bolt_xybrid_bundle_extract(`receiver`: Long, output_dir: java.nio.ByteBuffer, __boltffi_output_dir_len: Int): Unit
    @JvmStatic external fun boltffi_function_xybrid_bolt_tool_results_envelope(user_text: java.nio.ByteBuffer, __boltffi_user_text_len: Int, prior_assistant_text: java.nio.ByteBuffer, __boltffi_prior_assistant_text_len: Int, results: java.nio.ByteBuffer, __boltffi_results_len: Int): ByteArray?
    @JvmStatic external fun boltffi_function_xybrid_bolt_json_schema_to_gbnf(schema_json: java.nio.ByteBuffer, __boltffi_schema_json_len: Int): ByteArray?
    @JvmStatic external fun boltffi_function_xybrid_bolt_set_thermal_state(state: Int): Unit
    @JvmStatic external fun boltffi_function_xybrid_bolt_clear_thermal_state(): Unit
    @JvmStatic external fun boltffi_function_xybrid_bolt_set_battery_level(percent: Byte): Unit
    @JvmStatic external fun boltffi_function_xybrid_bolt_clear_battery_level(): Unit
    @JvmStatic external fun boltffi_function_xybrid_bolt_configure_runtime(api_key: java.nio.ByteBuffer, __boltffi_api_key_len: Int, gateway_url: java.nio.ByteBuffer, __boltffi_gateway_url_len: Int, ingest_url: java.nio.ByteBuffer, __boltffi_ingest_url_len: Int): Unit
    @JvmStatic external fun boltffi_function_xybrid_bolt_init_sdk_cache_dir(cache_dir: java.nio.ByteBuffer, __boltffi_cache_dir_len: Int): Unit
    @JvmStatic external fun boltffi_function_xybrid_bolt_set_binding(binding: java.nio.ByteBuffer, __boltffi_binding_len: Int): Unit
    @JvmStatic external fun boltffi_function_xybrid_bolt_set_api_key(api_key: java.nio.ByteBuffer, __boltffi_api_key_len: Int): Unit
    @JvmStatic external fun boltffi_function_xybrid_bolt_set_provider_api_key(provider: java.nio.ByteBuffer, __boltffi_provider_len: Int, api_key: java.nio.ByteBuffer, __boltffi_api_key_len: Int): Unit
    @JvmStatic external fun boltffi_function_xybrid_bolt_set_platform_url(url: java.nio.ByteBuffer, __boltffi_url_len: Int): Unit
    @JvmStatic external fun boltffi_function_xybrid_bolt_set_speculative_cloud(enabled: Boolean): Unit
    @JvmStatic external fun boltffi_function_xybrid_bolt_has_api_key(): Boolean
    @JvmStatic external fun boltffi_function_xybrid_bolt_is_speculative_cloud_enabled(): Boolean
    @JvmStatic external fun boltffi_function_xybrid_bolt_will_speculate_for_model(model_id: java.nio.ByteBuffer, __boltffi_model_id_len: Int): Boolean
    @JvmStatic external fun boltffi_function_xybrid_bolt_version(): ByteArray?
    @JvmStatic external fun boltffi_function_xybrid_bolt_release_memory(): Int
    @JvmStatic external fun boltffi_function_xybrid_bolt_set_auto_release(enabled: Boolean): Unit
    @JvmStatic external fun boltffi_function_xybrid_bolt_is_auto_release_enabled(): Boolean
    @JvmStatic external fun boltffi_function_xybrid_bolt_telemetry_default_endpoint(): ByteArray?
    @JvmStatic external fun boltffi_function_xybrid_bolt_telemetry_flush(): Unit
    @JvmStatic external fun boltffi_function_xybrid_bolt_telemetry_shutdown(): Unit
}


data class XybridMetadataEntry(
    val key: String,
    val value: String
) {
    internal fun wireSize(): Int {
        return 4 + Utf8Codec.maxBytes(this.key) + 4 + Utf8Codec.maxBytes(this.value)
    }

    internal fun writeTo(writer: WireWriter) {
        writer.writeString(this.key)
        writer.writeString(this.value)
    }

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridMetadataEntry {
            return XybridMetadataEntry(
                reader.readString(),
                reader.readString()
            )
        }

        internal fun fromByteArray(bytes: ByteArray): XybridMetadataEntry {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


data class XybridEnvelope(
    val kind: XybridEnvelopeKind,
    val metadata: List<XybridMetadataEntry>
) {
    internal fun wireSize(): Int {
        return this.kind.wireSize() + 4 + this.metadata.sumOf { __boltffi_value_0 -> (__boltffi_value_0.wireSize()).toInt() }
    }

    internal fun writeTo(writer: WireWriter) {
        this.kind.writeTo(writer)
        writer.writeSequence(this.metadata, this.metadata.size, { writer, __boltffi_value_0 -> __boltffi_value_0.writeTo(writer) })
    }

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridEnvelope {
            return XybridEnvelope(
                XybridEnvelopeKind.fromReader(reader),
                reader.readSequence({ reader -> XybridMetadataEntry.fromReader(reader) })
            )
        }

        internal fun fromByteArray(bytes: ByteArray): XybridEnvelope {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


data class XybridToolDefinition(
    val name: String,
    val description: String,
    val parametersJson: String
) {
    internal fun wireSize(): Int {
        return 4 + Utf8Codec.maxBytes(this.name) + 4 + Utf8Codec.maxBytes(this.description) + 4 + Utf8Codec.maxBytes(this.parametersJson)
    }

    internal fun writeTo(writer: WireWriter) {
        writer.writeString(this.name)
        writer.writeString(this.description)
        writer.writeString(this.parametersJson)
    }

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridToolDefinition {
            return XybridToolDefinition(
                reader.readString(),
                reader.readString(),
                reader.readString()
            )
        }

        internal fun fromByteArray(bytes: ByteArray): XybridToolDefinition {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


data class XybridToolCall(
    val id: String,
    val name: String,
    val argumentsJson: String
) {
    internal fun wireSize(): Int {
        return 4 + Utf8Codec.maxBytes(this.id) + 4 + Utf8Codec.maxBytes(this.name) + 4 + Utf8Codec.maxBytes(this.argumentsJson)
    }

    internal fun writeTo(writer: WireWriter) {
        writer.writeString(this.id)
        writer.writeString(this.name)
        writer.writeString(this.argumentsJson)
    }

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridToolCall {
            return XybridToolCall(
                reader.readString(),
                reader.readString(),
                reader.readString()
            )
        }

        internal fun fromByteArray(bytes: ByteArray): XybridToolCall {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


data class XybridToolResult(
    val callId: String,
    val name: String,
    val contentJson: String
) {
    internal fun wireSize(): Int {
        return 4 + Utf8Codec.maxBytes(this.callId) + 4 + Utf8Codec.maxBytes(this.name) + 4 + Utf8Codec.maxBytes(this.contentJson)
    }

    internal fun writeTo(writer: WireWriter) {
        writer.writeString(this.callId)
        writer.writeString(this.name)
        writer.writeString(this.contentJson)
    }

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridToolResult {
            return XybridToolResult(
                reader.readString(),
                reader.readString(),
                reader.readString()
            )
        }

        internal fun fromByteArray(bytes: ByteArray): XybridToolResult {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


data class XybridGenerationConfig(
    val maxTokens: UInt?,
    val temperature: Float?,
    val topP: Float?,
    val minP: Float?,
    val topK: UInt?,
    val repetitionPenalty: Float?,
    val stopSequences: List<String>,
    val grammar: String?,
    val tools: List<XybridToolDefinition>
) {
    internal fun wireSize(): Int {
        return 1 + (this.maxTokens?.let { __boltffi_value_0 -> 4 } ?: 0) + 1 + (this.temperature?.let { __boltffi_value_0 -> 4 } ?: 0) + 1 + (this.topP?.let { __boltffi_value_0 -> 4 } ?: 0) + 1 + (this.minP?.let { __boltffi_value_0 -> 4 } ?: 0) + 1 + (this.topK?.let { __boltffi_value_0 -> 4 } ?: 0) + 1 + (this.repetitionPenalty?.let { __boltffi_value_0 -> 4 } ?: 0) + 4 + this.stopSequences.sumOf { __boltffi_value_0 -> (4 + Utf8Codec.maxBytes(__boltffi_value_0)).toInt() } + 1 + (this.grammar?.let { __boltffi_value_0 -> 4 + Utf8Codec.maxBytes(__boltffi_value_0) } ?: 0) + 4 + this.tools.sumOf { __boltffi_value_0 -> (__boltffi_value_0.wireSize()).toInt() }
    }

    internal fun writeTo(writer: WireWriter) {
        writer.writeOptionalValue(this.maxTokens, { writer, __boltffi_value_0 -> writer.writeU32(__boltffi_value_0) })
        writer.writeOptionalValue(this.temperature, { writer, __boltffi_value_0 -> writer.writeF32(__boltffi_value_0) })
        writer.writeOptionalValue(this.topP, { writer, __boltffi_value_0 -> writer.writeF32(__boltffi_value_0) })
        writer.writeOptionalValue(this.minP, { writer, __boltffi_value_0 -> writer.writeF32(__boltffi_value_0) })
        writer.writeOptionalValue(this.topK, { writer, __boltffi_value_0 -> writer.writeU32(__boltffi_value_0) })
        writer.writeOptionalValue(this.repetitionPenalty, { writer, __boltffi_value_0 -> writer.writeF32(__boltffi_value_0) })
        writer.writeSequence(this.stopSequences, this.stopSequences.size, { writer, __boltffi_value_0 -> writer.writeString(__boltffi_value_0) })
        writer.writeOptionalValue(this.grammar, { writer, __boltffi_value_0 -> writer.writeString(__boltffi_value_0) })
        writer.writeSequence(this.tools, this.tools.size, { writer, __boltffi_value_0 -> __boltffi_value_0.writeTo(writer) })
    }

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridGenerationConfig {
            return XybridGenerationConfig(
                reader.readOptionalValue({ reader -> reader.readU32() }),
                reader.readOptionalValue({ reader -> reader.readF32() }),
                reader.readOptionalValue({ reader -> reader.readF32() }),
                reader.readOptionalValue({ reader -> reader.readF32() }),
                reader.readOptionalValue({ reader -> reader.readU32() }),
                reader.readOptionalValue({ reader -> reader.readF32() }),
                reader.readSequence({ reader -> reader.readString() }),
                reader.readOptionalValue({ reader -> reader.readString() }),
                reader.readSequence({ reader -> XybridToolDefinition.fromReader(reader) })
            )
        }

        internal fun fromByteArray(bytes: ByteArray): XybridGenerationConfig {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


data class XybridRunOptions(
    val generationConfig: XybridGenerationConfig?,
    val abortOn: List<XybridAbortSignal>,
    val fallbackToCloud: Boolean,
    val maxGraceTokens: UInt,
    val correlationId: String?
) {
    internal fun wireSize(): Int {
        return 1 + (this.generationConfig?.let { __boltffi_value_0 -> __boltffi_value_0.wireSize() } ?: 0) + 4 + this.abortOn.sumOf { __boltffi_value_0 -> (4).toInt() } + 1 + 4 + 1 + (this.correlationId?.let { __boltffi_value_0 -> 4 + Utf8Codec.maxBytes(__boltffi_value_0) } ?: 0)
    }

    internal fun writeTo(writer: WireWriter) {
        writer.writeOptionalValue(this.generationConfig, { writer, __boltffi_value_0 -> __boltffi_value_0.writeTo(writer) })
        writer.writeSequence(this.abortOn, this.abortOn.size, { writer, __boltffi_value_0 -> writer.writeI32(__boltffi_value_0.value) })
        writer.writeBool(this.fallbackToCloud)
        writer.writeU32(this.maxGraceTokens)
        writer.writeOptionalValue(this.correlationId, { writer, __boltffi_value_0 -> writer.writeString(__boltffi_value_0) })
    }

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridRunOptions {
            return XybridRunOptions(
                reader.readOptionalValue({ reader -> XybridGenerationConfig.fromReader(reader) }),
                reader.readSequence({ reader -> XybridAbortSignal.fromValue(reader.readI32()) }),
                reader.readBool(),
                reader.readU32(),
                reader.readOptionalValue({ reader -> reader.readString() })
            )
        }

        internal fun fromByteArray(bytes: ByteArray): XybridRunOptions {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


data class XybridStageLatency(
    val stageId: String,
    val latencyMs: UInt
) {
    internal fun wireSize(): Int {
        return 4 + Utf8Codec.maxBytes(this.stageId) + 4
    }

    internal fun writeTo(writer: WireWriter) {
        writer.writeString(this.stageId)
        writer.writeU32(this.latencyMs)
    }

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridStageLatency {
            return XybridStageLatency(
                reader.readString(),
                reader.readU32()
            )
        }

        internal fun fromByteArray(bytes: ByteArray): XybridStageLatency {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


data class XybridInferenceMetrics(
    val totalMs: UInt,
    val ttftMs: UInt?,
    val tokensPerSecond: Float?,
    val prefillTps: Float?,
    val decodeTps: Float?,
    val tokensOut: UInt?,
    val stageLatenciesMs: List<XybridStageLatency>
) {
    internal fun wireSize(): Int {
        return 4 + 1 + (this.ttftMs?.let { __boltffi_value_0 -> 4 } ?: 0) + 1 + (this.tokensPerSecond?.let { __boltffi_value_0 -> 4 } ?: 0) + 1 + (this.prefillTps?.let { __boltffi_value_0 -> 4 } ?: 0) + 1 + (this.decodeTps?.let { __boltffi_value_0 -> 4 } ?: 0) + 1 + (this.tokensOut?.let { __boltffi_value_0 -> 4 } ?: 0) + 4 + this.stageLatenciesMs.sumOf { __boltffi_value_0 -> (__boltffi_value_0.wireSize()).toInt() }
    }

    internal fun writeTo(writer: WireWriter) {
        writer.writeU32(this.totalMs)
        writer.writeOptionalValue(this.ttftMs, { writer, __boltffi_value_0 -> writer.writeU32(__boltffi_value_0) })
        writer.writeOptionalValue(this.tokensPerSecond, { writer, __boltffi_value_0 -> writer.writeF32(__boltffi_value_0) })
        writer.writeOptionalValue(this.prefillTps, { writer, __boltffi_value_0 -> writer.writeF32(__boltffi_value_0) })
        writer.writeOptionalValue(this.decodeTps, { writer, __boltffi_value_0 -> writer.writeF32(__boltffi_value_0) })
        writer.writeOptionalValue(this.tokensOut, { writer, __boltffi_value_0 -> writer.writeU32(__boltffi_value_0) })
        writer.writeSequence(this.stageLatenciesMs, this.stageLatenciesMs.size, { writer, __boltffi_value_0 -> __boltffi_value_0.writeTo(writer) })
    }

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridInferenceMetrics {
            return XybridInferenceMetrics(
                reader.readU32(),
                reader.readOptionalValue({ reader -> reader.readU32() }),
                reader.readOptionalValue({ reader -> reader.readF32() }),
                reader.readOptionalValue({ reader -> reader.readF32() }),
                reader.readOptionalValue({ reader -> reader.readF32() }),
                reader.readOptionalValue({ reader -> reader.readU32() }),
                reader.readSequence({ reader -> XybridStageLatency.fromReader(reader) })
            )
        }

        internal fun fromByteArray(bytes: ByteArray): XybridInferenceMetrics {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


data class XybridResult(
    val envelope: XybridEnvelope,
    val outputType: XybridOutputType,
    val modelId: String,
    val latencyMs: UInt,
    val executionTarget: XybridExecutionTarget,
    val metrics: XybridInferenceMetrics,
    val toolCalls: List<XybridToolCall>,
    val reasoningContent: String? = null
) {
    internal fun wireSize(): Int {
        return this.envelope.wireSize() + 4 + 4 + Utf8Codec.maxBytes(this.modelId) + 4 + 4 + this.metrics.wireSize() + 4 + this.toolCalls.sumOf { __boltffi_value_0 -> (__boltffi_value_0.wireSize()).toInt() } + 1 + (this.reasoningContent?.let { __boltffi_value_0 -> 4 + Utf8Codec.maxBytes(__boltffi_value_0) } ?: 0)
    }

    internal fun writeTo(writer: WireWriter) {
        this.envelope.writeTo(writer)
        writer.writeI32(this.outputType.value)
        writer.writeString(this.modelId)
        writer.writeU32(this.latencyMs)
        writer.writeI32(this.executionTarget.value)
        this.metrics.writeTo(writer)
        writer.writeSequence(this.toolCalls, this.toolCalls.size, { writer, __boltffi_value_0 -> __boltffi_value_0.writeTo(writer) })
        writer.writeOptionalValue(this.reasoningContent, { writer, __boltffi_value_0 -> writer.writeString(__boltffi_value_0) })
    }

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridResult {
            val envelope = XybridEnvelope.fromReader(reader)
            val outputType = XybridOutputType.fromValue(reader.readI32())
            val modelId = reader.readString()
            val latencyMs = reader.readU32()
            val executionTarget = XybridExecutionTarget.fromValue(reader.readI32())
            val metrics = XybridInferenceMetrics.fromReader(reader)
            val toolCalls = reader.readSequence({ reader -> XybridToolCall.fromReader(reader) })
            val reasoningContent = if (reader.hasRemaining()) {
                reader.readOptionalValue({ reader -> reader.readString() })
            } else {
                envelope.metadata.firstOrNull { it.key == "reasoning_content" }?.value
            }
            return XybridResult(
                envelope,
                outputType,
                modelId,
                latencyMs,
                executionTarget,
                metrics,
                toolCalls,
                reasoningContent
            )
        }

        internal fun fromByteArray(bytes: ByteArray): XybridResult {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


data class XybridDownloadStatus(
    val state: XybridDownloadState,
    val progress: Float
) {
    internal fun wireSize(): Int {
        return 4 + 4
    }

    internal fun writeTo(writer: WireWriter) {
        writer.writeI32(this.state.value)
        writer.writeF32(this.progress)
    }

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridDownloadStatus {
            return XybridDownloadStatus(
                XybridDownloadState.fromValue(reader.readI32()),
                reader.readF32()
            )
        }

        internal fun fromByteArray(bytes: ByteArray): XybridDownloadStatus {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


data class XybridStreamToken(
    val token: String,
    val tokenId: Long?,
    val index: ULong,
    val cumulativeText: String,
    val finishReason: String?,
    val toolCalls: List<XybridToolCall>,
    val rawText: String?
) {
    internal fun wireSize(): Int {
        return 4 + Utf8Codec.maxBytes(this.token) + 1 + (this.tokenId?.let { __boltffi_value_0 -> 8 } ?: 0) + 8 + 4 + Utf8Codec.maxBytes(this.cumulativeText) + 1 + (this.finishReason?.let { __boltffi_value_0 -> 4 + Utf8Codec.maxBytes(__boltffi_value_0) } ?: 0) + 4 + this.toolCalls.sumOf { __boltffi_value_0 -> (__boltffi_value_0.wireSize()).toInt() } + 1 + (this.rawText?.let { __boltffi_value_0 -> 4 + Utf8Codec.maxBytes(__boltffi_value_0) } ?: 0)
    }

    internal fun writeTo(writer: WireWriter) {
        writer.writeString(this.token)
        writer.writeOptionalValue(this.tokenId, { writer, __boltffi_value_0 -> writer.writeI64(__boltffi_value_0) })
        writer.writeU64(this.index)
        writer.writeString(this.cumulativeText)
        writer.writeOptionalValue(this.finishReason, { writer, __boltffi_value_0 -> writer.writeString(__boltffi_value_0) })
        writer.writeSequence(this.toolCalls, this.toolCalls.size, { writer, __boltffi_value_0 -> __boltffi_value_0.writeTo(writer) })
        writer.writeOptionalValue(this.rawText, { writer, __boltffi_value_0 -> writer.writeString(__boltffi_value_0) })
    }

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridStreamToken {
            return XybridStreamToken(
                reader.readString(),
                reader.readOptionalValue({ reader -> reader.readI64() }),
                reader.readU64(),
                reader.readString(),
                reader.readOptionalValue({ reader -> reader.readString() }),
                reader.readSequence({ reader -> XybridToolCall.fromReader(reader) }),
                reader.readOptionalValue({ reader -> reader.readString() })
            )
        }

        internal fun fromByteArray(bytes: ByteArray): XybridStreamToken {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


data class XybridStreamEvent(
    val kind: XybridStreamEventKind,
    val token: XybridStreamToken?
) {
    internal fun wireSize(): Int {
        return 4 + 1 + (this.token?.let { __boltffi_value_0 -> __boltffi_value_0.wireSize() } ?: 0)
    }

    internal fun writeTo(writer: WireWriter) {
        writer.writeI32(this.kind.value)
        writer.writeOptionalValue(this.token, { writer, __boltffi_value_0 -> __boltffi_value_0.writeTo(writer) })
    }

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridStreamEvent {
            return XybridStreamEvent(
                XybridStreamEventKind.fromValue(reader.readI32()),
                reader.readOptionalValue({ reader -> XybridStreamToken.fromReader(reader) })
            )
        }

        internal fun fromByteArray(bytes: ByteArray): XybridStreamEvent {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


data class XybridVoiceInfo(
    val id: String,
    val name: String,
    val gender: String?,
    val language: String?,
    val style: String?
) {
    internal fun wireSize(): Int {
        return 4 + Utf8Codec.maxBytes(this.id) + 4 + Utf8Codec.maxBytes(this.name) + 1 + (this.gender?.let { __boltffi_value_0 -> 4 + Utf8Codec.maxBytes(__boltffi_value_0) } ?: 0) + 1 + (this.language?.let { __boltffi_value_0 -> 4 + Utf8Codec.maxBytes(__boltffi_value_0) } ?: 0) + 1 + (this.style?.let { __boltffi_value_0 -> 4 + Utf8Codec.maxBytes(__boltffi_value_0) } ?: 0)
    }

    internal fun writeTo(writer: WireWriter) {
        writer.writeString(this.id)
        writer.writeString(this.name)
        writer.writeOptionalValue(this.gender, { writer, __boltffi_value_0 -> writer.writeString(__boltffi_value_0) })
        writer.writeOptionalValue(this.language, { writer, __boltffi_value_0 -> writer.writeString(__boltffi_value_0) })
        writer.writeOptionalValue(this.style, { writer, __boltffi_value_0 -> writer.writeString(__boltffi_value_0) })
    }

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridVoiceInfo {
            return XybridVoiceInfo(
                reader.readString(),
                reader.readString(),
                reader.readOptionalValue({ reader -> reader.readString() }),
                reader.readOptionalValue({ reader -> reader.readString() }),
                reader.readOptionalValue({ reader -> reader.readString() })
            )
        }

        internal fun fromByteArray(bytes: ByteArray): XybridVoiceInfo {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


sealed class XybridError : Exception() {
    internal abstract fun wireSize(): Int

    internal abstract fun writeTo(writer: WireWriter)

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }


    data class ModelNotFound(
        val id: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.id)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(0.toUInt())
            writer.writeString(this.id)
        }
    }
    data class DirectoryNotFound(
        val path: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.path)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(1.toUInt())
            writer.writeString(this.path)
        }
    }
    data class MetadataNotFound(
        val path: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.path)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(2.toUInt())
            writer.writeString(this.path)
        }
    }
    data class MetadataInvalid(
        override val message: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.message)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(3.toUInt())
            writer.writeString(this.message)
        }
    }
    data class LoadError(
        override val message: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.message)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(4.toUInt())
            writer.writeString(this.message)
        }
    }
    data class InferenceError(
        override val message: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.message)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(5.toUInt())
            writer.writeString(this.message)
        }
    }
    data class AbortedForCloudFallback(
        val reason: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.reason)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(6.toUInt())
            writer.writeString(this.reason)
        }
    }
    object StreamingNotSupported : XybridError() {
        internal override fun wireSize(): Int {
            return 4
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(7.toUInt())
        }
    }
    object NotLoaded : XybridError() {
        internal override fun wireSize(): Int {
            return 4
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(8.toUInt())
        }
    }
    data class ConfigError(
        override val message: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.message)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(9.toUInt())
            writer.writeString(this.message)
        }
    }
    data class NetworkError(
        override val message: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.message)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(10.toUInt())
            writer.writeString(this.message)
        }
    }
    data class Offline(
        override val message: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.message)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(11.toUInt())
            writer.writeString(this.message)
        }
    }
    data class IoError(
        override val message: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.message)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(12.toUInt())
            writer.writeString(this.message)
        }
    }
    data class CacheError(
        override val message: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.message)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(13.toUInt())
            writer.writeString(this.message)
        }
    }
    data class PipelineError(
        override val message: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.message)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(14.toUInt())
            writer.writeString(this.message)
        }
    }
    data class CircuitOpen(
        override val message: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.message)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(15.toUInt())
            writer.writeString(this.message)
        }
    }
    data class RateLimited(
        val retryAfterSecs: ULong
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 8
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(16.toUInt())
            writer.writeU64(this.retryAfterSecs)
        }
    }
    data class Timeout(
        val timeoutMs: ULong
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 8
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(17.toUInt())
            writer.writeU64(this.timeoutMs)
        }
    }
    data class MissingArtifact(
        override val message: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.message)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(18.toUInt())
            writer.writeString(this.message)
        }
    }
    data class UnsupportedModelCapability(
        override val message: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.message)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(19.toUInt())
            writer.writeString(this.message)
        }
    }
    data class UnsupportedBackendCapability(
        override val message: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.message)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(20.toUInt())
            writer.writeString(this.message)
        }
    }
    data class InvalidImage(
        override val message: String
    ) : XybridError() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.message)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(21.toUInt())
            writer.writeString(this.message)
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridError {
            val tag = reader.readU32()
            return when (tag) {
                0.toUInt() -> ModelNotFound(reader.readString())
                1.toUInt() -> DirectoryNotFound(reader.readString())
                2.toUInt() -> MetadataNotFound(reader.readString())
                3.toUInt() -> MetadataInvalid(reader.readString())
                4.toUInt() -> LoadError(reader.readString())
                5.toUInt() -> InferenceError(reader.readString())
                6.toUInt() -> AbortedForCloudFallback(reader.readString())
                7.toUInt() -> StreamingNotSupported
                8.toUInt() -> NotLoaded
                9.toUInt() -> ConfigError(reader.readString())
                10.toUInt() -> NetworkError(reader.readString())
                11.toUInt() -> Offline(reader.readString())
                12.toUInt() -> IoError(reader.readString())
                13.toUInt() -> CacheError(reader.readString())
                14.toUInt() -> PipelineError(reader.readString())
                15.toUInt() -> CircuitOpen(reader.readString())
                16.toUInt() -> RateLimited(reader.readU64())
                17.toUInt() -> Timeout(reader.readU64())
                18.toUInt() -> MissingArtifact(reader.readString())
                19.toUInt() -> UnsupportedModelCapability(reader.readString())
                20.toUInt() -> UnsupportedBackendCapability(reader.readString())
                21.toUInt() -> InvalidImage(reader.readString())
                else -> throw IllegalArgumentException("unknown XybridError tag: $tag")
            }
        }

        internal fun fromByteArray(bytes: ByteArray): XybridError {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


sealed class XybridEnvelopeKind {
    internal abstract fun wireSize(): Int

    internal abstract fun writeTo(writer: WireWriter)

    internal fun toByteArray(): ByteArray {
        val buffer = WireWriterPool.acquire(wireSize())
        val writer = buffer.writer
        try {
            writeTo(writer)
            return buffer.bytes()
        } finally {
            buffer.close()
        }
    }


    data class Text(
        val text: String
    ) : XybridEnvelopeKind() {
        internal override fun wireSize(): Int {
            return 4 + 4 + Utf8Codec.maxBytes(this.text)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(0.toUInt())
            writer.writeString(this.text)
        }
    }
    data class Audio(
        val bytes: ByteArray
    ) : XybridEnvelopeKind() {
        internal override fun wireSize(): Int {
            return 4 + 4 + this.bytes.size
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(1.toUInt())
            writer.writeBytes(this.bytes)
        }
    }
    data class Embedding(
        val values: FloatArray
    ) : XybridEnvelopeKind() {
        internal override fun wireSize(): Int {
            return 4 + 4 + this.values.size * 4
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(2.toUInt())
            writer.writeFloatArray(this.values)
        }
    }
    data class Image(
        val bytes: ByteArray,
        val format: String
    ) : XybridEnvelopeKind() {
        internal override fun wireSize(): Int {
            return 4 + 4 + this.bytes.size + 4 + Utf8Codec.maxBytes(this.format)
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(3.toUInt())
            writer.writeBytes(this.bytes)
            writer.writeString(this.format)
        }
    }
    data class MultiPart(
        val parts: List<ai.xybrid.XybridEnvelope>
    ) : XybridEnvelopeKind() {
        internal override fun wireSize(): Int {
            return 4 + 4 + this.parts.sumOf { __boltffi_value_0 -> (__boltffi_value_0.wireSize()).toInt() }
        }

        internal override fun writeTo(writer: WireWriter) {
            writer.writeU32(4.toUInt())
            writer.writeSequence(this.parts, this.parts.size, { writer, __boltffi_value_0 -> __boltffi_value_0.writeTo(writer) })
        }
    }

    companion object {
        internal fun fromReader(reader: WireReader): XybridEnvelopeKind {
            val tag = reader.readU32()
            return when (tag) {
                0.toUInt() -> Text(reader.readString())
                1.toUInt() -> Audio(reader.readBytes())
                2.toUInt() -> Embedding(reader.readFloatArray())
                3.toUInt() -> Image(reader.readBytes(), reader.readString())
                4.toUInt() -> MultiPart(reader.readSequence({ reader -> ai.xybrid.XybridEnvelope.fromReader(reader) }))
                else -> throw IllegalArgumentException("unknown XybridEnvelopeKind tag: $tag")
            }
        }

        internal fun fromByteArray(bytes: ByteArray): XybridEnvelopeKind {
            val reader = WireReader(bytes)
            return fromReader(reader)
        }
    }
}


enum class XybridMessageRole(val value: Int) {
    SYSTEM(0),
    USER(1),
    ASSISTANT(2);

    companion object {
        fun fromValue(value: Int): XybridMessageRole =
            entries.first { it.value == value }
    }
}


enum class XybridAbortSignal(val value: Int) {
    MEMORY_PRESSURE_WARN(0),
    MEMORY_PRESSURE_CRITICAL(1),
    THERMAL_HOT(2),
    THERMAL_CRITICAL(3);

    companion object {
        fun fromValue(value: Int): XybridAbortSignal =
            entries.first { it.value == value }
    }
}


enum class XybridOutputType(val value: Int) {
    TEXT(0),
    AUDIO(1),
    EMBEDDING(2),
    UNKNOWN(3);

    companion object {
        fun fromValue(value: Int): XybridOutputType =
            entries.first { it.value == value }
    }
}


enum class XybridExecutionTarget(val value: Int) {
    LOCAL(0),
    CLOUD(1);

    companion object {
        fun fromValue(value: Int): XybridExecutionTarget =
            entries.first { it.value == value }
    }
}


enum class XybridDownloadState(val value: Int) {
    DOWNLOADING(0),
    READY(1),
    FAILED(2);

    companion object {
        fun fromValue(value: Int): XybridDownloadState =
            entries.first { it.value == value }
    }
}


enum class XybridStreamEventKind(val value: Int) {
    TOKEN(0),
    COMPLETE(1);

    companion object {
        fun fromValue(value: Int): XybridStreamEventKind =
            entries.first { it.value == value }
    }
}


enum class XybridThermalState(val value: Int) {
    NORMAL(0),
    WARM(1),
    HOT(2),
    CRITICAL(3);

    companion object {
        fun fromValue(value: Int): XybridThermalState =
            entries.first { it.value == value }
    }
}

class XybridModel internal constructor(internal val handle: Long) : AutoCloseable {
    private val __boltffi_closed = java.util.concurrent.atomic.AtomicBoolean(false)

    override fun close() {
        if (__boltffi_closed.compareAndSet(false, true)) {
            Native.boltffi_release_class_xybrid_bolt_xybrid_model(handle)
        }
    }

    internal fun boltffiHandle(): Long {
        check(!__boltffi_closed.get()) { "XybridModel is closed" }
        return handle
    }

    constructor(id: String) : this(fromRegistry(id).handle)

    constructor(repo: String, revision: String) : this(fromHuggingfaceWithRevision(repo, revision).handle)

    companion object {
        fun fromRegistry(id: String): XybridModel {
            val __boltffi_id_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(id))
            val __boltffi_id_writer = __boltffi_id_wire.writer
            __boltffi_id_writer.writeString(id)
            try {
                return XybridModel(try { Native.boltffi_init_class_xybrid_bolt_xybrid_model_from_registry(__boltffi_id_wire.directBuffer(), __boltffi_id_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } })
            } finally {
                __boltffi_id_wire.close()
            }
        }
        fun fromRegistrySpeculative(id: String): XybridModel {
            val __boltffi_id_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(id))
            val __boltffi_id_writer = __boltffi_id_wire.writer
            __boltffi_id_writer.writeString(id)
            try {
                return XybridModel(try { Native.boltffi_init_class_xybrid_bolt_xybrid_model_from_registry_speculative(__boltffi_id_wire.directBuffer(), __boltffi_id_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } })
            } finally {
                __boltffi_id_wire.close()
            }
        }
        fun fromDirectory(path: String): XybridModel {
            val __boltffi_path_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(path))
            val __boltffi_path_writer = __boltffi_path_wire.writer
            __boltffi_path_writer.writeString(path)
            try {
                return XybridModel(try { Native.boltffi_init_class_xybrid_bolt_xybrid_model_from_directory(__boltffi_path_wire.directBuffer(), __boltffi_path_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } })
            } finally {
                __boltffi_path_wire.close()
            }
        }
        fun fromBundle(path: String): XybridModel {
            val __boltffi_path_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(path))
            val __boltffi_path_writer = __boltffi_path_wire.writer
            __boltffi_path_writer.writeString(path)
            try {
                return XybridModel(try { Native.boltffi_init_class_xybrid_bolt_xybrid_model_from_bundle(__boltffi_path_wire.directBuffer(), __boltffi_path_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } })
            } finally {
                __boltffi_path_wire.close()
            }
        }
        fun fromHuggingface(repo: String): XybridModel {
            val __boltffi_repo_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(repo))
            val __boltffi_repo_writer = __boltffi_repo_wire.writer
            __boltffi_repo_writer.writeString(repo)
            try {
                return XybridModel(try { Native.boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface(__boltffi_repo_wire.directBuffer(), __boltffi_repo_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } })
            } finally {
                __boltffi_repo_wire.close()
            }
        }
        fun fromHuggingfaceWithRevision(repo: String, revision: String): XybridModel {
            val __boltffi_repo_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(repo))
            val __boltffi_repo_writer = __boltffi_repo_wire.writer
            __boltffi_repo_writer.writeString(repo)
            val __boltffi_revision_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(revision))
            val __boltffi_revision_writer = __boltffi_revision_wire.writer
            __boltffi_revision_writer.writeString(revision)
            try {
                return XybridModel(try { Native.boltffi_init_class_xybrid_bolt_xybrid_model_from_huggingface_with_revision(__boltffi_repo_wire.directBuffer(), __boltffi_repo_wire.size(), __boltffi_revision_wire.directBuffer(), __boltffi_revision_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } })
            } finally {
                __boltffi_repo_wire.close()
                __boltffi_revision_wire.close()
            }
        }
        fun fromModelFile(path: String): XybridModel {
            val __boltffi_path_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(path))
            val __boltffi_path_writer = __boltffi_path_wire.writer
            __boltffi_path_writer.writeString(path)
            try {
                return XybridModel(try { Native.boltffi_init_class_xybrid_bolt_xybrid_model_from_model_file(__boltffi_path_wire.directBuffer(), __boltffi_path_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } })
            } finally {
                __boltffi_path_wire.close()
            }
        }
    }

    fun modelId(): String {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_model_model_id(this.boltffiHandle()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readString()
    }

    fun version(): String {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_model_version(this.boltffiHandle()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readString()
    }

    fun outputType(): XybridOutputType {
        return XybridOutputType.fromValue(Native.boltffi_method_class_xybrid_bolt_xybrid_model_output_type(this.boltffiHandle()))
    }

    fun isLoaded(): Boolean {
        return Native.boltffi_method_class_xybrid_bolt_xybrid_model_is_loaded(this.boltffiHandle())
    }

    fun isCloudServing(): Boolean {
        return Native.boltffi_method_class_xybrid_bolt_xybrid_model_is_cloud_serving(this.boltffiHandle())
    }

    fun downloadStatus(): XybridDownloadStatus {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_model_download_status(this.boltffiHandle()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return XybridDownloadStatus.fromReader(__boltffi_reader)
    }

    fun awaitDownload(timeoutMs: ULong): XybridDownloadStatus {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_model_await_download(this.boltffiHandle(), timeoutMs.toLong()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return XybridDownloadStatus.fromReader(__boltffi_reader)
    }

    fun supportsStreaming(): Boolean {
        return Native.boltffi_method_class_xybrid_bolt_xybrid_model_supports_streaming(this.boltffiHandle())
    }

    fun supportsTokenStreaming(): Boolean {
        return Native.boltffi_method_class_xybrid_bolt_xybrid_model_supports_token_streaming(this.boltffiHandle())
    }

    fun defaultGenerationConfig(): XybridGenerationConfig {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_model_default_generation_config(this.boltffiHandle()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return XybridGenerationConfig.fromReader(__boltffi_reader)
    }

    fun isLlm(): Boolean {
        return Native.boltffi_method_class_xybrid_bolt_xybrid_model_is_llm(this.boltffiHandle())
    }

    fun supportsToolCalling(): Boolean? {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_model_supports_tool_calling(this.boltffiHandle()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readOptionalBool()
    }

    fun hasVoices(): Boolean {
        return Native.boltffi_method_class_xybrid_bolt_xybrid_model_has_voices(this.boltffiHandle())
    }

    fun voices(): List<XybridVoiceInfo> {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_model_voices(this.boltffiHandle()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readSequence({ __boltffi_reader -> XybridVoiceInfo.fromReader(__boltffi_reader) })
    }

    fun defaultVoice(): XybridVoiceInfo? {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_model_default_voice(this.boltffiHandle()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readOptionalValue({ __boltffi_reader -> XybridVoiceInfo.fromReader(__boltffi_reader) })
    }

    fun voice(voiceId: String): XybridVoiceInfo? {
        val __boltffi_voiceId_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(voiceId))
        val __boltffi_voiceId_writer = __boltffi_voiceId_wire.writer
        __boltffi_voiceId_writer.writeString(voiceId)
        try {
            val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_model_voice(this.boltffiHandle(), __boltffi_voiceId_wire.directBuffer(), __boltffi_voiceId_wire.size()) ?: throw IllegalStateException("null buffer returned")
            val __boltffi_reader = WireReader(__boltffi_result)
            return __boltffi_reader.readOptionalValue({ __boltffi_reader -> XybridVoiceInfo.fromReader(__boltffi_reader) })
        } finally {
            __boltffi_voiceId_wire.close()
        }
    }

    fun run(envelope: XybridEnvelope, options: XybridRunOptions?): XybridResult {
        val __boltffi_envelope_wire = WireWriterPool.acquire(envelope.wireSize())
        val __boltffi_envelope_writer = __boltffi_envelope_wire.writer
        envelope.writeTo(__boltffi_envelope_writer)
        val __boltffi_options_wire = WireWriterPool.acquire(1 + (options?.let { __boltffi_value_0 -> __boltffi_value_0.wireSize() } ?: 0))
        val __boltffi_options_writer = __boltffi_options_wire.writer
        __boltffi_options_writer.writeOptionalValue(options, { __boltffi_options_writer, __boltffi_value_0 -> __boltffi_value_0.writeTo(__boltffi_options_writer) })
        try {
            val __boltffi_result = try { Native.boltffi_method_class_xybrid_bolt_xybrid_model_run(this.boltffiHandle(), __boltffi_envelope_wire.directBuffer(), __boltffi_envelope_wire.size(), __boltffi_options_wire.directBuffer(), __boltffi_options_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } } ?: throw IllegalStateException("null buffer returned")
            val __boltffi_reader = WireReader(__boltffi_result)
            return XybridResult.fromReader(__boltffi_reader)
        } finally {
            __boltffi_envelope_wire.close()
            __boltffi_options_wire.close()
        }
    }

    fun runStream(envelope: XybridEnvelope, options: XybridRunOptions?): ULong {
        val __boltffi_envelope_wire = WireWriterPool.acquire(envelope.wireSize())
        val __boltffi_envelope_writer = __boltffi_envelope_wire.writer
        envelope.writeTo(__boltffi_envelope_writer)
        val __boltffi_options_wire = WireWriterPool.acquire(1 + (options?.let { __boltffi_value_0 -> __boltffi_value_0.wireSize() } ?: 0))
        val __boltffi_options_writer = __boltffi_options_wire.writer
        __boltffi_options_writer.writeOptionalValue(options, { __boltffi_options_writer, __boltffi_value_0 -> __boltffi_value_0.writeTo(__boltffi_options_writer) })
        try {
            return try { Native.boltffi_method_class_xybrid_bolt_xybrid_model_run_stream(this.boltffiHandle(), __boltffi_envelope_wire.directBuffer(), __boltffi_envelope_wire.size(), __boltffi_options_wire.directBuffer(), __boltffi_options_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } }.toULong()
        } finally {
            __boltffi_envelope_wire.close()
            __boltffi_options_wire.close()
        }
    }

    fun streamNext(streamId: ULong): XybridStreamEvent {
        val __boltffi_result = try { Native.boltffi_method_class_xybrid_bolt_xybrid_model_stream_next(this.boltffiHandle(), streamId.toLong()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } } ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return XybridStreamEvent.fromReader(__boltffi_reader)
    }

    fun streamResult(streamId: ULong): XybridResult {
        val __boltffi_result = try { Native.boltffi_method_class_xybrid_bolt_xybrid_model_stream_result(this.boltffiHandle(), streamId.toLong()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } } ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return XybridResult.fromReader(__boltffi_reader)
    }

    fun streamClose(streamId: ULong) {
        Native.boltffi_method_class_xybrid_bolt_xybrid_model_stream_close(this.boltffiHandle(), streamId.toLong())
    }

    fun runWithContext(envelope: XybridEnvelope, context: XybridConversationContext, options: XybridRunOptions?): XybridResult {
        val __boltffi_envelope_wire = WireWriterPool.acquire(envelope.wireSize())
        val __boltffi_envelope_writer = __boltffi_envelope_wire.writer
        envelope.writeTo(__boltffi_envelope_writer)
        val __boltffi_options_wire = WireWriterPool.acquire(1 + (options?.let { __boltffi_value_0 -> __boltffi_value_0.wireSize() } ?: 0))
        val __boltffi_options_writer = __boltffi_options_wire.writer
        __boltffi_options_writer.writeOptionalValue(options, { __boltffi_options_writer, __boltffi_value_0 -> __boltffi_value_0.writeTo(__boltffi_options_writer) })
        try {
            val __boltffi_result = try { Native.boltffi_method_class_xybrid_bolt_xybrid_model_run_with_context(this.boltffiHandle(), __boltffi_envelope_wire.directBuffer(), __boltffi_envelope_wire.size(), context.boltffiHandle(), __boltffi_options_wire.directBuffer(), __boltffi_options_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } } ?: throw IllegalStateException("null buffer returned")
            val __boltffi_reader = WireReader(__boltffi_result)
            return XybridResult.fromReader(__boltffi_reader)
        } finally {
            __boltffi_envelope_wire.close()
            __boltffi_options_wire.close()
        }
    }

    fun runStreamWithContext(envelope: XybridEnvelope, context: XybridConversationContext, options: XybridRunOptions?): ULong {
        val __boltffi_envelope_wire = WireWriterPool.acquire(envelope.wireSize())
        val __boltffi_envelope_writer = __boltffi_envelope_wire.writer
        envelope.writeTo(__boltffi_envelope_writer)
        val __boltffi_options_wire = WireWriterPool.acquire(1 + (options?.let { __boltffi_value_0 -> __boltffi_value_0.wireSize() } ?: 0))
        val __boltffi_options_writer = __boltffi_options_wire.writer
        __boltffi_options_writer.writeOptionalValue(options, { __boltffi_options_writer, __boltffi_value_0 -> __boltffi_value_0.writeTo(__boltffi_options_writer) })
        try {
            return try { Native.boltffi_method_class_xybrid_bolt_xybrid_model_run_stream_with_context(this.boltffiHandle(), __boltffi_envelope_wire.directBuffer(), __boltffi_envelope_wire.size(), context.boltffiHandle(), __boltffi_options_wire.directBuffer(), __boltffi_options_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } }.toULong()
        } finally {
            __boltffi_envelope_wire.close()
            __boltffi_options_wire.close()
        }
    }

    fun warmup() {
        try { Native.boltffi_method_class_xybrid_bolt_xybrid_model_warmup(this.boltffiHandle()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } }
    }

    fun unload() {
        try { Native.boltffi_method_class_xybrid_bolt_xybrid_model_unload(this.boltffiHandle()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } }
    }
}

class XybridConversationContext internal constructor(internal val handle: Long) : AutoCloseable {
    private val __boltffi_closed = java.util.concurrent.atomic.AtomicBoolean(false)

    override fun close() {
        if (__boltffi_closed.compareAndSet(false, true)) {
            Native.boltffi_release_class_xybrid_bolt_xybrid_conversation_context(handle)
        }
    }

    internal fun boltffiHandle(): Long {
        check(!__boltffi_closed.get()) { "XybridConversationContext is closed" }
        return handle
    }

    constructor() : this(new().handle)

    constructor(id: String) : this(withId(id).handle)

    companion object {
        fun new(): XybridConversationContext {
            return XybridConversationContext(Native.boltffi_init_class_xybrid_bolt_xybrid_conversation_context_new())
        }
        fun withId(id: String): XybridConversationContext {
            val __boltffi_id_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(id))
            val __boltffi_id_writer = __boltffi_id_wire.writer
            __boltffi_id_writer.writeString(id)
            try {
                return XybridConversationContext(Native.boltffi_init_class_xybrid_bolt_xybrid_conversation_context_with_id(__boltffi_id_wire.directBuffer(), __boltffi_id_wire.size()))
            } finally {
                __boltffi_id_wire.close()
            }
        }
    }

    fun push(envelope: XybridEnvelope) {
        val __boltffi_envelope_wire = WireWriterPool.acquire(envelope.wireSize())
        val __boltffi_envelope_writer = __boltffi_envelope_wire.writer
        envelope.writeTo(__boltffi_envelope_writer)
        try {
            try { Native.boltffi_method_class_xybrid_bolt_xybrid_conversation_context_push(this.boltffiHandle(), __boltffi_envelope_wire.directBuffer(), __boltffi_envelope_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } }
        } finally {
            __boltffi_envelope_wire.close()
        }
    }

    fun setSystem(envelope: XybridEnvelope) {
        val __boltffi_envelope_wire = WireWriterPool.acquire(envelope.wireSize())
        val __boltffi_envelope_writer = __boltffi_envelope_wire.writer
        envelope.writeTo(__boltffi_envelope_writer)
        try {
            try { Native.boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_system(this.boltffiHandle(), __boltffi_envelope_wire.directBuffer(), __boltffi_envelope_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } }
        } finally {
            __boltffi_envelope_wire.close()
        }
    }

    fun clear() {
        Native.boltffi_method_class_xybrid_bolt_xybrid_conversation_context_clear(this.boltffiHandle())
    }

    fun id(): String {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_conversation_context_id(this.boltffiHandle()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readString()
    }

    fun historyLen(): UInt {
        return Native.boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history_len(this.boltffiHandle()).toUInt()
    }

    fun history(): List<XybridEnvelope> {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_conversation_context_history(this.boltffiHandle()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readSequence({ __boltffi_reader -> XybridEnvelope.fromReader(__boltffi_reader) })
    }

    fun hasSystem(): Boolean {
        return Native.boltffi_method_class_xybrid_bolt_xybrid_conversation_context_has_system(this.boltffiHandle())
    }

    fun setMaxHistoryLen(len: UInt) {
        Native.boltffi_method_class_xybrid_bolt_xybrid_conversation_context_set_max_history_len(this.boltffiHandle(), len.toInt())
    }
}

class XybridTelemetryConfig internal constructor(internal val handle: Long) : AutoCloseable {
    private val __boltffi_closed = java.util.concurrent.atomic.AtomicBoolean(false)

    override fun close() {
        if (__boltffi_closed.compareAndSet(false, true)) {
            Native.boltffi_release_class_xybrid_bolt_xybrid_telemetry_config(handle)
        }
    }

    internal fun boltffiHandle(): Long {
        check(!__boltffi_closed.get()) { "XybridTelemetryConfig is closed" }
        return handle
    }

    constructor(apiKey: String) : this(new(apiKey).handle)

    companion object {
        fun new(apiKey: String): XybridTelemetryConfig {
            val __boltffi_apiKey_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(apiKey))
            val __boltffi_apiKey_writer = __boltffi_apiKey_wire.writer
            __boltffi_apiKey_writer.writeString(apiKey)
            try {
                return XybridTelemetryConfig(Native.boltffi_init_class_xybrid_bolt_xybrid_telemetry_config_new(__boltffi_apiKey_wire.directBuffer(), __boltffi_apiKey_wire.size()))
            } finally {
                __boltffi_apiKey_wire.close()
            }
        }
    }

    fun setEndpoint(endpoint: String) {
        val __boltffi_endpoint_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(endpoint))
        val __boltffi_endpoint_writer = __boltffi_endpoint_wire.writer
        __boltffi_endpoint_writer.writeString(endpoint)
        try {
            Native.boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_endpoint(this.boltffiHandle(), __boltffi_endpoint_wire.directBuffer(), __boltffi_endpoint_wire.size())
        } finally {
            __boltffi_endpoint_wire.close()
        }
    }

    fun setAppVersion(version: String) {
        val __boltffi_version_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(version))
        val __boltffi_version_writer = __boltffi_version_wire.writer
        __boltffi_version_writer.writeString(version)
        try {
            Native.boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_app_version(this.boltffiHandle(), __boltffi_version_wire.directBuffer(), __boltffi_version_wire.size())
        } finally {
            __boltffi_version_wire.close()
        }
    }

    fun setDeviceLabel(label: String) {
        val __boltffi_label_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(label))
        val __boltffi_label_writer = __boltffi_label_wire.writer
        __boltffi_label_writer.writeString(label)
        try {
            Native.boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_label(this.boltffiHandle(), __boltffi_label_wire.directBuffer(), __boltffi_label_wire.size())
        } finally {
            __boltffi_label_wire.close()
        }
    }

    fun setDeviceAttribute(key: String, value: String) {
        val __boltffi_key_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(key))
        val __boltffi_key_writer = __boltffi_key_wire.writer
        __boltffi_key_writer.writeString(key)
        val __boltffi_value_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(value))
        val __boltffi_value_writer = __boltffi_value_wire.writer
        __boltffi_value_writer.writeString(value)
        try {
            Native.boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_device_attribute(this.boltffiHandle(), __boltffi_key_wire.directBuffer(), __boltffi_key_wire.size(), __boltffi_value_wire.directBuffer(), __boltffi_value_wire.size())
        } finally {
            __boltffi_key_wire.close()
            __boltffi_value_wire.close()
        }
    }

    fun setBatchSize(batchSize: UInt) {
        Native.boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_batch_size(this.boltffiHandle(), batchSize.toInt())
    }

    fun setFlushIntervalSecs(secs: UInt) {
        Native.boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_set_flush_interval_secs(this.boltffiHandle(), secs.toInt())
    }

    fun `init`() {
        try { Native.boltffi_method_class_xybrid_bolt_xybrid_telemetry_config_init(this.boltffiHandle()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } }
    }
}

class XybridBundle internal constructor(internal val handle: Long) : AutoCloseable {
    private val __boltffi_closed = java.util.concurrent.atomic.AtomicBoolean(false)

    override fun close() {
        if (__boltffi_closed.compareAndSet(false, true)) {
            Native.boltffi_release_class_xybrid_bolt_xybrid_bundle(handle)
        }
    }

    internal fun boltffiHandle(): Long {
        check(!__boltffi_closed.get()) { "XybridBundle is closed" }
        return handle
    }

    constructor(path: String) : this(`open`(path).handle)

    companion object {
        fun `open`(path: String): XybridBundle {
            val __boltffi_path_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(path))
            val __boltffi_path_writer = __boltffi_path_wire.writer
            __boltffi_path_writer.writeString(path)
            try {
                return XybridBundle(try { Native.boltffi_init_class_xybrid_bolt_xybrid_bundle_open(__boltffi_path_wire.directBuffer(), __boltffi_path_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } })
            } finally {
                __boltffi_path_wire.close()
            }
        }
    }

    fun modelId(): String {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_bundle_model_id(this.boltffiHandle()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readString()
    }

    fun version(): String {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_bundle_version(this.boltffiHandle()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readString()
    }

    fun target(): String {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_bundle_target(this.boltffiHandle()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readString()
    }

    fun hash(): String {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_bundle_hash(this.boltffiHandle()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readString()
    }

    fun hasMetadata(): Boolean {
        return Native.boltffi_method_class_xybrid_bolt_xybrid_bundle_has_metadata(this.boltffiHandle())
    }

    fun fileCount(): UInt {
        return Native.boltffi_method_class_xybrid_bolt_xybrid_bundle_file_count(this.boltffiHandle()).toUInt()
    }

    fun fileName(index: UInt): String? {
        val __boltffi_result = Native.boltffi_method_class_xybrid_bolt_xybrid_bundle_file_name(this.boltffiHandle(), index.toInt()) ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readOptionalValue({ __boltffi_reader -> __boltffi_reader.readString() })
    }

    fun manifestJson(): String {
        val __boltffi_result = try { Native.boltffi_method_class_xybrid_bolt_xybrid_bundle_manifest_json(this.boltffiHandle()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } } ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readString()
    }

    fun metadataJson(): String? {
        val __boltffi_result = try { Native.boltffi_method_class_xybrid_bolt_xybrid_bundle_metadata_json(this.boltffiHandle()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } } ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readOptionalValue({ __boltffi_reader -> __boltffi_reader.readString() })
    }

    fun extract(outputDir: String) {
        val __boltffi_outputDir_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(outputDir))
        val __boltffi_outputDir_writer = __boltffi_outputDir_wire.writer
        __boltffi_outputDir_writer.writeString(outputDir)
        try {
            try { Native.boltffi_method_class_xybrid_bolt_xybrid_bundle_extract(this.boltffiHandle(), __boltffi_outputDir_wire.directBuffer(), __boltffi_outputDir_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } }
        } finally {
            __boltffi_outputDir_wire.close()
        }
    }
}

fun toolResultsEnvelope(userText: String, priorAssistantText: String, results: List<XybridToolResult>): XybridEnvelope {
    val __boltffi_userText_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(userText))
    val __boltffi_userText_writer = __boltffi_userText_wire.writer
    __boltffi_userText_writer.writeString(userText)
    val __boltffi_priorAssistantText_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(priorAssistantText))
    val __boltffi_priorAssistantText_writer = __boltffi_priorAssistantText_wire.writer
    __boltffi_priorAssistantText_writer.writeString(priorAssistantText)
    val __boltffi_results_wire = WireWriterPool.acquire(4 + results.sumOf { __boltffi_value_0 -> (__boltffi_value_0.wireSize()).toInt() })
    val __boltffi_results_writer = __boltffi_results_wire.writer
    __boltffi_results_writer.writeSequence(results, results.size, { __boltffi_results_writer, __boltffi_value_0 -> __boltffi_value_0.writeTo(__boltffi_results_writer) })
    try {
        val __boltffi_result = try { Native.boltffi_function_xybrid_bolt_tool_results_envelope(__boltffi_userText_wire.directBuffer(), __boltffi_userText_wire.size(), __boltffi_priorAssistantText_wire.directBuffer(), __boltffi_priorAssistantText_wire.size(), __boltffi_results_wire.directBuffer(), __boltffi_results_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } } ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return XybridEnvelope.fromReader(__boltffi_reader)
    } finally {
        __boltffi_userText_wire.close()
        __boltffi_priorAssistantText_wire.close()
        __boltffi_results_wire.close()
    }
}

fun jsonSchemaToGbnf(schemaJson: String): String {
    val __boltffi_schemaJson_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(schemaJson))
    val __boltffi_schemaJson_writer = __boltffi_schemaJson_wire.writer
    __boltffi_schemaJson_writer.writeString(schemaJson)
    try {
        val __boltffi_result = try { Native.boltffi_function_xybrid_bolt_json_schema_to_gbnf(__boltffi_schemaJson_wire.directBuffer(), __boltffi_schemaJson_wire.size()) } catch (__boltffi_error: BoltFfiErrorBufferException) { run { val __boltffi_error_reader = WireReader(__boltffi_error.bytes); throw XybridError.fromReader(__boltffi_error_reader) } } ?: throw IllegalStateException("null buffer returned")
        val __boltffi_reader = WireReader(__boltffi_result)
        return __boltffi_reader.readString()
    } finally {
        __boltffi_schemaJson_wire.close()
    }
}

fun setThermalState(state: XybridThermalState) {
    Native.boltffi_function_xybrid_bolt_set_thermal_state(state.value)
}

fun clearThermalState() {
    Native.boltffi_function_xybrid_bolt_clear_thermal_state()
}

fun setBatteryLevel(percent: UByte) {
    Native.boltffi_function_xybrid_bolt_set_battery_level(percent.toByte())
}

fun clearBatteryLevel() {
    Native.boltffi_function_xybrid_bolt_clear_battery_level()
}

fun configureRuntime(apiKey: String?, gatewayUrl: String?, ingestUrl: String?) {
    val __boltffi_apiKey_wire = WireWriterPool.acquire(1 + (apiKey?.let { __boltffi_value_0 -> 4 + Utf8Codec.maxBytes(__boltffi_value_0) } ?: 0))
    val __boltffi_apiKey_writer = __boltffi_apiKey_wire.writer
    __boltffi_apiKey_writer.writeOptionalValue(apiKey, { __boltffi_apiKey_writer, __boltffi_value_0 -> __boltffi_apiKey_writer.writeString(__boltffi_value_0) })
    val __boltffi_gatewayUrl_wire = WireWriterPool.acquire(1 + (gatewayUrl?.let { __boltffi_value_0 -> 4 + Utf8Codec.maxBytes(__boltffi_value_0) } ?: 0))
    val __boltffi_gatewayUrl_writer = __boltffi_gatewayUrl_wire.writer
    __boltffi_gatewayUrl_writer.writeOptionalValue(gatewayUrl, { __boltffi_gatewayUrl_writer, __boltffi_value_0 -> __boltffi_gatewayUrl_writer.writeString(__boltffi_value_0) })
    val __boltffi_ingestUrl_wire = WireWriterPool.acquire(1 + (ingestUrl?.let { __boltffi_value_0 -> 4 + Utf8Codec.maxBytes(__boltffi_value_0) } ?: 0))
    val __boltffi_ingestUrl_writer = __boltffi_ingestUrl_wire.writer
    __boltffi_ingestUrl_writer.writeOptionalValue(ingestUrl, { __boltffi_ingestUrl_writer, __boltffi_value_0 -> __boltffi_ingestUrl_writer.writeString(__boltffi_value_0) })
    try {
        Native.boltffi_function_xybrid_bolt_configure_runtime(__boltffi_apiKey_wire.directBuffer(), __boltffi_apiKey_wire.size(), __boltffi_gatewayUrl_wire.directBuffer(), __boltffi_gatewayUrl_wire.size(), __boltffi_ingestUrl_wire.directBuffer(), __boltffi_ingestUrl_wire.size())
    } finally {
        __boltffi_apiKey_wire.close()
        __boltffi_gatewayUrl_wire.close()
        __boltffi_ingestUrl_wire.close()
    }
}

fun initSdkCacheDir(cacheDir: String) {
    val __boltffi_cacheDir_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(cacheDir))
    val __boltffi_cacheDir_writer = __boltffi_cacheDir_wire.writer
    __boltffi_cacheDir_writer.writeString(cacheDir)
    try {
        Native.boltffi_function_xybrid_bolt_init_sdk_cache_dir(__boltffi_cacheDir_wire.directBuffer(), __boltffi_cacheDir_wire.size())
    } finally {
        __boltffi_cacheDir_wire.close()
    }
}

fun setBinding(binding: String) {
    val __boltffi_binding_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(binding))
    val __boltffi_binding_writer = __boltffi_binding_wire.writer
    __boltffi_binding_writer.writeString(binding)
    try {
        Native.boltffi_function_xybrid_bolt_set_binding(__boltffi_binding_wire.directBuffer(), __boltffi_binding_wire.size())
    } finally {
        __boltffi_binding_wire.close()
    }
}

fun setApiKey(apiKey: String) {
    val __boltffi_apiKey_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(apiKey))
    val __boltffi_apiKey_writer = __boltffi_apiKey_wire.writer
    __boltffi_apiKey_writer.writeString(apiKey)
    try {
        Native.boltffi_function_xybrid_bolt_set_api_key(__boltffi_apiKey_wire.directBuffer(), __boltffi_apiKey_wire.size())
    } finally {
        __boltffi_apiKey_wire.close()
    }
}

fun setProviderApiKey(provider: String, apiKey: String) {
    val __boltffi_provider_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(provider))
    val __boltffi_provider_writer = __boltffi_provider_wire.writer
    __boltffi_provider_writer.writeString(provider)
    val __boltffi_apiKey_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(apiKey))
    val __boltffi_apiKey_writer = __boltffi_apiKey_wire.writer
    __boltffi_apiKey_writer.writeString(apiKey)
    try {
        Native.boltffi_function_xybrid_bolt_set_provider_api_key(__boltffi_provider_wire.directBuffer(), __boltffi_provider_wire.size(), __boltffi_apiKey_wire.directBuffer(), __boltffi_apiKey_wire.size())
    } finally {
        __boltffi_provider_wire.close()
        __boltffi_apiKey_wire.close()
    }
}

fun setPlatformUrl(url: String) {
    val __boltffi_url_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(url))
    val __boltffi_url_writer = __boltffi_url_wire.writer
    __boltffi_url_writer.writeString(url)
    try {
        Native.boltffi_function_xybrid_bolt_set_platform_url(__boltffi_url_wire.directBuffer(), __boltffi_url_wire.size())
    } finally {
        __boltffi_url_wire.close()
    }
}

fun setSpeculativeCloud(enabled: Boolean) {
    Native.boltffi_function_xybrid_bolt_set_speculative_cloud(enabled)
}

fun hasApiKey(): Boolean {
    return Native.boltffi_function_xybrid_bolt_has_api_key()
}

fun isSpeculativeCloudEnabled(): Boolean {
    return Native.boltffi_function_xybrid_bolt_is_speculative_cloud_enabled()
}

fun willSpeculateForModel(modelId: String): Boolean {
    val __boltffi_modelId_wire = WireWriterPool.acquire(4 + Utf8Codec.maxBytes(modelId))
    val __boltffi_modelId_writer = __boltffi_modelId_wire.writer
    __boltffi_modelId_writer.writeString(modelId)
    try {
        return Native.boltffi_function_xybrid_bolt_will_speculate_for_model(__boltffi_modelId_wire.directBuffer(), __boltffi_modelId_wire.size())
    } finally {
        __boltffi_modelId_wire.close()
    }
}

fun version(): String {
    val __boltffi_result = Native.boltffi_function_xybrid_bolt_version() ?: throw IllegalStateException("null buffer returned")
    val __boltffi_reader = WireReader(__boltffi_result)
    return __boltffi_reader.readString()
}

fun releaseMemory(): UInt {
    return Native.boltffi_function_xybrid_bolt_release_memory().toUInt()
}

fun setAutoRelease(enabled: Boolean) {
    Native.boltffi_function_xybrid_bolt_set_auto_release(enabled)
}

fun isAutoReleaseEnabled(): Boolean {
    return Native.boltffi_function_xybrid_bolt_is_auto_release_enabled()
}

fun telemetryDefaultEndpoint(): String {
    val __boltffi_result = Native.boltffi_function_xybrid_bolt_telemetry_default_endpoint() ?: throw IllegalStateException("null buffer returned")
    val __boltffi_reader = WireReader(__boltffi_result)
    return __boltffi_reader.readString()
}

fun telemetryFlush() {
    Native.boltffi_function_xybrid_bolt_telemetry_flush()
}

fun telemetryShutdown() {
    Native.boltffi_function_xybrid_bolt_telemetry_shutdown()
}
