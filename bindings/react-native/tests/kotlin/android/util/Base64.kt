package android.util

// JVM stand-in for the two android.util.Base64 calls the codec makes. Only
// compiled into the codec test, never into the module.
object Base64 {
  const val DEFAULT = 0
  const val NO_WRAP = 2

  @JvmStatic
  fun decode(str: String, flags: Int): ByteArray = java.util.Base64.getMimeDecoder().decode(str)

  @JvmStatic
  fun encodeToString(input: ByteArray, flags: Int): String = java.util.Base64.getEncoder().encodeToString(input)
}
