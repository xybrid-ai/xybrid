package com.facebook.react.bridge

// Recording doubles for the small bridge surface used by the encoder. These
// live outside Android's source sets and do not claim to exercise RN/JNI.
class WritableMap {
  val values = linkedMapOf<String, Any?>()

  fun putDouble(key: String, value: Double) { values[key] = value }
  fun putString(key: String, value: String?) { values[key] = value }
  fun putArray(key: String, value: WritableArray?) { values[key] = value }
}

class WritableArray {
  val values = mutableListOf<WritableMap?>()

  fun pushMap(value: WritableMap?) { values.add(value) }
}

object Arguments {
  fun createMap(): WritableMap = WritableMap()
  fun createArray(): WritableArray = WritableArray()
}
