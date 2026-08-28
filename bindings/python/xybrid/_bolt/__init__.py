from __future__ import annotations


from dataclasses import dataclass



from enum import IntEnum



from collections.abc import Sequence




import struct


import sys
import uuid
from pathlib import Path

from . import _native


def _shared_library_filename() -> str:
    if sys.platform == "win32":
        return "xybrid_bolt.dll"
    if sys.platform == "darwin":
        return "libxybrid_bolt.dylib"
    return "libxybrid_bolt.so"


_native._initialize_loader(str(Path(__file__).resolve().with_name(_shared_library_filename())))



_BOLTFFI_STRUCT_I8 = struct.Struct("<b")
_BOLTFFI_STRUCT_U8 = struct.Struct("<B")
_BOLTFFI_STRUCT_I16 = struct.Struct("<h")
_BOLTFFI_STRUCT_U16 = struct.Struct("<H")
_BOLTFFI_STRUCT_I32 = struct.Struct("<i")
_BOLTFFI_STRUCT_U32 = struct.Struct("<I")
_BOLTFFI_STRUCT_I64 = struct.Struct("<q")
_BOLTFFI_STRUCT_U64 = struct.Struct("<Q")
_BOLTFFI_STRUCT_F32 = struct.Struct("<f")
_BOLTFFI_STRUCT_F64 = struct.Struct("<d")

_BOLTFFI_UNPACK_I8 = _BOLTFFI_STRUCT_I8.unpack_from
_BOLTFFI_UNPACK_I16 = _BOLTFFI_STRUCT_I16.unpack_from
_BOLTFFI_UNPACK_U16 = _BOLTFFI_STRUCT_U16.unpack_from
_BOLTFFI_UNPACK_I32 = _BOLTFFI_STRUCT_I32.unpack_from
_BOLTFFI_UNPACK_U32 = _BOLTFFI_STRUCT_U32.unpack_from
_BOLTFFI_UNPACK_I64 = _BOLTFFI_STRUCT_I64.unpack_from
_BOLTFFI_UNPACK_U64 = _BOLTFFI_STRUCT_U64.unpack_from
_BOLTFFI_UNPACK_F32 = _BOLTFFI_STRUCT_F32.unpack_from
_BOLTFFI_UNPACK_F64 = _BOLTFFI_STRUCT_F64.unpack_from


def _boltffi_u32(value: int) -> bytes:
    return _BOLTFFI_STRUCT_U32.pack(int(value))


def _boltffi_wire_bool(value: bool) -> bytes:
    return b"\x01" if value else b"\x00"


def _boltffi_wire_i8(value: int) -> bytes:
    return _BOLTFFI_STRUCT_I8.pack(int(value))


def _boltffi_wire_u8(value: int) -> bytes:
    return _BOLTFFI_STRUCT_U8.pack(int(value))


def _boltffi_wire_i16(value: int) -> bytes:
    return _BOLTFFI_STRUCT_I16.pack(int(value))


def _boltffi_wire_u16(value: int) -> bytes:
    return _BOLTFFI_STRUCT_U16.pack(int(value))


def _boltffi_wire_i32(value: int) -> bytes:
    return _BOLTFFI_STRUCT_I32.pack(int(value))


def _boltffi_wire_u32(value: int) -> bytes:
    return _BOLTFFI_STRUCT_U32.pack(int(value))


def _boltffi_wire_i64(value: int) -> bytes:
    return _BOLTFFI_STRUCT_I64.pack(int(value))


def _boltffi_wire_u64(value: int) -> bytes:
    return _BOLTFFI_STRUCT_U64.pack(int(value))


def _boltffi_wire_isize(value: int) -> bytes:
    return _BOLTFFI_STRUCT_I64.pack(int(value))


def _boltffi_wire_usize(value: int) -> bytes:
    return _BOLTFFI_STRUCT_U64.pack(int(value))


def _boltffi_wire_f32(value: float) -> bytes:
    return _BOLTFFI_STRUCT_F32.pack(float(value))


def _boltffi_wire_f64(value: float) -> bytes:
    return _BOLTFFI_STRUCT_F64.pack(float(value))


def _boltffi_wire_string(value: str) -> bytes:
    payload = value.encode("utf-8")
    return _boltffi_u32(len(payload)) + payload


def _boltffi_wire_bytes(value: bytes) -> bytes:
    payload = bytes(value)
    return _boltffi_u32(len(payload)) + payload


def _boltffi_split_duration(value: float) -> tuple[int, int]:
    total = float(value)
    if total < 0:
        raise ValueError("duration must be non-negative")
    seconds = int(total)
    nanos = round((total - seconds) * 1_000_000_000)
    if nanos == 1_000_000_000:
        return seconds + 1, 0
    return seconds, nanos


def _boltffi_split_system_time(value: float) -> tuple[int, int]:
    total = float(value)
    seconds = int(total // 1)
    nanos = round((total - seconds) * 1_000_000_000)
    if nanos == 1_000_000_000:
        return seconds + 1, 0
    return seconds, nanos


def _boltffi_wire_duration(value: float) -> bytes:
    seconds, nanos = _boltffi_split_duration(value)
    return seconds.to_bytes(8, "little", signed=False) + nanos.to_bytes(4, "little", signed=False)


def _boltffi_wire_system_time(value: float) -> bytes:
    seconds, nanos = _boltffi_split_system_time(value)
    return seconds.to_bytes(8, "little", signed=True) + nanos.to_bytes(4, "little", signed=False)


def _boltffi_wire_uuid(value: uuid.UUID | str) -> bytes:
    raw = uuid.UUID(str(value)).bytes
    high = int.from_bytes(raw[:8], "big")
    low = int.from_bytes(raw[8:], "big")
    return high.to_bytes(8, "little", signed=False) + low.to_bytes(8, "little", signed=False)


def _boltffi_wire_url(value: str) -> bytes:
    return _boltffi_wire_string(str(value))


def _boltffi_wire_optional(value, encode) -> bytes:
    if value is None:
        return b"\x00"
    return b"\x01" + encode(value)


def _boltffi_wire_result(value, encode_ok, encode_err) -> bytes:
    ok, payload = value
    if ok:
        return b"\x00" + encode_ok(payload)
    return b"\x01" + encode_err(payload)


def _boltffi_wire_sequence(value, count, encode) -> bytes:
    items = list(value)
    if len(items) != count:
        raise ValueError("invalid BoltFFI sequence count")
    return _boltffi_u32(count) + b"".join(encode(item) for item in items)


def _boltffi_wire_map(value, encode_key, encode_value) -> bytes:
    items = list(value.items())
    return _boltffi_u32(len(items)) + b"".join(
        encode_key(key) + encode_value(item) for key, item in items
    )


def _boltffi_enum_value(value, enum_type, enum_name: str) -> int:
    if not isinstance(value, enum_type):
        raise TypeError(f"expected {enum_name}")
    return int(value)


def _boltffi_error_exception(error):
    for error_type in type(error).__mro__:
        exception_type = globals().get(f"{error_type.__name__}Exception")
        if exception_type is not None:
            return exception_type(error)
    return RuntimeError(error)


def _boltffi_call(error_decoder, call):
    try:
        return call()
    except RuntimeError as error:
        if error.args and isinstance(error.args[0], bytes):
            raise _boltffi_error_exception(error_decoder(error.args[0])) from error
        raise


class _BoltFfiWireReader:
    __slots__ = ("_data", "_offset")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def finish(self) -> None:
        if self._offset != len(self._data):
            raise ValueError("trailing BoltFFI wire bytes")

    def has_remaining(self) -> bool:
        return self._offset < len(self._data)

    def read(self, count: int) -> bytes:
        offset = self._offset
        end = offset + count
        if end > len(self._data):
            raise ValueError("truncated BoltFFI wire bytes")
        self._offset = end
        return self._data[offset:end]

    def bool(self) -> bool:
        value = self.u8()
        if value > 1:
            raise ValueError("invalid BoltFFI bool")
        return value == 1

    def i8(self) -> int:
        offset = self._offset
        self._offset = offset + 1
        return _BOLTFFI_UNPACK_I8(self._data, offset)[0]

    def u8(self) -> int:
        offset = self._offset
        if offset >= len(self._data):
            raise ValueError("truncated BoltFFI wire bytes")
        self._offset = offset + 1
        return self._data[offset]

    def i16(self) -> int:
        offset = self._offset
        self._offset = offset + 2
        return _BOLTFFI_UNPACK_I16(self._data, offset)[0]

    def u16(self) -> int:
        offset = self._offset
        self._offset = offset + 2
        return _BOLTFFI_UNPACK_U16(self._data, offset)[0]

    def i32(self) -> int:
        offset = self._offset
        self._offset = offset + 4
        return _BOLTFFI_UNPACK_I32(self._data, offset)[0]

    def u32(self) -> int:
        offset = self._offset
        self._offset = offset + 4
        return _BOLTFFI_UNPACK_U32(self._data, offset)[0]

    def i64(self) -> int:
        offset = self._offset
        self._offset = offset + 8
        return _BOLTFFI_UNPACK_I64(self._data, offset)[0]

    def u64(self) -> int:
        offset = self._offset
        self._offset = offset + 8
        return _BOLTFFI_UNPACK_U64(self._data, offset)[0]

    def isize(self) -> int:
        return self.i64()

    def usize(self) -> int:
        return self.u64()

    def f32(self) -> float:
        offset = self._offset
        self._offset = offset + 4
        return _BOLTFFI_UNPACK_F32(self._data, offset)[0]

    def f64(self) -> float:
        offset = self._offset
        self._offset = offset + 8
        return _BOLTFFI_UNPACK_F64(self._data, offset)[0]

    def string(self) -> str:
        count = self.u32()
        offset = self._offset
        end = offset + count
        if end > len(self._data):
            raise ValueError("truncated BoltFFI wire bytes")
        self._offset = end
        return str(memoryview(self._data)[offset:end], "utf-8")

    def bytes(self) -> bytes:
        return self.read(self.u32())

    def fixed(self, layout) -> tuple:
        offset = self._offset
        self._offset = offset + layout.size
        return layout.unpack_from(self._data, offset)

    def fixed_sequence(self, layout, factory) -> list:
        count = self.u32()
        offset = self._offset
        end = offset + count * layout.size
        if end > len(self._data):
            raise ValueError("truncated BoltFFI wire bytes")
        self._offset = end
        window = memoryview(self._data)[offset:end]
        return [factory(*values) for values in layout.iter_unpack(window)]

    def enum_sequence(self, layout, enum_type) -> list:
        count = self.u32()
        offset = self._offset
        end = offset + count * layout.size
        if end > len(self._data):
            raise ValueError("truncated BoltFFI wire bytes")
        self._offset = end
        window = memoryview(self._data)[offset:end]
        members = enum_type._value2member_map_
        try:
            return [members[value] for (value,) in layout.iter_unpack(window)]
        except KeyError as error:
            raise ValueError(f"invalid {enum_type.__name__} value") from error

    def duration(self) -> float:
        return self.u64() + self.u32() / 1_000_000_000

    def system_time(self) -> float:
        return self.i64() + self.u32() / 1_000_000_000

    def uuid(self) -> uuid.UUID:
        high = self.u64().to_bytes(8, "big", signed=False)
        low = self.u64().to_bytes(8, "big", signed=False)
        return uuid.UUID(bytes=high + low)

    def url(self) -> str:
        return self.string()

    def optional(self, decode):
        tag = self.u8()
        if tag == 0:
            return None
        if tag == 1:
            return decode()
        raise ValueError("invalid BoltFFI option tag")

    def result(self, decode_ok, decode_err):
        tag = self.u8()
        if tag == 0:
            return (True, decode_ok())
        if tag == 1:
            return (False, decode_err())
        raise ValueError("invalid BoltFFI result tag")

    def sequence(self, decode) -> list:
        return [decode() for _ in range(self.u32())]

    def map(self, decode_key, decode_value) -> dict:
        return {decode_key(): decode_value() for _ in range(self.u32())}


def _boltffi_read_wire(data: bytes, decode):
    reader = _BoltFfiWireReader(data)
    try:
        value = decode(reader)
    except struct.error as error:
        raise ValueError("truncated BoltFFI wire bytes") from error
    reader.finish()
    return value



def _boltffi_read_fe83cddcf3822a1d(data: bytes):
    return _boltffi_read_wire(data, lambda reader: XybridError._boltffi_from_reader(reader))


_native._register_wire_codec("read_fe83cddcf3822a1d", _boltffi_read_fe83cddcf3822a1d)


def _boltffi_read_89cd31291d2aefa4(data: bytes):
    return _boltffi_read_wire(data, lambda reader: reader.string())


_native._register_wire_codec("read_89cd31291d2aefa4", _boltffi_read_89cd31291d2aefa4)


def _boltffi_read_29c0b1cb6cb65e99(data: bytes):
    return _boltffi_read_wire(data, lambda reader: XybridDownloadStatus._boltffi_from_reader(reader))


_native._register_wire_codec("read_29c0b1cb6cb65e99", _boltffi_read_29c0b1cb6cb65e99)


def _boltffi_read_49d0adb26a1528e6(data: bytes):
    return _boltffi_read_wire(data, lambda reader: XybridGenerationConfig._boltffi_from_reader(reader))


_native._register_wire_codec("read_49d0adb26a1528e6", _boltffi_read_49d0adb26a1528e6)


def _boltffi_read_bd1359a0ca4e78d7(data: bytes):
    return _boltffi_read_wire(data, lambda reader: reader.sequence(lambda: XybridVoiceInfo._boltffi_from_reader(reader)))


_native._register_wire_codec("read_bd1359a0ca4e78d7", _boltffi_read_bd1359a0ca4e78d7)


def _boltffi_read_74dbe00a1a77ad93(data: bytes):
    return _boltffi_read_wire(data, lambda reader: reader.optional(lambda: XybridVoiceInfo._boltffi_from_reader(reader)))


_native._register_wire_codec("read_74dbe00a1a77ad93", _boltffi_read_74dbe00a1a77ad93)


def _boltffi_read_c9bb5dd3c2ec1b2a(data: bytes):
    return _boltffi_read_wire(data, lambda reader: XybridResult._boltffi_from_reader(reader))


_native._register_wire_codec("read_c9bb5dd3c2ec1b2a", _boltffi_read_c9bb5dd3c2ec1b2a)


def _boltffi_read_e9a0b9fd71f8c9ff(data: bytes):
    return _boltffi_read_wire(data, lambda reader: XybridStreamEvent._boltffi_from_reader(reader))


_native._register_wire_codec("read_e9a0b9fd71f8c9ff", _boltffi_read_e9a0b9fd71f8c9ff)


def _boltffi_read_3cfe09c223256b1b(data: bytes):
    return _boltffi_read_wire(data, lambda reader: reader.sequence(lambda: XybridEnvelope._boltffi_from_reader(reader)))


_native._register_wire_codec("read_3cfe09c223256b1b", _boltffi_read_3cfe09c223256b1b)


def _boltffi_read_9415281aa52df749(data: bytes):
    return _boltffi_read_wire(data, lambda reader: reader.optional(lambda: reader.string()))


_native._register_wire_codec("read_9415281aa52df749", _boltffi_read_9415281aa52df749)


def _boltffi_read_c9e5fd91113e36a2(data: bytes):
    return _boltffi_read_wire(data, lambda reader: XybridEnvelope._boltffi_from_reader(reader))


_native._register_wire_codec("read_c9e5fd91113e36a2", _boltffi_read_c9e5fd91113e36a2)



def _boltffi_write_c26bffea5b1b16cc(id) -> bytes:
    return _boltffi_wire_string(id)


_native._register_wire_codec("write_c26bffea5b1b16cc", _boltffi_write_c26bffea5b1b16cc)


def _boltffi_write_766cdeb069dd2b0a(path) -> bytes:
    return _boltffi_wire_string(path)


_native._register_wire_codec("write_766cdeb069dd2b0a", _boltffi_write_766cdeb069dd2b0a)


def _boltffi_write_23c08924af812de7(repo) -> bytes:
    return _boltffi_wire_string(repo)


_native._register_wire_codec("write_23c08924af812de7", _boltffi_write_23c08924af812de7)


def _boltffi_write_8b5b57b4a65a4084(revision) -> bytes:
    return _boltffi_wire_string(revision)


_native._register_wire_codec("write_8b5b57b4a65a4084", _boltffi_write_8b5b57b4a65a4084)


def _boltffi_write_8d84d7157f6e715c(voice_id) -> bytes:
    return _boltffi_wire_string(voice_id)


_native._register_wire_codec("write_8d84d7157f6e715c", _boltffi_write_8d84d7157f6e715c)


def _boltffi_write_62eeac930738df49(envelope) -> bytes:
    return envelope._boltffi_wire()


_native._register_wire_codec("write_62eeac930738df49", _boltffi_write_62eeac930738df49)


def _boltffi_write_922e13039dd3c493(options) -> bytes:
    return _boltffi_wire_optional(options, lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire())


_native._register_wire_codec("write_922e13039dd3c493", _boltffi_write_922e13039dd3c493)


def _boltffi_write_45cfac4c89613282(api_key) -> bytes:
    return _boltffi_wire_string(api_key)


_native._register_wire_codec("write_45cfac4c89613282", _boltffi_write_45cfac4c89613282)


def _boltffi_write_736b3e4af7f4fdd8(endpoint) -> bytes:
    return _boltffi_wire_string(endpoint)


_native._register_wire_codec("write_736b3e4af7f4fdd8", _boltffi_write_736b3e4af7f4fdd8)


def _boltffi_write_f1696b1e73a7f219(version) -> bytes:
    return _boltffi_wire_string(version)


_native._register_wire_codec("write_f1696b1e73a7f219", _boltffi_write_f1696b1e73a7f219)


def _boltffi_write_15a81e8bd2929d67(label) -> bytes:
    return _boltffi_wire_string(label)


_native._register_wire_codec("write_15a81e8bd2929d67", _boltffi_write_15a81e8bd2929d67)


def _boltffi_write_94d2821c547dda88(key) -> bytes:
    return _boltffi_wire_string(key)


_native._register_wire_codec("write_94d2821c547dda88", _boltffi_write_94d2821c547dda88)


def _boltffi_write_ed06f1a2bac0816e(value) -> bytes:
    return _boltffi_wire_string(value)


_native._register_wire_codec("write_ed06f1a2bac0816e", _boltffi_write_ed06f1a2bac0816e)


def _boltffi_write_1b888e23ceb4a009(output_dir) -> bytes:
    return _boltffi_wire_string(output_dir)


_native._register_wire_codec("write_1b888e23ceb4a009", _boltffi_write_1b888e23ceb4a009)


def _boltffi_write_3f05cdfbd6f68333(user_text) -> bytes:
    return _boltffi_wire_string(user_text)


_native._register_wire_codec("write_3f05cdfbd6f68333", _boltffi_write_3f05cdfbd6f68333)


def _boltffi_write_544f2725dda888e0(prior_assistant_text) -> bytes:
    return _boltffi_wire_string(prior_assistant_text)


_native._register_wire_codec("write_544f2725dda888e0", _boltffi_write_544f2725dda888e0)


def _boltffi_write_d82fa724b184c72a(results) -> bytes:
    return _boltffi_wire_sequence(results, len(results), lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire())


_native._register_wire_codec("write_d82fa724b184c72a", _boltffi_write_d82fa724b184c72a)


def _boltffi_write_cd5b56c1c6bfc6e0(schema_json) -> bytes:
    return _boltffi_wire_string(schema_json)


_native._register_wire_codec("write_cd5b56c1c6bfc6e0", _boltffi_write_cd5b56c1c6bfc6e0)


def _boltffi_write_a4eb0446f96b83ef(api_key) -> bytes:
    return _boltffi_wire_optional(api_key, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0))


_native._register_wire_codec("write_a4eb0446f96b83ef", _boltffi_write_a4eb0446f96b83ef)


def _boltffi_write_5eb4d1ef4dc0ea3f(gateway_url) -> bytes:
    return _boltffi_wire_optional(gateway_url, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0))


_native._register_wire_codec("write_5eb4d1ef4dc0ea3f", _boltffi_write_5eb4d1ef4dc0ea3f)


def _boltffi_write_a67118e385bc3069(ingest_url) -> bytes:
    return _boltffi_wire_optional(ingest_url, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0))


_native._register_wire_codec("write_a67118e385bc3069", _boltffi_write_a67118e385bc3069)


def _boltffi_write_73b9be8d33badc3c(cache_dir) -> bytes:
    return _boltffi_wire_string(cache_dir)


_native._register_wire_codec("write_73b9be8d33badc3c", _boltffi_write_73b9be8d33badc3c)


def _boltffi_write_a087f842b9a13bc6(binding) -> bytes:
    return _boltffi_wire_string(binding)


_native._register_wire_codec("write_a087f842b9a13bc6", _boltffi_write_a087f842b9a13bc6)


def _boltffi_write_c0b19b1465c99138(provider) -> bytes:
    return _boltffi_wire_string(provider)


_native._register_wire_codec("write_c0b19b1465c99138", _boltffi_write_c0b19b1465c99138)


def _boltffi_write_b4a023e995953df2(url) -> bytes:
    return _boltffi_wire_string(url)


_native._register_wire_codec("write_b4a023e995953df2", _boltffi_write_b4a023e995953df2)


def _boltffi_write_83cc917c5525e5c3(model_id) -> bytes:
    return _boltffi_wire_string(model_id)


_native._register_wire_codec("write_83cc917c5525e5c3", _boltffi_write_83cc917c5525e5c3)



@dataclass(frozen=True, slots=True)
class XybridMetadataEntry:
    key: str
    value: str

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_string(self.key),
            _boltffi_wire_string(self.value),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridMetadataEntry":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridMetadataEntry":
        return cls(
            key=reader.string(),
            value=reader.string(),
        )


_native._register_xybrid_metadata_entry(XybridMetadataEntry)



@dataclass(frozen=True, slots=True)
class XybridEnvelope:
    kind: XybridEnvelopeKind
    metadata: list[XybridMetadataEntry]

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            self.kind._boltffi_wire(),
            _boltffi_wire_sequence(self.metadata, len(self.metadata), lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridEnvelope":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridEnvelope":
        return cls(
            kind=XybridEnvelopeKind._boltffi_from_reader(reader),
            metadata=reader.sequence(lambda: XybridMetadataEntry._boltffi_from_reader(reader)),
        )


_native._register_xybrid_envelope(XybridEnvelope)



@dataclass(frozen=True, slots=True)
class XybridToolDefinition:
    name: str
    description: str
    parameters_json: str

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_string(self.name),
            _boltffi_wire_string(self.description),
            _boltffi_wire_string(self.parameters_json),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridToolDefinition":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridToolDefinition":
        return cls(
            name=reader.string(),
            description=reader.string(),
            parameters_json=reader.string(),
        )


_native._register_xybrid_tool_definition(XybridToolDefinition)



@dataclass(frozen=True, slots=True)
class XybridToolCall:
    id: str
    name: str
    arguments_json: str

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_string(self.id),
            _boltffi_wire_string(self.name),
            _boltffi_wire_string(self.arguments_json),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridToolCall":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridToolCall":
        return cls(
            id=reader.string(),
            name=reader.string(),
            arguments_json=reader.string(),
        )


_native._register_xybrid_tool_call(XybridToolCall)



@dataclass(frozen=True, slots=True)
class XybridToolResult:
    call_id: str
    name: str
    content_json: str

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_string(self.call_id),
            _boltffi_wire_string(self.name),
            _boltffi_wire_string(self.content_json),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridToolResult":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridToolResult":
        return cls(
            call_id=reader.string(),
            name=reader.string(),
            content_json=reader.string(),
        )


_native._register_xybrid_tool_result(XybridToolResult)



@dataclass(frozen=True, slots=True)
class XybridGenerationConfig:
    max_tokens: int | None
    temperature: float | None
    top_p: float | None
    min_p: float | None
    top_k: int | None
    repetition_penalty: float | None
    stop_sequences: list[str]
    grammar: str | None
    tools: list[XybridToolDefinition]

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_optional(self.max_tokens, lambda __boltffi_value_0: _boltffi_wire_u32(__boltffi_value_0)),
            _boltffi_wire_optional(self.temperature, lambda __boltffi_value_0: _boltffi_wire_f32(__boltffi_value_0)),
            _boltffi_wire_optional(self.top_p, lambda __boltffi_value_0: _boltffi_wire_f32(__boltffi_value_0)),
            _boltffi_wire_optional(self.min_p, lambda __boltffi_value_0: _boltffi_wire_f32(__boltffi_value_0)),
            _boltffi_wire_optional(self.top_k, lambda __boltffi_value_0: _boltffi_wire_u32(__boltffi_value_0)),
            _boltffi_wire_optional(self.repetition_penalty, lambda __boltffi_value_0: _boltffi_wire_f32(__boltffi_value_0)),
            _boltffi_wire_sequence(self.stop_sequences, len(self.stop_sequences), lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)),
            _boltffi_wire_optional(self.grammar, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)),
            _boltffi_wire_sequence(self.tools, len(self.tools), lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridGenerationConfig":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridGenerationConfig":
        return cls(
            max_tokens=reader.optional(lambda: reader.u32()),
            temperature=reader.optional(lambda: reader.f32()),
            top_p=reader.optional(lambda: reader.f32()),
            min_p=reader.optional(lambda: reader.f32()),
            top_k=reader.optional(lambda: reader.u32()),
            repetition_penalty=reader.optional(lambda: reader.f32()),
            stop_sequences=reader.sequence(lambda: reader.string()),
            grammar=reader.optional(lambda: reader.string()),
            tools=reader.sequence(lambda: XybridToolDefinition._boltffi_from_reader(reader)),
        )


_native._register_xybrid_generation_config(XybridGenerationConfig)



@dataclass(frozen=True, slots=True)
class XybridRunOptions:
    generation_config: XybridGenerationConfig | None
    abort_on: list[XybridAbortSignal]
    fallback_to_cloud: bool
    max_grace_tokens: int
    correlation_id: str | None

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_optional(self.generation_config, lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()),
            _boltffi_wire_sequence(self.abort_on, len(self.abort_on), lambda __boltffi_value_0: _boltffi_wire_i32(_boltffi_enum_value(__boltffi_value_0, XybridAbortSignal, "XybridAbortSignal"))),
            _boltffi_wire_bool(self.fallback_to_cloud),
            _boltffi_wire_u32(self.max_grace_tokens),
            _boltffi_wire_optional(self.correlation_id, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridRunOptions":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridRunOptions":
        return cls(
            generation_config=reader.optional(lambda: XybridGenerationConfig._boltffi_from_reader(reader)),
            abort_on=reader.enum_sequence(_BOLTFFI_STRUCT_I32, XybridAbortSignal),
            fallback_to_cloud=reader.bool(),
            max_grace_tokens=reader.u32(),
            correlation_id=reader.optional(lambda: reader.string()),
        )


_native._register_xybrid_run_options(XybridRunOptions)



@dataclass(frozen=True, slots=True)
class XybridStageLatency:
    stage_id: str
    latency_ms: int

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_string(self.stage_id),
            _boltffi_wire_u32(self.latency_ms),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridStageLatency":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridStageLatency":
        return cls(
            stage_id=reader.string(),
            latency_ms=reader.u32(),
        )


_native._register_xybrid_stage_latency(XybridStageLatency)



@dataclass(frozen=True, slots=True)
class XybridInferenceMetrics:
    total_ms: int
    ttft_ms: int | None
    tokens_per_second: float | None
    prefill_tps: float | None
    decode_tps: float | None
    tokens_out: int | None
    stage_latencies_ms: list[XybridStageLatency]

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_u32(self.total_ms),
            _boltffi_wire_optional(self.ttft_ms, lambda __boltffi_value_0: _boltffi_wire_u32(__boltffi_value_0)),
            _boltffi_wire_optional(self.tokens_per_second, lambda __boltffi_value_0: _boltffi_wire_f32(__boltffi_value_0)),
            _boltffi_wire_optional(self.prefill_tps, lambda __boltffi_value_0: _boltffi_wire_f32(__boltffi_value_0)),
            _boltffi_wire_optional(self.decode_tps, lambda __boltffi_value_0: _boltffi_wire_f32(__boltffi_value_0)),
            _boltffi_wire_optional(self.tokens_out, lambda __boltffi_value_0: _boltffi_wire_u32(__boltffi_value_0)),
            _boltffi_wire_sequence(self.stage_latencies_ms, len(self.stage_latencies_ms), lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridInferenceMetrics":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridInferenceMetrics":
        return cls(
            total_ms=reader.u32(),
            ttft_ms=reader.optional(lambda: reader.u32()),
            tokens_per_second=reader.optional(lambda: reader.f32()),
            prefill_tps=reader.optional(lambda: reader.f32()),
            decode_tps=reader.optional(lambda: reader.f32()),
            tokens_out=reader.optional(lambda: reader.u32()),
            stage_latencies_ms=reader.sequence(lambda: XybridStageLatency._boltffi_from_reader(reader)),
        )


_native._register_xybrid_inference_metrics(XybridInferenceMetrics)



@dataclass(frozen=True, slots=True)
class XybridResult:
    envelope: XybridEnvelope
    output_type: XybridOutputType
    model_id: str
    latency_ms: int
    execution_target: XybridExecutionTarget
    metrics: XybridInferenceMetrics
    tool_calls: list[XybridToolCall]
    reasoning_content: str | None = None

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            self.envelope._boltffi_wire(),
            _boltffi_wire_i32(_boltffi_enum_value(self.output_type, XybridOutputType, "XybridOutputType")),
            _boltffi_wire_string(self.model_id),
            _boltffi_wire_u32(self.latency_ms),
            _boltffi_wire_i32(_boltffi_enum_value(self.execution_target, XybridExecutionTarget, "XybridExecutionTarget")),
            self.metrics._boltffi_wire(),
            _boltffi_wire_sequence(self.tool_calls, len(self.tool_calls), lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()),
            _boltffi_wire_optional(self.reasoning_content, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridResult":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridResult":
        envelope = XybridEnvelope._boltffi_from_reader(reader)
        output_type = XybridOutputType(reader.i32())
        model_id = reader.string()
        latency_ms = reader.u32()
        execution_target = XybridExecutionTarget(reader.i32())
        metrics = XybridInferenceMetrics._boltffi_from_reader(reader)
        tool_calls = reader.sequence(lambda: XybridToolCall._boltffi_from_reader(reader))
        reasoning_content = (
            reader.optional(lambda: reader.string())
            if reader.has_remaining()
            else next(
                (entry.value for entry in envelope.metadata if entry.key == "reasoning_content"),
                None,
            )
        )
        return cls(
            envelope=envelope,
            output_type=output_type,
            model_id=model_id,
            latency_ms=latency_ms,
            execution_target=execution_target,
            metrics=metrics,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )


_native._register_xybrid_result(XybridResult)



@dataclass(frozen=True, slots=True)
class XybridDownloadStatus:
    state: XybridDownloadState
    progress: float

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_i32(_boltffi_enum_value(self.state, XybridDownloadState, "XybridDownloadState")),
            _boltffi_wire_f32(self.progress),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridDownloadStatus":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridDownloadStatus":
        return cls(
            state=XybridDownloadState(reader.i32()),
            progress=reader.f32(),
        )


_native._register_xybrid_download_status(XybridDownloadStatus)



@dataclass(frozen=True, slots=True)
class XybridStreamToken:
    token: str
    token_id: int | None
    index: int
    cumulative_text: str
    finish_reason: str | None
    tool_calls: list[XybridToolCall]
    raw_text: str | None

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_string(self.token),
            _boltffi_wire_optional(self.token_id, lambda __boltffi_value_0: _boltffi_wire_i64(__boltffi_value_0)),
            _boltffi_wire_u64(self.index),
            _boltffi_wire_string(self.cumulative_text),
            _boltffi_wire_optional(self.finish_reason, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)),
            _boltffi_wire_sequence(self.tool_calls, len(self.tool_calls), lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()),
            _boltffi_wire_optional(self.raw_text, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridStreamToken":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridStreamToken":
        return cls(
            token=reader.string(),
            token_id=reader.optional(lambda: reader.i64()),
            index=reader.u64(),
            cumulative_text=reader.string(),
            finish_reason=reader.optional(lambda: reader.string()),
            tool_calls=reader.sequence(lambda: XybridToolCall._boltffi_from_reader(reader)),
            raw_text=reader.optional(lambda: reader.string()),
        )


_native._register_xybrid_stream_token(XybridStreamToken)



@dataclass(frozen=True, slots=True)
class XybridStreamEvent:
    kind: XybridStreamEventKind
    token: XybridStreamToken | None

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_i32(_boltffi_enum_value(self.kind, XybridStreamEventKind, "XybridStreamEventKind")),
            _boltffi_wire_optional(self.token, lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridStreamEvent":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridStreamEvent":
        return cls(
            kind=XybridStreamEventKind(reader.i32()),
            token=reader.optional(lambda: XybridStreamToken._boltffi_from_reader(reader)),
        )


_native._register_xybrid_stream_event(XybridStreamEvent)



@dataclass(frozen=True, slots=True)
class XybridVoiceInfo:
    id: str
    name: str
    gender: str | None
    language: str | None
    style: str | None

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_string(self.id),
            _boltffi_wire_string(self.name),
            _boltffi_wire_optional(self.gender, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)),
            _boltffi_wire_optional(self.language, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)),
            _boltffi_wire_optional(self.style, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridVoiceInfo":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridVoiceInfo":
        return cls(
            id=reader.string(),
            name=reader.string(),
            gender=reader.optional(lambda: reader.string()),
            language=reader.optional(lambda: reader.string()),
            style=reader.optional(lambda: reader.string()),
        )


_native._register_xybrid_voice_info(XybridVoiceInfo)




class XybridError:
    __slots__ = ()

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridError":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridError":
        tag = reader.u32()
        if tag == 0:
            return XybridErrorModelNotFound._boltffi_from_reader_payload(reader)
        if tag == 1:
            return XybridErrorDirectoryNotFound._boltffi_from_reader_payload(reader)
        if tag == 2:
            return XybridErrorMetadataNotFound._boltffi_from_reader_payload(reader)
        if tag == 3:
            return XybridErrorMetadataInvalid._boltffi_from_reader_payload(reader)
        if tag == 4:
            return XybridErrorLoadError._boltffi_from_reader_payload(reader)
        if tag == 5:
            return XybridErrorInferenceError._boltffi_from_reader_payload(reader)
        if tag == 6:
            return XybridErrorAbortedForCloudFallback._boltffi_from_reader_payload(reader)
        if tag == 7:
            return XybridErrorStreamingNotSupported._boltffi_from_reader_payload(reader)
        if tag == 8:
            return XybridErrorNotLoaded._boltffi_from_reader_payload(reader)
        if tag == 9:
            return XybridErrorConfigError._boltffi_from_reader_payload(reader)
        if tag == 10:
            return XybridErrorNetworkError._boltffi_from_reader_payload(reader)
        if tag == 11:
            return XybridErrorOffline._boltffi_from_reader_payload(reader)
        if tag == 12:
            return XybridErrorIoError._boltffi_from_reader_payload(reader)
        if tag == 13:
            return XybridErrorCacheError._boltffi_from_reader_payload(reader)
        if tag == 14:
            return XybridErrorPipelineError._boltffi_from_reader_payload(reader)
        if tag == 15:
            return XybridErrorCircuitOpen._boltffi_from_reader_payload(reader)
        if tag == 16:
            return XybridErrorRateLimited._boltffi_from_reader_payload(reader)
        if tag == 17:
            return XybridErrorTimeout._boltffi_from_reader_payload(reader)
        if tag == 18:
            return XybridErrorMissingArtifact._boltffi_from_reader_payload(reader)
        if tag == 19:
            return XybridErrorUnsupportedModelCapability._boltffi_from_reader_payload(reader)
        if tag == 20:
            return XybridErrorUnsupportedBackendCapability._boltffi_from_reader_payload(reader)
        if tag == 21:
            return XybridErrorInvalidImage._boltffi_from_reader_payload(reader)
        raise ValueError("invalid XybridError tag")


@dataclass(frozen=True, slots=True)
class XybridErrorModelNotFound(XybridError):
    id: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(0) + b"".join((
            _boltffi_wire_string(self.id),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorModelNotFound":
        return cls(
            id=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorDirectoryNotFound(XybridError):
    path: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(1) + b"".join((
            _boltffi_wire_string(self.path),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorDirectoryNotFound":
        return cls(
            path=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorMetadataNotFound(XybridError):
    path: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(2) + b"".join((
            _boltffi_wire_string(self.path),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorMetadataNotFound":
        return cls(
            path=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorMetadataInvalid(XybridError):
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(3) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorMetadataInvalid":
        return cls(
            message=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorLoadError(XybridError):
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(4) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorLoadError":
        return cls(
            message=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorInferenceError(XybridError):
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(5) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorInferenceError":
        return cls(
            message=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorAbortedForCloudFallback(XybridError):
    reason: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(6) + b"".join((
            _boltffi_wire_string(self.reason),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorAbortedForCloudFallback":
        return cls(
            reason=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorStreamingNotSupported(XybridError):
    pass

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(7)

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorStreamingNotSupported":
        return cls()


@dataclass(frozen=True, slots=True)
class XybridErrorNotLoaded(XybridError):
    pass

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(8)

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorNotLoaded":
        return cls()


@dataclass(frozen=True, slots=True)
class XybridErrorConfigError(XybridError):
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(9) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorConfigError":
        return cls(
            message=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorNetworkError(XybridError):
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(10) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorNetworkError":
        return cls(
            message=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorOffline(XybridError):
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(11) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorOffline":
        return cls(
            message=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorIoError(XybridError):
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(12) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorIoError":
        return cls(
            message=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorCacheError(XybridError):
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(13) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorCacheError":
        return cls(
            message=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorPipelineError(XybridError):
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(14) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorPipelineError":
        return cls(
            message=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorCircuitOpen(XybridError):
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(15) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorCircuitOpen":
        return cls(
            message=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorRateLimited(XybridError):
    retry_after_secs: int

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(16) + b"".join((
            _boltffi_wire_u64(self.retry_after_secs),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorRateLimited":
        return cls(
            retry_after_secs=reader.u64(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorTimeout(XybridError):
    timeout_ms: int

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(17) + b"".join((
            _boltffi_wire_u64(self.timeout_ms),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorTimeout":
        return cls(
            timeout_ms=reader.u64(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorMissingArtifact(XybridError):
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(18) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorMissingArtifact":
        return cls(
            message=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorUnsupportedModelCapability(XybridError):
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(19) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorUnsupportedModelCapability":
        return cls(
            message=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorUnsupportedBackendCapability(XybridError):
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(20) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorUnsupportedBackendCapability":
        return cls(
            message=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridErrorInvalidImage(XybridError):
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(21) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorInvalidImage":
        return cls(
            message=reader.string(),
        )



_native._register_xybrid_error(XybridError)


class XybridErrorException(RuntimeError):
    __slots__ = ("error",)

    def __init__(self, error: XybridError) -> None:
        self.error = error
        super().__init__(error)



class XybridEnvelopeKind:
    __slots__ = ()

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridEnvelopeKind":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridEnvelopeKind":
        tag = reader.u32()
        if tag == 0:
            return XybridEnvelopeKindText._boltffi_from_reader_payload(reader)
        if tag == 1:
            return XybridEnvelopeKindAudio._boltffi_from_reader_payload(reader)
        if tag == 2:
            return XybridEnvelopeKindEmbedding._boltffi_from_reader_payload(reader)
        if tag == 3:
            return XybridEnvelopeKindImage._boltffi_from_reader_payload(reader)
        if tag == 4:
            return XybridEnvelopeKindMultiPart._boltffi_from_reader_payload(reader)
        raise ValueError("invalid XybridEnvelopeKind tag")


@dataclass(frozen=True, slots=True)
class XybridEnvelopeKindText(XybridEnvelopeKind):
    text: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(0) + b"".join((
            _boltffi_wire_string(self.text),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridEnvelopeKindText":
        return cls(
            text=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridEnvelopeKindAudio(XybridEnvelopeKind):
    bytes: bytes

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(1) + b"".join((
            _boltffi_wire_bytes(self.bytes),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridEnvelopeKindAudio":
        return cls(
            bytes=reader.bytes(),
        )


@dataclass(frozen=True, slots=True)
class XybridEnvelopeKindEmbedding(XybridEnvelopeKind):
    values: list[float]

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(2) + b"".join((
            _boltffi_wire_sequence(self.values, len(self.values), lambda __boltffi_value_0: _boltffi_wire_f32(__boltffi_value_0)),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridEnvelopeKindEmbedding":
        return cls(
            values=reader.sequence(lambda: reader.f32()),
        )


@dataclass(frozen=True, slots=True)
class XybridEnvelopeKindImage(XybridEnvelopeKind):
    bytes: bytes
    format: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(3) + b"".join((
            _boltffi_wire_bytes(self.bytes),
            _boltffi_wire_string(self.format),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridEnvelopeKindImage":
        return cls(
            bytes=reader.bytes(),
            format=reader.string(),
        )


@dataclass(frozen=True, slots=True)
class XybridEnvelopeKindMultiPart(XybridEnvelopeKind):
    parts: list[XybridEnvelope]

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(4) + b"".join((
            _boltffi_wire_sequence(self.parts, len(self.parts), lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridEnvelopeKindMultiPart":
        return cls(
            parts=reader.sequence(lambda: XybridEnvelope._boltffi_from_reader(reader)),
        )



_native._register_xybrid_envelope_kind(XybridEnvelopeKind)



class XybridMessageRole(IntEnum):
    SYSTEM = 0
    USER = 1
    ASSISTANT = 2

_native._register_xybrid_message_role(XybridMessageRole)



class XybridAbortSignal(IntEnum):
    MEMORY_PRESSURE_WARN = 0
    MEMORY_PRESSURE_CRITICAL = 1
    THERMAL_HOT = 2
    THERMAL_CRITICAL = 3

_native._register_xybrid_abort_signal(XybridAbortSignal)



class XybridOutputType(IntEnum):
    TEXT = 0
    AUDIO = 1
    EMBEDDING = 2
    UNKNOWN = 3

_native._register_xybrid_output_type(XybridOutputType)



class XybridExecutionTarget(IntEnum):
    LOCAL = 0
    CLOUD = 1

_native._register_xybrid_execution_target(XybridExecutionTarget)



class XybridDownloadState(IntEnum):
    DOWNLOADING = 0
    READY = 1
    FAILED = 2

_native._register_xybrid_download_state(XybridDownloadState)



class XybridStreamEventKind(IntEnum):
    TOKEN = 0
    COMPLETE = 1

_native._register_xybrid_stream_event_kind(XybridStreamEventKind)



class XybridThermalState(IntEnum):
    NORMAL = 0
    WARM = 1
    HOT = 2
    CRITICAL = 3

_native._register_xybrid_thermal_state(XybridThermalState)




class XybridModel:
    __slots__ = ("_handle",)


    def __init__(self) -> None:
        raise TypeError("XybridModel cannot be constructed directly")


    @classmethod
    def _from_handle(cls, handle: int) -> "XybridModel":
        value = cls.__new__(cls)
        value._handle = handle
        return value

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._handle = None
            _native._boltffi_xybrid_model_release(handle)

    @classmethod
    def from_registry(cls, id: str) -> "XybridModel":
        return XybridModel._from_handle(_boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_from_registry(id)))

    @classmethod
    def from_registry_speculative(cls, id: str) -> "XybridModel":
        return XybridModel._from_handle(_boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_from_registry_speculative(id)))

    @classmethod
    def from_directory(cls, path: str) -> "XybridModel":
        return XybridModel._from_handle(_boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_from_directory(path)))

    @classmethod
    def from_bundle(cls, path: str) -> "XybridModel":
        return XybridModel._from_handle(_boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_from_bundle(path)))

    @classmethod
    def from_huggingface(cls, repo: str) -> "XybridModel":
        return XybridModel._from_handle(_boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_from_huggingface(repo)))

    @classmethod
    def from_huggingface_with_revision(cls, repo: str, revision: str) -> "XybridModel":
        return XybridModel._from_handle(_boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_from_huggingface_with_revision(repo, revision)))

    @classmethod
    def from_model_file(cls, path: str) -> "XybridModel":
        return XybridModel._from_handle(_boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_from_model_file(path)))

    def model_id(self) -> str:
        return _native._boltffi_xybrid_model_model_id(self._handle)

    def version(self) -> str:
        return _native._boltffi_xybrid_model_version(self._handle)

    def output_type(self) -> XybridOutputType:
        return _native._boltffi_xybrid_model_output_type(self._handle)

    def is_loaded(self) -> bool:
        return _native._boltffi_xybrid_model_is_loaded(self._handle)

    def is_cloud_serving(self) -> bool:
        return _native._boltffi_xybrid_model_is_cloud_serving(self._handle)

    def download_status(self) -> XybridDownloadStatus:
        return _boltffi_read_wire(_native._boltffi_xybrid_model_download_status(self._handle), lambda reader: XybridDownloadStatus._boltffi_from_reader(reader))

    def await_download(self, timeout_ms: int) -> XybridDownloadStatus:
        return _boltffi_read_wire(_native._boltffi_xybrid_model_await_download(self._handle, timeout_ms), lambda reader: XybridDownloadStatus._boltffi_from_reader(reader))

    def supports_streaming(self) -> bool:
        return _native._boltffi_xybrid_model_supports_streaming(self._handle)

    def supports_token_streaming(self) -> bool:
        return _native._boltffi_xybrid_model_supports_token_streaming(self._handle)

    def default_generation_config(self) -> XybridGenerationConfig:
        return _boltffi_read_wire(_native._boltffi_xybrid_model_default_generation_config(self._handle), lambda reader: XybridGenerationConfig._boltffi_from_reader(reader))

    def is_llm(self) -> bool:
        return _native._boltffi_xybrid_model_is_llm(self._handle)

    def supports_tool_calling(self) -> bool | None:
        return _native._boltffi_xybrid_model_supports_tool_calling(self._handle)

    def has_voices(self) -> bool:
        return _native._boltffi_xybrid_model_has_voices(self._handle)

    def voices(self) -> list[XybridVoiceInfo]:
        return _boltffi_read_wire(_native._boltffi_xybrid_model_voices(self._handle), lambda reader: reader.sequence(lambda: XybridVoiceInfo._boltffi_from_reader(reader)))

    def default_voice(self) -> XybridVoiceInfo | None:
        return _boltffi_read_wire(_native._boltffi_xybrid_model_default_voice(self._handle), lambda reader: reader.optional(lambda: XybridVoiceInfo._boltffi_from_reader(reader)))

    def voice(self, voice_id: str) -> XybridVoiceInfo | None:
        return _boltffi_read_wire(_native._boltffi_xybrid_model_voice(self._handle, voice_id), lambda reader: reader.optional(lambda: XybridVoiceInfo._boltffi_from_reader(reader)))

    def run(self, envelope: XybridEnvelope, options: XybridRunOptions | None) -> XybridResult:
        return _boltffi_read_wire(_boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_run(self._handle, envelope._boltffi_wire(), _boltffi_wire_optional(options, lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()))), lambda reader: XybridResult._boltffi_from_reader(reader))

    def run_stream(self, envelope: XybridEnvelope, options: XybridRunOptions | None) -> int:
        return _boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_run_stream(self._handle, envelope._boltffi_wire(), _boltffi_wire_optional(options, lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire())))

    def stream_next(self, stream_id: int) -> XybridStreamEvent:
        return _boltffi_read_wire(_boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_stream_next(self._handle, stream_id)), lambda reader: XybridStreamEvent._boltffi_from_reader(reader))

    def stream_result(self, stream_id: int) -> XybridResult:
        return _boltffi_read_wire(_boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_stream_result(self._handle, stream_id)), lambda reader: XybridResult._boltffi_from_reader(reader))

    def stream_close(self, stream_id: int) -> None:
        _native._boltffi_xybrid_model_stream_close(self._handle, stream_id)

    def run_with_context(self, envelope: XybridEnvelope, context: XybridConversationContext, options: XybridRunOptions | None) -> XybridResult:
        return _boltffi_read_wire(_boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_run_with_context(self._handle, envelope._boltffi_wire(), context._handle, _boltffi_wire_optional(options, lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()))), lambda reader: XybridResult._boltffi_from_reader(reader))

    def run_stream_with_context(self, envelope: XybridEnvelope, context: XybridConversationContext, options: XybridRunOptions | None) -> int:
        return _boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_run_stream_with_context(self._handle, envelope._boltffi_wire(), context._handle, _boltffi_wire_optional(options, lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire())))

    def warmup(self) -> None:
        _boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_warmup(self._handle))

    def unload(self) -> None:
        _boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_model_unload(self._handle))



class XybridConversationContext:
    __slots__ = ("_handle",)



    def __init__(self) -> None:
        self._handle = _native._boltffi_xybrid_conversation_context_new()



    @classmethod
    def _from_handle(cls, handle: int) -> "XybridConversationContext":
        value = cls.__new__(cls)
        value._handle = handle
        return value

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._handle = None
            _native._boltffi_xybrid_conversation_context_release(handle)

    @classmethod
    def with_id(cls, id: str) -> "XybridConversationContext":
        return XybridConversationContext._from_handle(_native._boltffi_xybrid_conversation_context_with_id(id))

    def push(self, envelope: XybridEnvelope) -> None:
        _boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_conversation_context_push(self._handle, envelope._boltffi_wire()))

    def set_system(self, envelope: XybridEnvelope) -> None:
        _boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_conversation_context_set_system(self._handle, envelope._boltffi_wire()))

    def clear(self) -> None:
        _native._boltffi_xybrid_conversation_context_clear(self._handle)

    def id(self) -> str:
        return _native._boltffi_xybrid_conversation_context_id(self._handle)

    def history_len(self) -> int:
        return _native._boltffi_xybrid_conversation_context_history_len(self._handle)

    def history(self) -> list[XybridEnvelope]:
        return _boltffi_read_wire(_native._boltffi_xybrid_conversation_context_history(self._handle), lambda reader: reader.sequence(lambda: XybridEnvelope._boltffi_from_reader(reader)))

    def has_system(self) -> bool:
        return _native._boltffi_xybrid_conversation_context_has_system(self._handle)

    def set_max_history_len(self, len: int) -> None:
        _native._boltffi_xybrid_conversation_context_set_max_history_len(self._handle, len)



class XybridTelemetryConfig:
    __slots__ = ("_handle",)



    def __init__(self, api_key: str) -> None:
        self._handle = _native._boltffi_xybrid_telemetry_config_new(api_key)



    @classmethod
    def _from_handle(cls, handle: int) -> "XybridTelemetryConfig":
        value = cls.__new__(cls)
        value._handle = handle
        return value

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._handle = None
            _native._boltffi_xybrid_telemetry_config_release(handle)

    def set_endpoint(self, endpoint: str) -> None:
        _native._boltffi_xybrid_telemetry_config_set_endpoint(self._handle, endpoint)

    def set_app_version(self, version: str) -> None:
        _native._boltffi_xybrid_telemetry_config_set_app_version(self._handle, version)

    def set_device_label(self, label: str) -> None:
        _native._boltffi_xybrid_telemetry_config_set_device_label(self._handle, label)

    def set_device_attribute(self, key: str, value: str) -> None:
        _native._boltffi_xybrid_telemetry_config_set_device_attribute(self._handle, key, value)

    def set_batch_size(self, batch_size: int) -> None:
        _native._boltffi_xybrid_telemetry_config_set_batch_size(self._handle, batch_size)

    def set_flush_interval_secs(self, secs: int) -> None:
        _native._boltffi_xybrid_telemetry_config_set_flush_interval_secs(self._handle, secs)

    def init(self) -> None:
        _boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_telemetry_config_init(self._handle))



class XybridBundle:
    __slots__ = ("_handle",)


    def __init__(self) -> None:
        raise TypeError("XybridBundle cannot be constructed directly")


    @classmethod
    def _from_handle(cls, handle: int) -> "XybridBundle":
        value = cls.__new__(cls)
        value._handle = handle
        return value

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._handle = None
            _native._boltffi_xybrid_bundle_release(handle)

    @classmethod
    def open(cls, path: str) -> "XybridBundle":
        return XybridBundle._from_handle(_boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_bundle_open(path)))

    def model_id(self) -> str:
        return _native._boltffi_xybrid_bundle_model_id(self._handle)

    def version(self) -> str:
        return _native._boltffi_xybrid_bundle_version(self._handle)

    def target(self) -> str:
        return _native._boltffi_xybrid_bundle_target(self._handle)

    def hash(self) -> str:
        return _native._boltffi_xybrid_bundle_hash(self._handle)

    def has_metadata(self) -> bool:
        return _native._boltffi_xybrid_bundle_has_metadata(self._handle)

    def file_count(self) -> int:
        return _native._boltffi_xybrid_bundle_file_count(self._handle)

    def file_name(self, index: int) -> str | None:
        return _boltffi_read_wire(_native._boltffi_xybrid_bundle_file_name(self._handle, index), lambda reader: reader.optional(lambda: reader.string()))

    def manifest_json(self) -> str:
        return _boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_bundle_manifest_json(self._handle))

    def metadata_json(self) -> str | None:
        return _boltffi_read_wire(_boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_bundle_metadata_json(self._handle)), lambda reader: reader.optional(lambda: reader.string()))

    def extract(self, output_dir: str) -> None:
        _boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native._boltffi_xybrid_bundle_extract(self._handle, output_dir))






def tool_results_envelope(user_text: str, prior_assistant_text: str, results: Sequence[XybridToolResult]) -> XybridEnvelope:
    return _boltffi_read_wire(_boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native.tool_results_envelope(user_text, prior_assistant_text, results)), lambda reader: XybridEnvelope._boltffi_from_reader(reader))
def json_schema_to_gbnf(schema_json: str) -> str:
    return _boltffi_call(_boltffi_read_fe83cddcf3822a1d, lambda: _native.json_schema_to_gbnf(schema_json))
def set_thermal_state(state: XybridThermalState) -> None:
    _native.set_thermal_state(state)
def clear_thermal_state() -> None:
    _native.clear_thermal_state()
def set_battery_level(percent: int) -> None:
    _native.set_battery_level(percent)
def clear_battery_level() -> None:
    _native.clear_battery_level()
def configure_runtime(api_key: str | None, gateway_url: str | None, ingest_url: str | None) -> None:
    _native.configure_runtime(_boltffi_wire_optional(api_key, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)), _boltffi_wire_optional(gateway_url, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)), _boltffi_wire_optional(ingest_url, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)))
def init_sdk_cache_dir(cache_dir: str) -> None:
    _native.init_sdk_cache_dir(cache_dir)
def set_binding(binding: str) -> None:
    _native.set_binding(binding)
def set_api_key(api_key: str) -> None:
    _native.set_api_key(api_key)
def set_provider_api_key(provider: str, api_key: str) -> None:
    _native.set_provider_api_key(provider, api_key)
def set_platform_url(url: str) -> None:
    _native.set_platform_url(url)
def set_speculative_cloud(enabled: bool) -> None:
    _native.set_speculative_cloud(enabled)
def has_api_key() -> bool:
    return _native.has_api_key()
def is_speculative_cloud_enabled() -> bool:
    return _native.is_speculative_cloud_enabled()
def will_speculate_for_model(model_id: str) -> bool:
    return _native.will_speculate_for_model(model_id)
def version() -> str:
    return _native.version()
def release_memory() -> int:
    return _native.release_memory()
def set_auto_release(enabled: bool) -> None:
    _native.set_auto_release(enabled)
def is_auto_release_enabled() -> bool:
    return _native.is_auto_release_enabled()
def telemetry_default_endpoint() -> str:
    return _native.telemetry_default_endpoint()
def telemetry_flush() -> None:
    _native.telemetry_flush()
def telemetry_shutdown() -> None:
    _native.telemetry_shutdown()

MODULE_NAME = "xybrid_bolt"
PACKAGE_NAME = "xybrid_bolt"
PACKAGE_VERSION = "0.7.0"

__all__ = [
    "MODULE_NAME",
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "XybridMetadataEntry",
    "XybridEnvelope",
    "XybridToolDefinition",
    "XybridToolCall",
    "XybridToolResult",
    "XybridGenerationConfig",
    "XybridRunOptions",
    "XybridStageLatency",
    "XybridInferenceMetrics",
    "XybridResult",
    "XybridDownloadStatus",
    "XybridStreamToken",
    "XybridStreamEvent",
    "XybridVoiceInfo",
    "XybridError",
    "XybridErrorException",
    "XybridErrorModelNotFound",
    "XybridErrorDirectoryNotFound",
    "XybridErrorMetadataNotFound",
    "XybridErrorMetadataInvalid",
    "XybridErrorLoadError",
    "XybridErrorInferenceError",
    "XybridErrorAbortedForCloudFallback",
    "XybridErrorStreamingNotSupported",
    "XybridErrorNotLoaded",
    "XybridErrorConfigError",
    "XybridErrorNetworkError",
    "XybridErrorOffline",
    "XybridErrorIoError",
    "XybridErrorCacheError",
    "XybridErrorPipelineError",
    "XybridErrorCircuitOpen",
    "XybridErrorRateLimited",
    "XybridErrorTimeout",
    "XybridErrorMissingArtifact",
    "XybridErrorUnsupportedModelCapability",
    "XybridErrorUnsupportedBackendCapability",
    "XybridErrorInvalidImage",
    "XybridEnvelopeKind",
    "XybridEnvelopeKindText",
    "XybridEnvelopeKindAudio",
    "XybridEnvelopeKindEmbedding",
    "XybridEnvelopeKindImage",
    "XybridEnvelopeKindMultiPart",
    "XybridMessageRole",
    "XybridAbortSignal",
    "XybridOutputType",
    "XybridExecutionTarget",
    "XybridDownloadState",
    "XybridStreamEventKind",
    "XybridThermalState",
    "XybridModel",
    "XybridConversationContext",
    "XybridTelemetryConfig",
    "XybridBundle",
    "tool_results_envelope",
    "json_schema_to_gbnf",
    "set_thermal_state",
    "clear_thermal_state",
    "set_battery_level",
    "clear_battery_level",
    "configure_runtime",
    "init_sdk_cache_dir",
    "set_binding",
    "set_api_key",
    "set_provider_api_key",
    "set_platform_url",
    "set_speculative_cloud",
    "has_api_key",
    "is_speculative_cloud_enabled",
    "will_speculate_for_model",
    "version",
    "release_memory",
    "set_auto_release",
    "is_auto_release_enabled",
    "telemetry_default_endpoint",
    "telemetry_flush",
    "telemetry_shutdown",
]
