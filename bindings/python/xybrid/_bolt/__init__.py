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



def _boltffi_read_347d2e2b11e825e8(data: bytes):
    return _boltffi_read_wire(data, lambda reader: XybridDownloadStatus._boltffi_from_reader(reader))


_native._register_wire_codec("read_347d2e2b11e825e8", _boltffi_read_347d2e2b11e825e8)


def _boltffi_read_4319db60c88eabca(data: bytes):
    return _boltffi_read_wire(data, lambda reader: reader.optional(lambda: reader.string()))


_native._register_wire_codec("read_4319db60c88eabca", _boltffi_read_4319db60c88eabca)


def _boltffi_read_09404a3c98b3f16c(data: bytes):
    return _boltffi_read_wire(data, lambda reader: XybridError._boltffi_from_reader(reader))


_native._register_wire_codec("read_09404a3c98b3f16c", _boltffi_read_09404a3c98b3f16c)


def _boltffi_read_89cd31291d2aefa4(data: bytes):
    return _boltffi_read_wire(data, lambda reader: reader.string())


_native._register_wire_codec("read_89cd31291d2aefa4", _boltffi_read_89cd31291d2aefa4)


def _boltffi_read_94828222bbb26957(data: bytes):
    return _boltffi_read_wire(data, lambda reader: XybridGenerationConfig._boltffi_from_reader(reader))


_native._register_wire_codec("read_94828222bbb26957", _boltffi_read_94828222bbb26957)


def _boltffi_read_4d62ca46c12c8415(data: bytes):
    return _boltffi_read_wire(data, lambda reader: reader.sequence(lambda: XybridVoiceInfo._boltffi_from_reader(reader)))


_native._register_wire_codec("read_4d62ca46c12c8415", _boltffi_read_4d62ca46c12c8415)


def _boltffi_read_9105d99f798275b3(data: bytes):
    return _boltffi_read_wire(data, lambda reader: reader.optional(lambda: XybridVoiceInfo._boltffi_from_reader(reader)))


_native._register_wire_codec("read_9105d99f798275b3", _boltffi_read_9105d99f798275b3)


def _boltffi_read_146d324414895b9b(data: bytes):
    return _boltffi_read_wire(data, lambda reader: XybridResult._boltffi_from_reader(reader))


_native._register_wire_codec("read_146d324414895b9b", _boltffi_read_146d324414895b9b)


def _boltffi_read_b467de4c6abf182c(data: bytes):
    return _boltffi_read_wire(data, lambda reader: XybridStreamEvent._boltffi_from_reader(reader))


_native._register_wire_codec("read_b467de4c6abf182c", _boltffi_read_b467de4c6abf182c)


def _boltffi_read_f45d365d172a914e(data: bytes):
    return _boltffi_read_wire(data, lambda reader: XybridPipelineResult._boltffi_from_reader(reader))


_native._register_wire_codec("read_f45d365d172a914e", _boltffi_read_f45d365d172a914e)


def _boltffi_read_88fe13077020b58c(data: bytes):
    return _boltffi_read_wire(data, lambda reader: reader.sequence(lambda: reader.string()))


_native._register_wire_codec("read_88fe13077020b58c", _boltffi_read_88fe13077020b58c)


def _boltffi_read_0d42d278c66eef7b(data: bytes):
    return _boltffi_read_wire(data, lambda reader: reader.sequence(lambda: XybridEnvelope._boltffi_from_reader(reader)))


_native._register_wire_codec("read_0d42d278c66eef7b", _boltffi_read_0d42d278c66eef7b)


def _boltffi_read_1497d20162db7713(data: bytes):
    return _boltffi_read_wire(data, lambda reader: XybridEnvelope._boltffi_from_reader(reader))


_native._register_wire_codec("read_1497d20162db7713", _boltffi_read_1497d20162db7713)


def _boltffi_read_14429286c6374023(data: bytes):
    return _boltffi_read_wire(data, lambda reader: XybridCacheStatus._boltffi_from_reader(reader))


_native._register_wire_codec("read_14429286c6374023", _boltffi_read_14429286c6374023)


def _boltffi_read_6cf2e6f5d394fbfa(data: bytes):
    return _boltffi_read_wire(data, lambda reader: reader.sequence(lambda: XybridCacheEntry._boltffi_from_reader(reader)))


_native._register_wire_codec("read_6cf2e6f5d394fbfa", _boltffi_read_6cf2e6f5d394fbfa)



def _boltffi_write_c26bffea5b1b16cc(id) -> bytes:
    return _boltffi_wire_string(id)


_native._register_wire_codec("write_c26bffea5b1b16cc", _boltffi_write_c26bffea5b1b16cc)


def _boltffi_write_cfe97cd6dcce32b6(platform) -> bytes:
    return _boltffi_wire_string(platform)


_native._register_wire_codec("write_cfe97cd6dcce32b6", _boltffi_write_cfe97cd6dcce32b6)


def _boltffi_write_6ff5cf7b33e854ed(config) -> bytes:
    return config._boltffi_wire()


_native._register_wire_codec("write_6ff5cf7b33e854ed", _boltffi_write_6ff5cf7b33e854ed)


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


def _boltffi_write_183cd822b59b9ed8(envelope) -> bytes:
    return envelope._boltffi_wire()


_native._register_wire_codec("write_183cd822b59b9ed8", _boltffi_write_183cd822b59b9ed8)


def _boltffi_write_360ec15d925d8351(options) -> bytes:
    return _boltffi_wire_optional(options, lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire())


_native._register_wire_codec("write_360ec15d925d8351", _boltffi_write_360ec15d925d8351)


def _boltffi_write_ab1ae15bdd9d5612(yaml) -> bytes:
    return _boltffi_wire_string(yaml)


_native._register_wire_codec("write_ab1ae15bdd9d5612", _boltffi_write_ab1ae15bdd9d5612)


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


def _boltffi_write_27d48eea4b9a1762(results) -> bytes:
    return _boltffi_wire_sequence(results, len(results), lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire())


_native._register_wire_codec("write_27d48eea4b9a1762", _boltffi_write_27d48eea4b9a1762)


def _boltffi_write_cd5b56c1c6bfc6e0(schema_json) -> bytes:
    return _boltffi_wire_string(schema_json)


_native._register_wire_codec("write_cd5b56c1c6bfc6e0", _boltffi_write_cd5b56c1c6bfc6e0)


def _boltffi_write_17f7606cd077d76c(api_key) -> bytes:
    return _boltffi_wire_optional(api_key, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0))


_native._register_wire_codec("write_17f7606cd077d76c", _boltffi_write_17f7606cd077d76c)


def _boltffi_write_d1c12e1524cd3dbc(gateway_url) -> bytes:
    return _boltffi_wire_optional(gateway_url, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0))


_native._register_wire_codec("write_d1c12e1524cd3dbc", _boltffi_write_d1c12e1524cd3dbc)


def _boltffi_write_5575cc29a91ce4ea(ingest_url) -> bytes:
    return _boltffi_wire_optional(ingest_url, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0))


_native._register_wire_codec("write_5575cc29a91ce4ea", _boltffi_write_5575cc29a91ce4ea)


def _boltffi_write_73b9be8d33badc3c(cache_dir) -> bytes:
    return _boltffi_wire_string(cache_dir)


_native._register_wire_codec("write_73b9be8d33badc3c", _boltffi_write_73b9be8d33badc3c)


def _boltffi_write_83cc917c5525e5c3(model_id) -> bytes:
    return _boltffi_wire_string(model_id)


_native._register_wire_codec("write_83cc917c5525e5c3", _boltffi_write_83cc917c5525e5c3)


def _boltffi_write_a087f842b9a13bc6(binding) -> bytes:
    return _boltffi_wire_string(binding)


_native._register_wire_codec("write_a087f842b9a13bc6", _boltffi_write_a087f842b9a13bc6)


def _boltffi_write_c0b19b1465c99138(provider) -> bytes:
    return _boltffi_wire_string(provider)


_native._register_wire_codec("write_c0b19b1465c99138", _boltffi_write_c0b19b1465c99138)


def _boltffi_write_b4a023e995953df2(url) -> bytes:
    return _boltffi_wire_string(url)


_native._register_wire_codec("write_b4a023e995953df2", _boltffi_write_b4a023e995953df2)



class XybridError:
    """Errors surfaced across the FFI boundary. Variants mirror
    [`facade::Error`] — the facade owns the SDK→FFI translation; this enum
    only re-decorates it for the BoltFFI generator (proc macros must live
    on the type definition).

    Named `XybridError` (not `Error`) so the emitted Swift type doesn't
    shadow / collide with Swift's stdlib `Error` protocol, and so the
    Kotlin sealed-hierarchy name matches the existing uniffi consumer
    expectations.

    **Variant order is part of the wire contract.** BoltFFI encodes `#[error]`
    (and `#[data]`) enums by ordinal tag, so reordering or inserting a variant
    renumbers every variant after it and breaks already-built foreign clients.
    Only ever append at the tail, and keep this order in lockstep with
    [`facade::Error`] and its `code()` table.
    """
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
        if tag == 22:
            return XybridErrorCancelled._boltffi_from_reader_payload(reader)
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


@dataclass(frozen=True, slots=True)
class XybridErrorCancelled(XybridError):
    """The host called `cancel` — today, on a model download."""
    message: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(22) + b"".join((
            _boltffi_wire_string(self.message),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridErrorCancelled":
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
    """Where a result was produced — observed fact, not a routing preference."""
    LOCAL = 0
    CLOUD = 1

_native._register_xybrid_execution_target(XybridExecutionTarget)



class XybridDownloadState(IntEnum):
    """Lifecycle of a model download — a standalone [`XybridDownload`] or
    the background download behind a speculative load.
    """
    DOWNLOADING = 0
    READY = 1
    FAILED = 2
    CANCELLED = 3

_native._register_xybrid_download_state(XybridDownloadState)



class XybridStreamEventKind(IntEnum):
    TOKEN = 0
    COMPLETE = 1

_native._register_xybrid_stream_event_kind(XybridStreamEventKind)



class XybridCacheEntryLocation(IntEnum):
    REGISTRY = 0
    EXTRACTED = 1
    HUGGING_FACE = 2
    HUGGING_FACE_HUB = 3

_native._register_xybrid_cache_entry_location(XybridCacheEntryLocation)



class XybridThermalState(IntEnum):
    NORMAL = 0
    WARM = 1
    HOT = 2
    CRITICAL = 3

_native._register_xybrid_thermal_state(XybridThermalState)



class XybridVadMode:
    """How voice-activity detection (VAD) chunking is resolved for a session.

    There is deliberately no "on, with the default model" variant: nothing
    ships a bundled Silero model, and the core handles VAD-enabled-without-a-
    directory by warning and silently falling back to fixed-window chunking.
    Enabling VAD therefore requires naming a directory.
    """
    __slots__ = ()

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridVadMode":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridVadMode":
        tag = reader.u32()
        if tag == 0:
            return XybridVadModeOff._boltffi_from_reader_payload(reader)
        if tag == 1:
            return XybridVadModeEnabled._boltffi_from_reader_payload(reader)
        raise ValueError("invalid XybridVadMode tag")


@dataclass(frozen=True, slots=True)
class XybridVadModeOff(XybridVadMode):
    """Fixed time-window chunking; no voice-activity detection."""
    pass

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(0)

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridVadModeOff":
        return cls()


@dataclass(frozen=True, slots=True)
class XybridVadModeEnabled(XybridVadMode):
    """VAD on, using the Silero model in this directory, which must contain a
    `model.onnx`.
    """
    model_dir: str

    def _boltffi_wire(self) -> bytes:
        return _boltffi_wire_u32(1) + b"".join((
            _boltffi_wire_string(self.model_dir),
        ))

    @classmethod
    def _boltffi_from_reader_payload(cls, reader: "_BoltFfiWireReader") -> "XybridVadModeEnabled":
        return cls(
            model_dir=reader.string(),
        )



_native._register_xybrid_vad_mode(XybridVadMode)




@dataclass(frozen=True, slots=True)
class XybridMetadataEntry:
    """Single metadata key/value entry. BoltFFI doesn't auto-derive
    `WireEncode` for `HashMap<String, String>`, so we expose metadata as
    `Vec<XybridMetadataEntry>`. The conversion back to `HashMap` happens
    at the facade boundary inside [`XybridEnvelope::into`].
    """
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
    """A tool (function) the model may ask to call.

    `parameters_json` is the JSON Schema for the arguments, carried as a JSON
    string because no binding generator can describe an arbitrary JSON tree.
    """
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
    """One tool call the model emitted, from [`XybridResult::tool_calls`]."""
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
    """The outcome of running one tool, fed back with [`tool_results_envelope`]."""
    call_id: str
    """The [`XybridToolCall::id`] this answers."""
    name: str
    content_json: str
    """The tool's output as a JSON string."""

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
    """Optional GBNF grammar constraining generation to structured output
    (local llama backend only). Produce one from a JSON Schema with
    [`json_schema_to_gbnf`], or pass raw GBNF. Appended last: `#[data]`
    PODs serialize by field order across the FFI boundary.
    """
    tools: list[XybridToolDefinition]
    """Tools the model may call this turn. Empty means no tool calling —
    existing behavior, unchanged. Appended after `grammar` for the same
    field-order reason.

    Tool calling is llama.cpp-only today; unsupported paths (no embedded
    chat template, the mistralrs backend, the cloud fallback leg) reject
    tool-bearing requests rather than quietly generating without them.
    """

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
    cloud_provider: str | None = None
    cloud_model: str | None = None
    cloud_gateway_url: str | None = None

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_optional(self.generation_config, lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()),
            _boltffi_wire_sequence(self.abort_on, len(self.abort_on), lambda __boltffi_value_0: _boltffi_wire_i32(_boltffi_enum_value(__boltffi_value_0, XybridAbortSignal, "XybridAbortSignal"))),
            _boltffi_wire_bool(self.fallback_to_cloud),
            _boltffi_wire_u32(self.max_grace_tokens),
            _boltffi_wire_optional(self.correlation_id, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)),
            _boltffi_wire_optional(self.cloud_provider, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)),
            _boltffi_wire_optional(self.cloud_model, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)),
            _boltffi_wire_optional(self.cloud_gateway_url, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)),
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
            cloud_provider=reader.optional(lambda: reader.string()),
            cloud_model=reader.optional(lambda: reader.string()),
            cloud_gateway_url=reader.optional(lambda: reader.string()),
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
    """Inference output. Named `XybridResult` (not `XybridInferenceResult`)
    to match the existing uniffi-generated Kotlin/Swift name — the iOS
    example references `XybridResult` directly.
    """
    envelope: XybridEnvelope
    output_type: XybridOutputType
    model_id: str
    latency_ms: int
    execution_target: XybridExecutionTarget
    """Where the answer actually came from. Cloud fallback keeps `model_id`
    identical on both legs, so this is the only way to tell them apart.
    """
    metrics: XybridInferenceMetrics
    tool_calls: list[XybridToolCall]
    """Tool calls the model emitted this turn. Empty unless the request
    offered tools via [`XybridGenerationConfig::tools`].
    `#[data]` PODs serialize by field order across the FFI boundary.
    """
    reasoning_content: str | None = None
    """Model reasoning emitted separately from the final answer text.
    Appended last because `#[data]` fields serialize in declaration order.
    """

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
    """Download progress, bytes and state in one consistent read.

    `progress` is aggregated across every artifact the model needs (weights
    plus companions such as a vision projector), never moves backwards, and
    reaches 1.0 only alongside `Ready`. `totalBytes` is null when the source
    declares no size — a Hugging Face repo, or a registry entry without one —
    in which case `downloadedBytes` is still exact and `progress` is coarser.

    Derives `Copy` because it is carried as a stream item.
    """
    state: XybridDownloadState
    progress: float
    """0.0..=1.0."""
    downloaded_bytes: int
    """Bytes written so far, across every artifact."""
    total_bytes: int | None
    """Declared total across every artifact, or null when unknown."""

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_i32(_boltffi_enum_value(self.state, XybridDownloadState, "XybridDownloadState")),
            _boltffi_wire_f32(self.progress),
            _boltffi_wire_u64(self.downloaded_bytes),
            _boltffi_wire_optional(self.total_bytes, lambda __boltffi_value_0: _boltffi_wire_u64(__boltffi_value_0)),
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
            downloaded_bytes=reader.u64(),
            total_bytes=reader.optional(lambda: reader.u64()),
        )


_native._register_xybrid_download_status(XybridDownloadStatus)



@dataclass(frozen=True, slots=True)
class XybridStageResult:
    """What one stage of a pipeline run produced."""
    stage_id: str
    """Stage identifier from the pipeline YAML (`id:`), or the model ID when
    the stage declares none. Matches [`XybridPipeline::stage_names`].
    """
    envelope: XybridEnvelope
    """This stage's output, which is also the next stage's input — the
    transcript of an ASR stage, the reply of an LLM stage.
    """
    output_type: XybridOutputType
    latency_ms: int
    execution_target: XybridExecutionTarget
    """Where this stage ran. Stages of one pipeline can run in different
    places.
    """
    metrics: XybridInferenceMetrics
    """Generation figures (TTFT, tokens per second) when this stage is a
    language model; `total_ms` is the stage latency.
    """

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_string(self.stage_id),
            self.envelope._boltffi_wire(),
            _boltffi_wire_i32(_boltffi_enum_value(self.output_type, XybridOutputType, "XybridOutputType")),
            _boltffi_wire_u32(self.latency_ms),
            _boltffi_wire_i32(_boltffi_enum_value(self.execution_target, XybridExecutionTarget, "XybridExecutionTarget")),
            self.metrics._boltffi_wire(),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridStageResult":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridStageResult":
        return cls(
            stage_id=reader.string(),
            envelope=XybridEnvelope._boltffi_from_reader(reader),
            output_type=XybridOutputType(reader.i32()),
            latency_ms=reader.u32(),
            execution_target=XybridExecutionTarget(reader.i32()),
            metrics=XybridInferenceMetrics._boltffi_from_reader(reader),
        )


_native._register_xybrid_stage_result(XybridStageResult)



@dataclass(frozen=True, slots=True)
class XybridPipelineResult:
    """Result of [`XybridPipeline::run`]: the final output plus every stage's own
    output, so a voice pipeline can show the transcript and the reply as well
    as play the audio.
    """
    envelope: XybridEnvelope
    """The final stage's output — the same envelope as the last entry of
    `stages`.
    """
    output_type: XybridOutputType
    latency_ms: int
    """Wall-clock time of the whole run."""
    stages: list[XybridStageResult]
    """Every executed stage, in order."""

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            self.envelope._boltffi_wire(),
            _boltffi_wire_i32(_boltffi_enum_value(self.output_type, XybridOutputType, "XybridOutputType")),
            _boltffi_wire_u32(self.latency_ms),
            _boltffi_wire_sequence(self.stages, len(self.stages), lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridPipelineResult":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridPipelineResult":
        return cls(
            envelope=XybridEnvelope._boltffi_from_reader(reader),
            output_type=XybridOutputType(reader.i32()),
            latency_ms=reader.u32(),
            stages=reader.sequence(lambda: XybridStageResult._boltffi_from_reader(reader)),
        )


_native._register_xybrid_pipeline_result(XybridPipelineResult)



@dataclass(frozen=True, slots=True)
class XybridStreamToken:
    token: str
    token_id: int | None
    index: int
    cumulative_text: str
    finish_reason: str | None
    """`"tool_calls"` when the turn ended on a parseable tool-call block."""
    tool_calls: list[XybridToolCall]
    """Tool calls parsed from the completed turn — populated on the
    **terminal** token only (the one carrying `finish_reason`).

    Tool-call blocks are suppressed from the emitted stream, so there is
    nothing in the token text to parse: a streaming caller halts here,
    runs the tools, then continues the turn by streaming a
    [`tool_results_envelope`] through the same call. Empty on every
    mid-stream token and on turns that emitted no call.
    """
    raw_text: str | None
    """The completed turn's raw output text, tool-call block included — pass
    it to [`tool_results_envelope`] as `prior_assistant_text`.

    Present only alongside a non-empty [`Self::tool_calls`]. Not the same
    as `cumulative_text`, which reports the *emitted* text with the
    protocol blocks suppressed — which is why this field exists at all.
    """

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
    """One pull from a streaming inference session.

    This is a flat record instead of a data-carrying enum because the pinned
    C# generator cannot lower that enum shape reliably. `kind` selects the one
    populated payload: `token` for `Token`, none for `Complete`. A `Complete`
    event is followed by [`XybridModel::stream_result`] to retrieve the final
    result. Inference failures are returned as typed [`XybridError`] values by
    [`XybridModel::stream_next`].
    """
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



@dataclass(frozen=True, slots=True)
class XybridCacheEntry:
    model_id: str
    location: XybridCacheEntryLocation
    path: str
    size_bytes: int

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_string(self.model_id),
            _boltffi_wire_i32(_boltffi_enum_value(self.location, XybridCacheEntryLocation, "XybridCacheEntryLocation")),
            _boltffi_wire_string(self.path),
            _boltffi_wire_u64(self.size_bytes),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridCacheEntry":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridCacheEntry":
        return cls(
            model_id=reader.string(),
            location=XybridCacheEntryLocation(reader.i32()),
            path=reader.string(),
            size_bytes=reader.u64(),
        )


_native._register_xybrid_cache_entry(XybridCacheEntry)



@dataclass(frozen=True, slots=True)
class XybridCacheStatus:
    total_size_bytes: int
    entry_count: int
    model_count: int
    extracted_model_count: int
    cache_root: str

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_u64(self.total_size_bytes),
            _boltffi_wire_u32(self.entry_count),
            _boltffi_wire_u32(self.model_count),
            _boltffi_wire_u32(self.extracted_model_count),
            _boltffi_wire_string(self.cache_root),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridCacheStatus":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridCacheStatus":
        return cls(
            total_size_bytes=reader.u64(),
            entry_count=reader.u32(),
            model_count=reader.u32(),
            extracted_model_count=reader.u32(),
            cache_root=reader.string(),
        )


_native._register_xybrid_cache_status(XybridCacheStatus)



@dataclass(frozen=True, slots=True)
class XybridStreamingConfig:
    """Configuration for a live ASR session.

    The model is not named here — it comes from the loaded `XybridModel` the
    session is opened on. This only configures *how* the audio is chunked.
    """
    sample_rate: int
    """Sample rate of the audio you will feed. Must be 16000; the ASR
    backends are fixed there, so anything else is rejected rather than
    silently resampled.
    """
    vad: XybridVadMode
    """Voice-activity-detection mode."""
    vad_threshold: float
    """VAD sensitivity, 0.0–1.0. Ignored when `vad` is `Off`."""
    language: str | None
    """Language hint (e.g. `"en"`); null uses the model default."""
    audio_ctx: int | None
    """Whisper encoder context in mel frames; null uses the model default."""

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_u32(self.sample_rate),
            self.vad._boltffi_wire(),
            _boltffi_wire_f32(self.vad_threshold),
            _boltffi_wire_optional(self.language, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)),
            _boltffi_wire_optional(self.audio_ctx, lambda __boltffi_value_0: _boltffi_wire_u32(__boltffi_value_0)),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridStreamingConfig":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridStreamingConfig":
        return cls(
            sample_rate=reader.u32(),
            vad=XybridVadMode._boltffi_from_reader(reader),
            vad_threshold=reader.f32(),
            language=reader.optional(lambda: reader.string()),
            audio_ctx=reader.optional(lambda: reader.u32()),
        )


_native._register_xybrid_streaming_config(XybridStreamingConfig)



@dataclass(frozen=True, slots=True)
class XybridPartialResult:
    """A partial transcript emitted while audio is streaming."""
    text: str
    """Best-effort transcript so far. Cumulative, not a delta — render it in
    place of the previous partial rather than appending.
    """
    is_stable: bool
    """`true` once this span is committed and will not change."""
    chunk_sequence: int
    """Monotonic chunk sequence number this result corresponds to."""
    audio_duration_ms: int
    """Audio covered so far, in milliseconds."""

    def _boltffi_wire(self) -> bytes:
        return b"".join((
            _boltffi_wire_string(self.text),
            _boltffi_wire_bool(self.is_stable),
            _boltffi_wire_u64(self.chunk_sequence),
            _boltffi_wire_u64(self.audio_duration_ms),
        ))

    @classmethod
    def _boltffi_from_wire(cls, data: bytes) -> "XybridPartialResult":
        reader = _BoltFfiWireReader(data)
        try:
            value = cls._boltffi_from_reader(reader)
        except struct.error as error:
            raise ValueError("truncated BoltFFI wire bytes") from error
        reader.finish()
        return value

    @classmethod
    def _boltffi_from_reader(cls, reader: "_BoltFfiWireReader") -> "XybridPartialResult":
        return cls(
            text=reader.string(),
            is_stable=reader.bool(),
            chunk_sequence=reader.u64(),
            audio_duration_ms=reader.u64(),
        )


_native._register_xybrid_partial_result(XybridPartialResult)




class XybridDownload:
    __slots__ = ("_handle",)


    def __init__(self) -> None:
        raise TypeError("XybridDownload cannot be constructed directly")


    @classmethod
    def _from_handle(cls, handle: int) -> "XybridDownload":
        value = cls.__new__(cls)
        value._handle = handle
        return value

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._handle = None
            _native._boltffi_xybrid_download_release(handle)

    @classmethod
    def from_registry(cls, id: str) -> "XybridDownload":
        """Start downloading a registry model. Returns immediately."""
        return XybridDownload._from_handle(_native._boltffi_xybrid_download_from_registry(id))

    @classmethod
    def from_registry_with_platform(cls, id: str, platform: str) -> "XybridDownload":
        """Start downloading a registry model resolved for a specific platform."""
        return XybridDownload._from_handle(_native._boltffi_xybrid_download_from_registry_with_platform(id, platform))

    def status(self) -> XybridDownloadStatus:
        """Current snapshot. Never blocks — safe from a UI thread or a per-frame
        render loop.
        """
        return _boltffi_read_wire(_native._boltffi_xybrid_download_status(self._handle), lambda reader: XybridDownloadStatus._boltffi_from_reader(reader))

    def is_finished(self) -> bool:
        """Whether the download reached a terminal state."""
        return _native._boltffi_xybrid_download_is_finished(self._handle)

    def error(self) -> str | None:
        """The failure message once the download ended in `Failed` or
        `Cancelled`; null otherwise. The stream carries the terminal *state*,
        this carries the reason.
        """
        return _boltffi_read_wire(_native._boltffi_xybrid_download_error(self._handle), lambda reader: reader.optional(lambda: reader.string()))

    def cancel(self) -> None:
        """Ask the download to stop. Takes effect within one chunk read, discards
        the partial file, and moves the status to `Cancelled`. Idempotent, and
        a no-op once the download is terminal.
        """
        _native._boltffi_xybrid_download_cancel(self._handle)

    def progress(self) -> "XybridDownloadProgressSubscription":
        """Pushed progress updates, closing once the download is terminal.

        Generated as an `AsyncStream` in Swift, a `Flow` in Kotlin, an
        `IAsyncEnumerable` in C# and a subscription object in Python.
        Cancelling the consuming task / scope / token unsubscribes; it does
        **not** cancel the download itself — call [`Self::cancel`] for that.

        The current snapshot is delivered first, so subscribing late still
        yields a frame, and a download that already finished closes at once
        instead of hanging.
        """
        return XybridDownloadProgressSubscription._from_handle(_native.progress(self._handle))


class XybridDownloadProgressSubscription:
    __slots__ = ("_handle",)

    def __init__(self) -> None:
        raise TypeError("XybridDownloadProgressSubscription cannot be constructed directly")

    @classmethod
    def _from_handle(cls, handle: int) -> "XybridDownloadProgressSubscription":
        value = cls.__new__(cls)
        value._handle = handle
        return value

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._handle = None
            _native.progress_free(handle)

    def pop_batch(self, max_count: int = 16) -> list[XybridDownloadStatus]:
        data = _native.progress_pop_batch(self._require_handle(), max_count)
        return _boltffi_read_wire(data, lambda reader: reader.sequence(lambda: XybridDownloadStatus._boltffi_from_reader(reader))) if data else []

    def wait(self, timeout_milliseconds: int) -> int:
        return _native.progress_wait(self._require_handle(), timeout_milliseconds)

    def unsubscribe(self) -> None:
        handle = self._require_handle()
        self._handle = None
        _native.progress_unsubscribe(handle)
        _native.progress_free(handle)

    def _require_handle(self) -> int:
        handle = self._handle
        if handle is None:
            raise RuntimeError("stream subscription is closed")
        return handle



class XybridStreamingSession:
    __slots__ = ("_handle",)


    def __init__(self) -> None:
        raise TypeError("XybridStreamingSession cannot be constructed directly")


    @classmethod
    def _from_handle(cls, handle: int) -> "XybridStreamingSession":
        value = cls.__new__(cls)
        value._handle = handle
        return value

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._handle = None
            _native._boltffi_xybrid_streaming_session_release(handle)

    @classmethod
    def for_model(cls, model: XybridModel, config: XybridStreamingConfig) -> "XybridStreamingSession":
        """Open a session on an already-loaded ASR model.

        Starts a worker thread and warms the weights, so the first spoken
        words do not pay the cold-start cost. Returns an error for a model
        that does not support streaming, or a sample rate other than 16000.
        """
        return XybridStreamingSession._from_handle(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_streaming_session_for_model(model._handle, config._boltffi_wire())))

    def feed(self, samples: Sequence[float]) -> None:
        """Feed PCM f32 mono 16 kHz samples.

        Hands the buffer to the worker and returns; transcription happens
        there, never on the caller's thread. Blocks only when the queue is
        full, which back-pressures a producer feeding faster than the model
        can keep up.
        """
        _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_streaming_session_feed(self._handle, samples))

    def flush(self) -> str:
        """Finalize: drain buffered audio and return the complete transcript.

        The session is over afterwards — `feed` fails and the partial stream
        closes. Blocks until the last chunk is transcribed, so call it off the
        UI thread.
        """
        return _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_streaming_session_flush(self._handle))

    def reset(self) -> None:
        """Reset to transcribe fresh audio without reloading the model."""
        _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_streaming_session_reset(self._handle))

    def cancel(self) -> None:
        """Stop the session and release the model, discarding buffered audio.

        Idempotent. Use [`Self::flush`] when you want the transcript — this is
        the "user walked away" path. Named `cancel` rather than `close`
        because BoltFFI already gives every handle a generated `close()` for
        the host's disposal idiom.
        """
        _native._boltffi_xybrid_streaming_session_cancel(self._handle)

    def is_running(self) -> bool:
        """Whether the session is still accepting audio."""
        return _native._boltffi_xybrid_streaming_session_is_running(self._handle)

    def partials(self) -> "XybridStreamingSessionPartialsSubscription":
        """Pushed partial transcripts, closing once the session ends.

        Generated as an `AsyncStream` in Swift, a `Flow` in Kotlin, an
        `IAsyncEnumerable` in C# and an iterable subscription in Python.

        A partial produced before subscribing is delivered immediately, so
        audio fed before the stream is attached is never silently lost, and
        subscribing to a finished session closes at once instead of hanging.
        """
        return XybridStreamingSessionPartialsSubscription._from_handle(_native.partials(self._handle))


class XybridStreamingSessionPartialsSubscription:
    __slots__ = ("_handle",)

    def __init__(self) -> None:
        raise TypeError("XybridStreamingSessionPartialsSubscription cannot be constructed directly")

    @classmethod
    def _from_handle(cls, handle: int) -> "XybridStreamingSessionPartialsSubscription":
        value = cls.__new__(cls)
        value._handle = handle
        return value

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._handle = None
            _native.partials_free(handle)

    def pop_batch(self, max_count: int = 16) -> list[XybridPartialResult]:
        data = _native.partials_pop_batch(self._require_handle(), max_count)
        return _boltffi_read_wire(data, lambda reader: reader.sequence(lambda: XybridPartialResult._boltffi_from_reader(reader))) if data else []

    def wait(self, timeout_milliseconds: int) -> int:
        return _native.partials_wait(self._require_handle(), timeout_milliseconds)

    def unsubscribe(self) -> None:
        handle = self._require_handle()
        self._handle = None
        _native.partials_unsubscribe(handle)
        _native.partials_free(handle)

    def _require_handle(self) -> int:
        handle = self._handle
        if handle is None:
            raise RuntimeError("stream subscription is closed")
        return handle



class XybridCancellationToken:
    __slots__ = ("_handle",)



    def __init__(self) -> None:
        """Create a fresh, un-cancelled token."""
        self._handle = _native._boltffi_xybrid_cancellation_token_new()



    @classmethod
    def _from_handle(cls, handle: int) -> "XybridCancellationToken":
        value = cls.__new__(cls)
        value._handle = handle
        return value

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._handle = None
            _native._boltffi_xybrid_cancellation_token_release(handle)

    def cancel(self) -> None:
        """Request cancellation. Idempotent, and safe to call from any thread."""
        _native._boltffi_xybrid_cancellation_token_cancel(self._handle)

    def is_cancelled(self) -> bool:
        """Whether [`Self::cancel`] has been called on this token."""
        return _native._boltffi_xybrid_cancellation_token_is_cancelled(self._handle)



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
        """Load from the xybrid registry. Recommended path."""
        return XybridModel._from_handle(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_from_registry(id)))

    @classmethod
    def from_registry_speculative(cls, id: str) -> "XybridModel":
        """Load from the registry, serving from the cloud gateway while the weights
        download in the background.

        Returns almost immediately instead of blocking on the download. Requires
        a resolvable API key and an uncached model; otherwise it behaves exactly
        like `from_registry`. Poll `download_status` for progress and
        `is_cloud_serving` to know which leg is answering. LLM/chat models only.
        """
        return XybridModel._from_handle(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_from_registry_speculative(id)))

    @classmethod
    def from_directory(cls, path: str) -> "XybridModel":
        """Load from a local model directory (must contain `model_metadata.json`)."""
        return XybridModel._from_handle(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_from_directory(path)))

    @classmethod
    def from_bundle(cls, path: str) -> "XybridModel":
        """Load from a local `.xyb` bundle."""
        return XybridModel._from_handle(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_from_bundle(path)))

    @classmethod
    def from_huggingface(cls, repo: str) -> "XybridModel":
        """Resolve and load from a HuggingFace repo (`org/repo` or `org/repo:variant`)."""
        return XybridModel._from_handle(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_from_huggingface(repo)))

    @classmethod
    def from_huggingface_with_revision(cls, repo: str, revision: str) -> "XybridModel":
        """Resolve and load a HuggingFace repository pinned to a revision."""
        return XybridModel._from_handle(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_from_huggingface_with_revision(repo, revision)))

    @classmethod
    def from_model_file(cls, path: str) -> "XybridModel":
        """Load from a raw GGUF file, auto-generating `model_metadata.json` from the
        GGUF header (written next to the file if absent).
        """
        return XybridModel._from_handle(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_from_model_file(path)))

    def model_id(self) -> str:
        return _native._boltffi_xybrid_model_model_id(self._handle)

    def version(self) -> str:
        return _native._boltffi_xybrid_model_version(self._handle)

    def output_type(self) -> XybridOutputType:
        return _native._boltffi_xybrid_model_output_type(self._handle)

    def is_loaded(self) -> bool:
        return _native._boltffi_xybrid_model_is_loaded(self._handle)

    def is_cloud_serving(self) -> bool:
        """Whether runs are currently answered by the cloud because the local
        weights are not ready yet. `false` for ordinary local models.
        """
        return _native._boltffi_xybrid_model_is_cloud_serving(self._handle)

    def download_status(self) -> XybridDownloadStatus:
        """Download progress + state in one read — poll this to drive a progress
        bar. Reports `Ready` at 1.0 for an ordinary local model, so hosts need
        no special case.
        """
        return _boltffi_read_wire(_native._boltffi_xybrid_model_download_status(self._handle), lambda reader: XybridDownloadStatus._boltffi_from_reader(reader))

    def await_download(self, timeout_ms: int) -> XybridDownloadStatus:
        """Block until the download finishes or `timeout_ms` elapses, then report
        the status. Call it off the UI thread (the same place `from_registry` is
        already called). `timeout_ms = 0` makes it a non-blocking read.
        """
        return _boltffi_read_wire(_native._boltffi_xybrid_model_await_download(self._handle, timeout_ms), lambda reader: XybridDownloadStatus._boltffi_from_reader(reader))

    def supports_streaming(self) -> bool:
        return _native._boltffi_xybrid_model_supports_streaming(self._handle)

    def supports_token_streaming(self) -> bool:
        """Whether this model emits true token-by-token output."""
        return _native._boltffi_xybrid_model_supports_token_streaming(self._handle)

    def default_generation_config(self) -> XybridGenerationConfig:
        """Return the model's resolved generation defaults."""
        return _boltffi_read_wire(_native._boltffi_xybrid_model_default_generation_config(self._handle), lambda reader: XybridGenerationConfig._boltffi_from_reader(reader))

    def is_llm(self) -> bool:
        return _native._boltffi_xybrid_model_is_llm(self._handle)

    def supports_tool_calling(self) -> bool | None:
        """Whether the model bundle declares local tool-calling support.

        Advisory tri-state: `null` means the bundle says nothing, so the host
        cannot tell. Gate tool UI on it; enforcement stays at run time — a
        tools-bearing request against a model whose chat template has no tool
        support fails as invalid input regardless of what this reports.
        """
        return _native._boltffi_xybrid_model_supports_tool_calling(self._handle)

    def has_voices(self) -> bool:
        return _native._boltffi_xybrid_model_has_voices(self._handle)

    def voices(self) -> list[XybridVoiceInfo]:
        return _boltffi_read_wire(_native._boltffi_xybrid_model_voices(self._handle), lambda reader: reader.sequence(lambda: XybridVoiceInfo._boltffi_from_reader(reader)))

    def default_voice(self) -> XybridVoiceInfo | None:
        return _boltffi_read_wire(_native._boltffi_xybrid_model_default_voice(self._handle), lambda reader: reader.optional(lambda: XybridVoiceInfo._boltffi_from_reader(reader)))

    def voice(self, voice_id: str) -> XybridVoiceInfo | None:
        return _boltffi_read_wire(_native._boltffi_xybrid_model_voice(self._handle, voice_id), lambda reader: reader.optional(lambda: XybridVoiceInfo._boltffi_from_reader(reader)))

    def run(self, envelope: XybridEnvelope, options: XybridRunOptions | None, cancel: XybridCancellationToken) -> XybridResult:
        """Run inference, optionally with [`XybridRunOptions`] (generation config,
        abort signals, cloud-fallback). Pass `None` for the model's defaults.

        The hand-written wrappers add a one-arg `run(envelope)` convenience that
        forwards `None`, so simple call sites stay ergonomic.
        Pass a [`XybridCancellationToken`] to keep a stop button on the run;
        `None` means the run cannot be cancelled.
        """
        return _boltffi_read_wire(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_run(self._handle, envelope._boltffi_wire(), _boltffi_wire_optional(options, lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()), cancel._handle)), lambda reader: XybridResult._boltffi_from_reader(reader))

    def run_stream(self, envelope: XybridEnvelope, options: XybridRunOptions | None, cancel: XybridCancellationToken) -> int:
        """Start token streaming and return a model-scoped session identifier.

        The identifier remains valid until the final result is taken, an error
        is returned, or [`Self::stream_close`] is called.
        Pass a [`XybridCancellationToken`] to keep a stop button on the run;
        `None` means the run cannot be cancelled.
        """
        return _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_run_stream(self._handle, envelope._boltffi_wire(), _boltffi_wire_optional(options, lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()), cancel._handle))

    def stream_next(self, stream_id: int) -> XybridStreamEvent:
        """Block until the next item for `stream_id` is ready."""
        return _boltffi_read_wire(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_stream_next(self._handle, stream_id)), lambda reader: XybridStreamEvent._boltffi_from_reader(reader))

    def stream_result(self, stream_id: int) -> XybridResult:
        """Take the final result after receiving a `Complete` event."""
        return _boltffi_read_wire(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_stream_result(self._handle, stream_id)), lambda reader: XybridResult._boltffi_from_reader(reader))

    def stream_close(self, stream_id: int) -> None:
        """Forget a streaming session."""
        _native._boltffi_xybrid_model_stream_close(self._handle, stream_id)

    def run_with_context(self, envelope: XybridEnvelope, context: XybridConversationContext, options: XybridRunOptions | None, cancel: XybridCancellationToken) -> XybridResult:
        """Run inference seeded with a conversation `context` (multi-turn chat).

        Only the generation config from `options` is applied — abort signals and
        cloud fallback are not wired on the context path (matches the facade's
        `run_with_context`).
        Pass a [`XybridCancellationToken`] to keep a stop button on the run;
        `None` means the run cannot be cancelled.

        Routes through the facade's options path, so abort signals and cloud
        fallback on `options` are honoured rather than dropped.
        """
        return _boltffi_read_wire(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_run_with_context(self._handle, envelope._boltffi_wire(), context._handle, _boltffi_wire_optional(options, lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()), cancel._handle)), lambda reader: XybridResult._boltffi_from_reader(reader))

    def run_stream_with_context(self, envelope: XybridEnvelope, context: XybridConversationContext, options: XybridRunOptions | None, cancel: XybridCancellationToken) -> int:
        """Start context-aware token streaming; returns a model-scoped session id.
        The pull protocol is identical to [`Self::run_stream`]
        (`stream_next` / `stream_result` / `stream_close`).
        Pass a [`XybridCancellationToken`] to keep a stop button on the run;
        `None` means the run cannot be cancelled.
        """
        return _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_run_stream_with_context(self._handle, envelope._boltffi_wire(), context._handle, _boltffi_wire_optional(options, lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()), cancel._handle))

    def warmup(self) -> None:
        _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_warmup(self._handle))

    def unload(self) -> None:
        _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_model_unload(self._handle))

    def download_progress(self) -> "XybridModelDownloadProgressSubscription":
        """Pushed download updates for a speculatively-loaded model — the stream
        counterpart of [`Self::await_download`], and what issue #504 asks for.

        Emits the current snapshot first, then every update, then closes on
        the terminal state. An ordinary local model is already `Ready`, so its
        stream yields one frame and ends.
        """
        return XybridModelDownloadProgressSubscription._from_handle(_native.download_progress(self._handle))


class XybridModelDownloadProgressSubscription:
    __slots__ = ("_handle",)

    def __init__(self) -> None:
        raise TypeError("XybridModelDownloadProgressSubscription cannot be constructed directly")

    @classmethod
    def _from_handle(cls, handle: int) -> "XybridModelDownloadProgressSubscription":
        value = cls.__new__(cls)
        value._handle = handle
        return value

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._handle = None
            _native.download_progress_free(handle)

    def pop_batch(self, max_count: int = 16) -> list[XybridDownloadStatus]:
        data = _native.download_progress_pop_batch(self._require_handle(), max_count)
        return _boltffi_read_wire(data, lambda reader: reader.sequence(lambda: XybridDownloadStatus._boltffi_from_reader(reader))) if data else []

    def wait(self, timeout_milliseconds: int) -> int:
        return _native.download_progress_wait(self._require_handle(), timeout_milliseconds)

    def unsubscribe(self) -> None:
        handle = self._require_handle()
        self._handle = None
        _native.download_progress_unsubscribe(handle)
        _native.download_progress_free(handle)

    def _require_handle(self) -> int:
        handle = self._handle
        if handle is None:
            raise RuntimeError("stream subscription is closed")
        return handle



class XybridPipeline:
    __slots__ = ("_handle",)


    def __init__(self) -> None:
        raise TypeError("XybridPipeline cannot be constructed directly")


    @classmethod
    def _from_handle(cls, handle: int) -> "XybridPipeline":
        value = cls.__new__(cls)
        value._handle = handle
        return value

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._handle = None
            _native._boltffi_xybrid_pipeline_release(handle)

    @classmethod
    def from_yaml(cls, yaml: str) -> "XybridPipeline":
        """Parse and load a pipeline from YAML content."""
        return XybridPipeline._from_handle(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_pipeline_from_yaml(yaml)))

    @classmethod
    def from_file(cls, path: str) -> "XybridPipeline":
        """Read, parse, and load a pipeline from a YAML file."""
        return XybridPipeline._from_handle(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_pipeline_from_file(path)))

    @classmethod
    def from_bundle(cls, path: str) -> "XybridPipeline":
        """Load a pipeline bundle."""
        return XybridPipeline._from_handle(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_pipeline_from_bundle(path)))

    def run(self, envelope: XybridEnvelope, options: XybridRunOptions | None) -> XybridPipelineResult:
        """Execute every stage, downloading any missing models first, and return
        each stage's output alongside the final one.

        Of `options`, only `correlation_id` applies to a pipeline run. Setting
        `generation_config` or `abort_on` fails with `ConfigError` rather than
        being ignored; per-stage generation settings belong in the YAML.
        """
        return _boltffi_read_wire(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_pipeline_run(self._handle, envelope._boltffi_wire(), _boltffi_wire_optional(options, lambda __boltffi_value_0: __boltffi_value_0._boltffi_wire()))), lambda reader: XybridPipelineResult._boltffi_from_reader(reader))

    def name(self) -> str | None:
        """Pipeline name from the YAML definition, if present."""
        return _boltffi_read_wire(_native._boltffi_xybrid_pipeline_name(self._handle), lambda reader: reader.optional(lambda: reader.string()))

    def stage_names(self) -> list[str]:
        """Stage identifiers in execution order."""
        return _boltffi_read_wire(_native._boltffi_xybrid_pipeline_stage_names(self._handle), lambda reader: reader.sequence(lambda: reader.string()))

    def stage_count(self) -> int:
        """Number of stages in the pipeline."""
        return _native._boltffi_xybrid_pipeline_stage_count(self._handle)



class XybridConversationContext:
    __slots__ = ("_handle",)



    def __init__(self) -> None:
        """Create an empty conversation context (fresh id)."""
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
        """Create a context with a caller-supplied id (for telemetry correlation
        across turns).
        """
        return XybridConversationContext._from_handle(_native._boltffi_xybrid_conversation_context_with_id(id))

    def push(self, envelope: XybridEnvelope) -> None:
        """Append a turn — typically a user or assistant message envelope."""
        _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_conversation_context_push(self._handle, envelope._boltffi_wire()))

    def set_system(self, envelope: XybridEnvelope) -> None:
        """Set the persistent system-prompt envelope (survives [`clear`](Self::clear))."""
        _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_conversation_context_set_system(self._handle, envelope._boltffi_wire()))

    def clear(self) -> None:
        """Drop the history; the system envelope (if any) is preserved."""
        _native._boltffi_xybrid_conversation_context_clear(self._handle)

    def id(self) -> str:
        """The context id."""
        return _native._boltffi_xybrid_conversation_context_id(self._handle)

    def history_len(self) -> int:
        """Number of history turns (excludes the system envelope)."""
        return _native._boltffi_xybrid_conversation_context_history_len(self._handle)

    def history(self) -> list[XybridEnvelope]:
        """Return history turns, excluding the persistent system envelope."""
        return _boltffi_read_wire(_native._boltffi_xybrid_conversation_context_history(self._handle), lambda reader: reader.sequence(lambda: XybridEnvelope._boltffi_from_reader(reader)))

    def has_system(self) -> bool:
        """Whether a persistent system-prompt envelope is set."""
        return _native._boltffi_xybrid_conversation_context_has_system(self._handle)

    def set_max_history_len(self, len: int) -> None:
        """Set the max history length before FIFO pruning."""
        _native._boltffi_xybrid_conversation_context_set_max_history_len(self._handle, len)



class XybridTelemetryConfig:
    __slots__ = ("_handle",)



    def __init__(self, api_key: str) -> None:
        """A new config bound to the default ingest endpoint and the given API key."""
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
        """Override the ingest endpoint (self-hosted collector / non-prod)."""
        _native._boltffi_xybrid_telemetry_config_set_endpoint(self._handle, endpoint)

    def set_app_version(self, version: str) -> None:
        """Set the app version reported with every event."""
        _native._boltffi_xybrid_telemetry_config_set_app_version(self._handle, version)

    def set_device_label(self, label: str) -> None:
        """Set the human-friendly device label reported with every event."""
        _native._boltffi_xybrid_telemetry_config_set_device_label(self._handle, label)

    def set_device_attribute(self, key: str, value: str) -> None:
        """Attach an app-provided device attribute (stored under `device.custom`)."""
        _native._boltffi_xybrid_telemetry_config_set_device_attribute(self._handle, key, value)

    def set_batch_size(self, batch_size: int) -> None:
        """Set the number of events buffered before a flush."""
        _native._boltffi_xybrid_telemetry_config_set_batch_size(self._handle, batch_size)

    def set_flush_interval_secs(self, secs: int) -> None:
        """Set the background flush interval, in seconds."""
        _native._boltffi_xybrid_telemetry_config_set_flush_interval_secs(self._handle, secs)

    def init(self) -> None:
        """Start the process-global telemetry exporter from this config.

        Consumes the config: subsequent setters no-op and a second `init` on the
        same handle errors. Modeled as a method (not a free `telemetry_init`)
        because boltffi 0.25.3 drops free functions that take a handle
        parameter, but lowers a handle self-method fine (same reason the
        generated `run` lives on `XybridModel`).

        # Errors
        Errors if this config was already consumed, or if telemetry is already
        initialized without an intervening [`telemetry_shutdown`].
        """
        _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_telemetry_config_init(self._handle))



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
        """Open and parse a `.xyb` bundle (decompress zstd, parse tar, validate the
        manifest).
        """
        return XybridBundle._from_handle(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_bundle_open(path)))

    def model_id(self) -> str:
        """The model identifier from the manifest."""
        return _native._boltffi_xybrid_bundle_model_id(self._handle)

    def version(self) -> str:
        """The version string from the manifest."""
        return _native._boltffi_xybrid_bundle_version(self._handle)

    def target(self) -> str:
        """The target platform from the manifest."""
        return _native._boltffi_xybrid_bundle_target(self._handle)

    def hash(self) -> str:
        """The SHA-256 hash from the manifest."""
        return _native._boltffi_xybrid_bundle_hash(self._handle)

    def has_metadata(self) -> bool:
        """Whether the bundle carries a `model_metadata.json`."""
        return _native._boltffi_xybrid_bundle_has_metadata(self._handle)

    def file_count(self) -> int:
        """Number of files in the bundle (excludes `manifest.json`)."""
        return _native._boltffi_xybrid_bundle_file_count(self._handle)

    def file_name(self, index: int) -> str | None:
        """The file name at `index`, or `None` if out of bounds."""
        return _boltffi_read_wire(_native._boltffi_xybrid_bundle_file_name(self._handle, index), lambda reader: reader.optional(lambda: reader.string()))

    def manifest_json(self) -> str:
        """The full bundle manifest serialized as JSON."""
        return _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_bundle_manifest_json(self._handle))

    def metadata_json(self) -> str | None:
        """The `model_metadata.json` contents, or `None` if the bundle has none."""
        return _boltffi_read_wire(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_bundle_metadata_json(self._handle)), lambda reader: reader.optional(lambda: reader.string()))

    def extract(self, output_dir: str) -> None:
        """Extract every bundle file to `output_dir` (created if absent)."""
        _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native._boltffi_xybrid_bundle_extract(self._handle, output_dir))






def tool_results_envelope(user_text: str, prior_assistant_text: str, results: Sequence[XybridToolResult]) -> XybridEnvelope:
    """Build the continuation envelope for the turn after the model asked for
    tools.

    One `run` is one model turn, so the loop lives in your code: run a
    tools-bearing request, execute every [`XybridToolCall`] it returns, then
    run this envelope to feed the outcomes back. Pass the same tools on the
    continuation's [`XybridGenerationConfig`] as on the original turn.

    A free function rather than a constructor because `XybridEnvelope` is a
    `#[data]` record, not a handle type — records carry no methods across the
    generated bindings.
    """
    return _boltffi_read_wire(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native.tool_results_envelope(user_text, prior_assistant_text, results)), lambda reader: XybridEnvelope._boltffi_from_reader(reader))
def json_schema_to_gbnf(schema_json: str) -> str:
    """Convert a JSON Schema (as a JSON string) into a GBNF grammar for
    [`XybridGenerationConfig::grammar`]. Fails on invalid JSON or schema
    constructs outside the supported subset.
    """
    return _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native.json_schema_to_gbnf(schema_json))
def set_thermal_state(state: XybridThermalState) -> None:
    _native.set_thermal_state(state)
def clear_thermal_state() -> None:
    _native.clear_thermal_state()
def set_battery_level(percent: int) -> None:
    _native.set_battery_level(percent)
def clear_battery_level() -> None:
    _native.clear_battery_level()
def configure_runtime(api_key: str | None, gateway_url: str | None, ingest_url: str | None) -> None:
    """One-stop SDK initialization: API key + gateway/ingest URL overrides in
    one call. Delegates to [`facade::configure_runtime`]; blank strings are
    treated as absent. This is the canonical init the Swift
    `Xybrid.initialize(apiKey:gatewayUrl:ingestUrl:)` and Kotlin
    `Xybrid.init(context, apiKey, gatewayUrl, ingestUrl)` wrappers call.
    """
    _native.configure_runtime(_boltffi_wire_optional(api_key, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)), _boltffi_wire_optional(gateway_url, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)), _boltffi_wire_optional(ingest_url, lambda __boltffi_value_0: _boltffi_wire_string(__boltffi_value_0)))
def init_sdk_cache_dir(cache_dir: str) -> None:
    _native.init_sdk_cache_dir(cache_dir)
def cache_status() -> XybridCacheStatus:
    """Returns aggregate storage usage across every managed model-cache location."""
    return _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native.cache_status())
def cache_entries() -> list[XybridCacheEntry]:
    """Lists every physical model entry occupying managed cache storage."""
    return _boltffi_read_wire(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native.cache_entries()), lambda reader: reader.sequence(lambda: XybridCacheEntry._boltffi_from_reader(reader)))
def cache_is_model_cached(model_id: str) -> bool:
    """Returns whether a model occupies any managed cache entry."""
    return _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native.cache_is_model_cached(model_id))
def cache_model_path(model_id: str) -> str | None:
    """Resolves the preferred local cache path for a model, if present."""
    return _boltffi_read_wire(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native.cache_model_path(model_id)), lambda reader: reader.optional(lambda: reader.string()))
def cache_list_extracted_model_ids() -> list[str]:
    """Lists model IDs extracted, validated, and ready to run offline."""
    return _boltffi_read_wire(_boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native.cache_list_extracted_model_ids()), lambda reader: reader.sequence(lambda: reader.string()))
def cache_remove_model(model_id: str) -> int:
    """Removes every managed cache entry for one model.

    Do not call concurrently with a load of the same model.
    """
    return _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native.cache_remove_model(model_id))
def cache_clear() -> int:
    """Clears all managed model-cache storage.

    Do not call concurrently with any model load.
    """
    return _boltffi_call(_boltffi_read_09404a3c98b3f16c, lambda: _native.cache_clear())
def set_binding(binding: str) -> None:
    _native.set_binding(binding)
def set_api_key(api_key: str) -> None:
    _native.set_api_key(api_key)
def set_provider_api_key(provider: str, api_key: str) -> None:
    _native.set_provider_api_key(provider, api_key)
def set_platform_url(url: str) -> None:
    """Point the cloud gateway at a platform base URL (staging, self-hosted).
    Pass a bare base URL — the `/v1` suffix is applied internally.
    """
    _native.set_platform_url(url)
def set_speculative_cloud(enabled: bool) -> None:
    """Enable speculative cloud fallback globally: a registry model that isn't
    downloaded yet is served from the gateway while the weights download.

    LLM/chat only — prefer `XybridModel.fromRegistrySpeculative` when the app
    also loads ASR/TTS models, which cannot be served this way.
    """
    _native.set_speculative_cloud(enabled)
def has_api_key() -> bool:
    """Whether a Xybrid gateway API key is resolvable (in-memory or env)."""
    return _native.has_api_key()
def is_speculative_cloud_enabled() -> bool:
    """Whether the global speculative-cloud default is on."""
    return _native.is_speculative_cloud_enabled()
def will_speculate_for_model(model_id: str) -> bool:
    """Whether `XybridModel::from_registry_speculative(model_id)` would actually
    speculate: an API key resolves and the model is not already cached.

    Lets the hand-written Swift/Kotlin loader facades answer "will this
    speculate?" before loading. Never touches the network.
    """
    return _native.will_speculate_for_model(model_id)
def version() -> str:
    """The SDK version string (tracks `CARGO_PKG_VERSION`)."""
    return _native.version()
def release_memory() -> int:
    """Release every idle loaded model's memory; returns how many were released.

    Call this from the platform's low-memory hook (`didReceiveMemoryWarning`
    on iOS, `onTrimMemory` on Android). Models with a run in flight are
    skipped, and a released model reloads itself on next use — no reload call,
    no new error to handle.
    """
    return _native.release_memory()
def set_auto_release(enabled: bool) -> None:
    """Enable or disable automatic model release for subsequent loads.

    When enabled, loading a model under device memory pressure first releases
    least-recently-used idle models. Off by default; [`release_memory`] works
    either way.
    """
    _native.set_auto_release(enabled)
def is_auto_release_enabled() -> bool:
    """Whether automatic model release is enabled process-wide."""
    return _native.is_auto_release_enabled()
def telemetry_default_endpoint() -> str:
    """The SDK's default telemetry ingest endpoint (for display alongside a config)."""
    return _native.telemetry_default_endpoint()
def telemetry_flush() -> None:
    """Flush pending telemetry events. Safe before init / after shutdown."""
    _native.telemetry_flush()
def telemetry_shutdown() -> None:
    """Shut down the telemetry exporter. Idempotent."""
    _native.telemetry_shutdown()

MODULE_NAME = "xybrid_bolt"
PACKAGE_NAME = "xybrid_bolt"
PACKAGE_VERSION = "0.10.0-rc1"

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
    "XybridStageResult",
    "XybridPipelineResult",
    "XybridStreamToken",
    "XybridStreamEvent",
    "XybridVoiceInfo",
    "XybridCacheEntry",
    "XybridCacheStatus",
    "XybridStreamingConfig",
    "XybridPartialResult",
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
    "XybridErrorCancelled",
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
    "XybridCacheEntryLocation",
    "XybridThermalState",
    "XybridVadMode",
    "XybridVadModeOff",
    "XybridVadModeEnabled",
    "XybridDownload",
    "XybridDownloadProgressSubscription",
    "XybridStreamingSession",
    "XybridStreamingSessionPartialsSubscription",
    "XybridCancellationToken",
    "XybridModel",
    "XybridModelDownloadProgressSubscription",
    "XybridPipeline",
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
    "cache_status",
    "cache_entries",
    "cache_is_model_cached",
    "cache_model_path",
    "cache_list_extracted_model_ids",
    "cache_remove_model",
    "cache_clear",
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
