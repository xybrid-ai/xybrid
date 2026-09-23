/// xybrid addition — not part of upstream cargokit.
///
/// Precompiled binaries are published twice: as-is and gzip-compressed. A
/// static library compresses about 3.6x (measured: iOS 119 MB -> 32 MB), and
/// the download is what a consumer's first build spends its time on.
///
/// gzip rather than zstd or xz because `dart:io` ships it. build_tool runs on
/// every consumer's machine with pinned dependencies and no native tooling it
/// can rely on, so a codec that needs a package or an external binary is not
/// an option on the consuming side.
library;

import 'dart:io';
import 'dart:typed_data';

/// Appended to an asset's file name for its compressed form.
const compressedExtension = '.gz';

/// Compresses at the highest level: this runs once per release in CI, while
/// the result is downloaded by every consumer.
Uint8List compressArtifact(List<int> bytes) =>
    Uint8List.fromList(GZipCodec(level: ZLibOption.maxLevel).encode(bytes));

/// Reverses [compressArtifact]. Throws a [FormatException] on input that is not
/// valid gzip.
///
/// Callers verify the signature of the compressed bytes *before* calling this,
/// so the decompressor never sees data that did not come from the publisher.
Uint8List decompressArtifact(List<int> bytes) =>
    Uint8List.fromList(gzip.decode(bytes));
