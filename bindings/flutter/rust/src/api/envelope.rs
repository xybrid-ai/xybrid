//! Envelope FFI wrappers for Flutter.
use flutter_rust_bridge::frb;
use std::collections::HashMap;
use xybrid_sdk::ir::{
    Envelope, EnvelopeKind, ImagePlane, PixelFormat, YuvColorInfo, YuvColorMatrix, YuvColorRange,
};

use super::context::FfiMessageRole;

/// Raw pixel-buffer format for [`FfiEnvelope::image_raw`].
///
/// Mirrors `xybrid_core::ir::PixelFormat` 1:1 so camera/canvas frames can be
/// sent as raw pixels without JPEG re-encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfiPixelFormat {
    /// Packed RGB, 8 bits per channel.
    Rgb8,
    /// Packed RGBA, 8 bits per channel.
    Rgba8,
    /// Packed BGRA, 8 bits per channel.
    Bgra8,
    /// Semi-planar YUV 4:2:0 with interleaved UV chroma.
    Nv12,
    /// Semi-planar YUV 4:2:0 with interleaved VU chroma.
    Nv21,
    /// Tri-planar YUV 4:2:0, also known as I420.
    I420,
}

impl From<FfiPixelFormat> for PixelFormat {
    fn from(format: FfiPixelFormat) -> Self {
        match format {
            FfiPixelFormat::Rgb8 => PixelFormat::Rgb8,
            FfiPixelFormat::Rgba8 => PixelFormat::Rgba8,
            FfiPixelFormat::Bgra8 => PixelFormat::Bgra8,
            FfiPixelFormat::Nv12 => PixelFormat::Nv12,
            FfiPixelFormat::Nv21 => PixelFormat::Nv21,
            FfiPixelFormat::I420 => PixelFormat::I420,
        }
    }
}

/// One memory plane inside a raw pixel image.
///
/// Mirrors `xybrid_core::ir::ImagePlane` 1:1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FfiImagePlane {
    /// Byte offset into the raw pixel buffer where this plane begins.
    pub offset: usize,
    /// Bytes between adjacent rows in this plane.
    pub row_stride: usize,
    /// Bytes between adjacent samples in the same row.
    pub pixel_stride: usize,
    /// Plane width in samples. Chroma planes are usually subsampled.
    pub width: u32,
    /// Plane height in samples. Chroma planes are usually subsampled.
    pub height: u32,
}

impl From<FfiImagePlane> for ImagePlane {
    fn from(plane: FfiImagePlane) -> Self {
        ImagePlane {
            offset: plane.offset,
            row_stride: plane.row_stride,
            pixel_stride: plane.pixel_stride,
            width: plane.width,
            height: plane.height,
        }
    }
}

/// YUV color conversion matrix for raw YUV camera frames.
///
/// Mirrors `xybrid_core::ir::YuvColorMatrix` 1:1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfiYuvColorMatrix {
    /// ITU-R BT.601.
    Bt601,
    /// ITU-R BT.709.
    Bt709,
    /// ITU-R BT.2020.
    Bt2020,
}

impl From<FfiYuvColorMatrix> for YuvColorMatrix {
    fn from(matrix: FfiYuvColorMatrix) -> Self {
        match matrix {
            FfiYuvColorMatrix::Bt601 => YuvColorMatrix::Bt601,
            FfiYuvColorMatrix::Bt709 => YuvColorMatrix::Bt709,
            FfiYuvColorMatrix::Bt2020 => YuvColorMatrix::Bt2020,
        }
    }
}

/// YUV luma/chroma numeric range.
///
/// Mirrors `xybrid_core::ir::YuvColorRange` 1:1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfiYuvColorRange {
    /// Video/limited range.
    Limited,
    /// Full range.
    Full,
}

impl From<FfiYuvColorRange> for YuvColorRange {
    fn from(range: FfiYuvColorRange) -> Self {
        match range {
            FfiYuvColorRange::Limited => YuvColorRange::Limited,
            FfiYuvColorRange::Full => YuvColorRange::Full,
        }
    }
}

/// Color metadata required for raw YUV camera frames.
///
/// Mirrors `xybrid_core::ir::YuvColorInfo` 1:1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FfiYuvColorInfo {
    /// Conversion matrix.
    pub matrix: FfiYuvColorMatrix,
    /// Numeric range.
    pub range: FfiYuvColorRange,
}

impl From<FfiYuvColorInfo> for YuvColorInfo {
    fn from(color: FfiYuvColorInfo) -> Self {
        YuvColorInfo {
            matrix: color.matrix.into(),
            range: color.range.into(),
        }
    }
}

/// FFI wrapper for input envelopes.
#[frb(opaque)]
pub struct FfiEnvelope(pub(crate) Envelope);

impl FfiEnvelope {
    /// Create audio envelope with raw bytes and format metadata.
    #[frb(sync)]
    pub fn audio(bytes: Vec<u8>, sample_rate: u32, channels: u32) -> FfiEnvelope {
        let mut metadata = HashMap::new();
        metadata.insert("sample_rate".to_string(), sample_rate.to_string());
        metadata.insert("channels".to_string(), channels.to_string());
        FfiEnvelope(Envelope::with_metadata(
            EnvelopeKind::Audio(bytes),
            metadata,
        ))
    }

    /// Create text envelope for TTS with optional voice and speed.
    #[frb(sync)]
    pub fn text(text: String, voice_id: Option<String>, speed: Option<f64>) -> FfiEnvelope {
        let mut metadata = HashMap::new();
        if let Some(v) = voice_id {
            metadata.insert("voice_id".to_string(), v);
        }
        if let Some(s) = speed {
            metadata.insert("speed".to_string(), s.to_string());
        }
        FfiEnvelope(Envelope::with_metadata(EnvelopeKind::Text(text), metadata))
    }

    /// Create embedding envelope from float vector.
    #[frb(sync)]
    pub fn embedding(data: Vec<f32>) -> FfiEnvelope {
        FfiEnvelope(Envelope::new(EnvelopeKind::Embedding(data)))
    }

    /// Create an encoded image envelope.
    #[frb(sync)]
    pub fn image(bytes: Vec<u8>, format: String) -> Result<FfiEnvelope, String> {
        Envelope::image(bytes, format)
            .map(FfiEnvelope)
            .map_err(|err| err.to_string())
    }

    /// Create a raw pixel image envelope from a camera or canvas frame.
    ///
    /// Maps 1:1 to `Envelope::image_raw`: the FFI-facing format, planes, and
    /// color types are converted to their core counterparts and the core
    /// constructor performs all plane/dimension/color validation.
    #[frb(sync)]
    pub fn image_raw(
        pixels: Vec<u8>,
        pixel_format: FfiPixelFormat,
        width: u32,
        height: u32,
        planes: Vec<FfiImagePlane>,
        color: Option<FfiYuvColorInfo>,
    ) -> Result<FfiEnvelope, String> {
        let planes = planes.into_iter().map(ImagePlane::from).collect();
        let color = color.map(YuvColorInfo::from);
        Envelope::image_raw(pixels, pixel_format.into(), width, height, planes, color)
            .map(FfiEnvelope)
            .map_err(|err| err.to_string())
    }

    /// Create a user-role multi-part envelope with image attachments.
    #[frb(sync)]
    pub fn user_message(text: String, images: Vec<FfiEnvelope>) -> Result<FfiEnvelope, String> {
        let images = images.into_iter().map(|image| image.0).collect();
        Envelope::user_message(text, images)
            .map(FfiEnvelope)
            .map_err(|err| err.to_string())
    }

    /// Create a text envelope with a specific message role.
    ///
    /// This is useful for building conversation context.
    #[frb(sync)]
    pub fn text_with_role(text: String, role: FfiMessageRole) -> FfiEnvelope {
        let envelope = Envelope::new(EnvelopeKind::Text(text)).with_role(role.into());
        FfiEnvelope(envelope)
    }

    /// Set the message role on this envelope.
    ///
    /// Returns a new envelope with the role set.
    #[frb(sync)]
    pub fn with_role(&self, role: FfiMessageRole) -> FfiEnvelope {
        FfiEnvelope(self.0.clone().with_role(role.into()))
    }

    /// Get the message role of this envelope, if set.
    #[frb(sync)]
    pub fn role(&self) -> Option<FfiMessageRole> {
        self.0.role().map(|r| r.into())
    }

    /// Get the unique local ID of this envelope.
    ///
    /// Each envelope has a UUID generated on creation for tracking
    /// and duplicate detection.
    #[frb(sync)]
    pub fn local_id(&self) -> String {
        self.0.local_id().to_string()
    }

    /// Convert to inner Envelope for SDK calls.
    pub(crate) fn into_envelope(self) -> Envelope {
        self.0
    }

    /// Clone the inner envelope (for context operations).
    #[allow(dead_code)]
    pub(crate) fn clone_envelope(&self) -> Envelope {
        self.0.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xybrid_sdk::ir::MessageRole;

    #[test]
    fn image_rejects_unsupported_format() {
        let error = match FfiEnvelope::image(vec![1, 2, 3], "heic".to_string()) {
            Ok(_) => panic!("expected unsupported image format error"),
            Err(error) => error,
        };

        assert!(error.contains("Unsupported image format 'heic'"));
    }

    #[test]
    fn image_rejects_corrupt_bytes_with_redacted_error() {
        let error = match FfiEnvelope::image(vec![42, 42, 42, 42], "jpeg".to_string()) {
            Ok(_) => panic!("expected corrupt image bytes error"),
            Err(error) => error,
        };

        assert!(error.contains("invalid or corrupt jpeg image bytes"));
        assert!(!error.contains("[42"));
        assert!(!error.contains("42, 42"));
    }

    #[test]
    fn image_rejects_oversized_encoded_payload() {
        let bytes = vec![0; xybrid_sdk::ir::envelope::DEFAULT_MAX_ENCODED_IMAGE_BYTES + 1];
        let error = match FfiEnvelope::image(bytes, "png".to_string()) {
            Ok(_) => panic!("expected oversized image payload error"),
            Err(error) => error,
        };

        assert!(error.contains("Image payload too large"));
        assert!(!error.contains("[0"));
    }

    #[test]
    fn image_raw_maps_rgb8_frame_to_core_raw_source() {
        let envelope = FfiEnvelope::image_raw(
            vec![0u8; 2 * 2 * 3],
            FfiPixelFormat::Rgb8,
            2,
            2,
            vec![FfiImagePlane {
                offset: 0,
                row_stride: 6,
                pixel_stride: 3,
                width: 2,
                height: 2,
            }],
            None,
        )
        .expect("rgb8 raw frame should construct");

        let source = envelope
            .0
            .image_source()
            .expect("raw image envelope exposes an image source");
        let raw = source.as_raw().expect("source is raw pixels, not encoded");
        assert_eq!(raw.pixel_format, xybrid_sdk::ir::PixelFormat::Rgb8);
        assert_eq!(raw.dimensions.width, 2);
        assert_eq!(raw.dimensions.height, 2);
        assert_eq!(raw.planes.len(), 1);
        assert_eq!(raw.planes[0].row_stride, 6);
        assert_eq!(raw.planes[0].pixel_stride, 3);
        assert!(raw.color.is_none());
    }

    #[test]
    fn image_raw_maps_nv21_frame_with_color_to_core_raw_source() {
        // 2x2 NV21: 4 luma bytes + 2 interleaved chroma bytes.
        let envelope = FfiEnvelope::image_raw(
            vec![0u8; 6],
            FfiPixelFormat::Nv21,
            2,
            2,
            vec![
                FfiImagePlane {
                    offset: 0,
                    row_stride: 2,
                    pixel_stride: 1,
                    width: 2,
                    height: 2,
                },
                FfiImagePlane {
                    offset: 4,
                    row_stride: 2,
                    pixel_stride: 2,
                    width: 1,
                    height: 1,
                },
            ],
            Some(FfiYuvColorInfo {
                matrix: FfiYuvColorMatrix::Bt601,
                range: FfiYuvColorRange::Limited,
            }),
        )
        .expect("nv21 raw frame should construct");

        let source = envelope
            .0
            .image_source()
            .expect("raw image envelope exposes an image source");
        let raw = source.as_raw().expect("source is raw pixels, not encoded");
        assert_eq!(raw.pixel_format, xybrid_sdk::ir::PixelFormat::Nv21);
        assert_eq!(raw.dimensions.width, 2);
        assert_eq!(raw.dimensions.height, 2);
        assert_eq!(raw.planes.len(), 2);
        let color = raw.color.expect("nv21 carries YUV color metadata");
        assert_eq!(color.matrix, xybrid_sdk::ir::YuvColorMatrix::Bt601);
        assert_eq!(color.range, xybrid_sdk::ir::YuvColorRange::Limited);
    }

    #[test]
    fn user_message_sets_user_role_and_multipart_shape() {
        let envelope = FfiEnvelope::user_message("Describe this image".to_string(), Vec::new())
            .expect("empty image list still produces a user multipart envelope");

        assert_eq!(envelope.0.role(), Some(MessageRole::User));
        match envelope.0.kind {
            EnvelopeKind::MultiPart(parts) => {
                assert_eq!(parts.len(), 1);
                assert_eq!(parts[0].as_text(), Some("Describe this image"));
            }
            other => panic!("expected multipart envelope, got {other:?}"),
        }
    }
}
