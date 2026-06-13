//! Envelope IR - Typed payload container for pipeline data flow.
//!
//! The Envelope is the Intermediate Representation (IR) that defines how data
//! flows between pipeline stages. It encapsulates typed payloads such as audio,
//! text, or embeddings, along with metadata for routing and telemetry.
//!
//! # Serialization
//!
//! Envelopes are serialized using `bincode` for efficient binary encoding.
//! They can be stored or streamed between local processes or over HTTP to
//! cloud endpoints, maintaining consistent encoding regardless of runtime backend.
//!
//! # Example
//!
//! ```no_run
//! use xybrid_core::ir::{Envelope, EnvelopeKind};
//! use std::collections::HashMap;
//!
//! // Create an audio envelope
//! let mut metadata = HashMap::new();
//! metadata.insert("sample_rate".to_string(), "16000".to_string());
//! let envelope = Envelope {
//!     kind: EnvelopeKind::Audio(vec![0u8; 1024]),
//!     metadata,
//! };
//!
//! // Serialize to bytes
//! let bytes = envelope.to_bytes().unwrap();
//!
//! // Deserialize from bytes
//! let deserialized = Envelope::from_bytes(&bytes).unwrap();
//! ```

#[cfg(feature = "vision")]
use std::sync::Arc;
use std::{collections::HashMap, fmt};
use thiserror::Error;
use uuid::Uuid;

/// Encoded image formats supported by vision envelopes.
#[cfg(feature = "vision")]
#[derive(Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    /// PNG encoded image bytes.
    Png,
    /// JPEG encoded image bytes.
    Jpeg,
    /// WebP encoded image bytes.
    WebP,
}

#[cfg(feature = "vision")]
impl ImageFormat {
    /// Parse an image format from a user-facing hint.
    pub fn from_hint(hint: impl AsRef<str>) -> Result<Self, EnvelopeError> {
        match hint.as_ref().trim().to_ascii_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            "webp" => Ok(Self::WebP),
            other => Err(EnvelopeError::UnsupportedImageFormat {
                format: other.to_string(),
            }),
        }
    }

    /// Return the canonical lowercase format string.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::WebP => "webp",
        }
    }
}

#[cfg(feature = "vision")]
impl fmt::Debug for ImageFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(feature = "vision")]
impl fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Default encoded image byte cap for public image constructors.
#[cfg(feature = "vision")]
pub const DEFAULT_MAX_ENCODED_IMAGE_BYTES: usize = 10 * 1024 * 1024;

/// Default decoded-pixel cap for public image constructors.
#[cfg(feature = "vision")]
pub const DEFAULT_MAX_DECODED_IMAGE_PIXELS: u64 = 16_777_216;

/// Decoded image dimensions discovered during envelope validation.
#[cfg(feature = "vision")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ImageDimensions {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

#[cfg(feature = "vision")]
impl ImageDimensions {
    /// Total decoded pixel count.
    pub fn pixels(self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// Raw pixel-buffer formats accepted by `Envelope::image_raw`.
#[cfg(feature = "vision")]
#[derive(Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelFormat {
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

#[cfg(feature = "vision")]
impl PixelFormat {
    /// Parse a raw pixel format from a binding-facing hint.
    pub fn from_hint(hint: impl AsRef<str>) -> Result<Self, EnvelopeError> {
        let normalized = hint.as_ref().trim().to_ascii_lowercase();
        match normalized.as_str() {
            "rgb8" => Ok(Self::Rgb8),
            "rgba8" => Ok(Self::Rgba8),
            "bgra8" => Ok(Self::Bgra8),
            "nv12" => Ok(Self::Nv12),
            "nv21" => Ok(Self::Nv21),
            "i420" => Ok(Self::I420),
            other => Err(EnvelopeError::UnsupportedPixelFormat {
                format: other.to_string(),
            }),
        }
    }

    /// Canonical lowercase string used in diagnostics and JSON summaries.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rgb8 => "rgb8",
            Self::Rgba8 => "rgba8",
            Self::Bgra8 => "bgra8",
            Self::Nv12 => "nv12",
            Self::Nv21 => "nv21",
            Self::I420 => "i420",
        }
    }

    fn is_yuv(self) -> bool {
        matches!(self, Self::Nv12 | Self::Nv21 | Self::I420)
    }

    fn expected_plane_count(self) -> usize {
        match self {
            Self::Rgb8 | Self::Rgba8 | Self::Bgra8 => 1,
            Self::Nv12 | Self::Nv21 => 2,
            Self::I420 => 3,
        }
    }

    fn expected_plane_specs(self, width: u32, height: u32) -> Vec<(u32, u32, usize)> {
        match self {
            Self::Rgb8 | Self::Rgba8 | Self::Bgra8 => {
                vec![(width, height, usize::from(bytes_per_packed_pixel(self)))]
            }
            Self::Nv12 | Self::Nv21 => {
                let (chroma_width, chroma_height) =
                    PlaneDimensionRole::Chroma420.dimensions(width, height);
                vec![(width, height, 1), (chroma_width, chroma_height, 2)]
            }
            Self::I420 => {
                let (chroma_width, chroma_height) =
                    PlaneDimensionRole::Chroma420.dimensions(width, height);
                vec![
                    (width, height, 1),
                    (chroma_width, chroma_height, 1),
                    (chroma_width, chroma_height, 1),
                ]
            }
        }
    }
}

#[cfg(feature = "vision")]
impl fmt::Debug for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(feature = "vision")]
impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(feature = "vision")]
fn bytes_per_packed_pixel(format: PixelFormat) -> u8 {
    match format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => 4,
        PixelFormat::Nv12 | PixelFormat::Nv21 | PixelFormat::I420 => 1,
    }
}

#[cfg(feature = "vision")]
#[derive(Clone, Copy)]
enum PlaneDimensionRole {
    Chroma420,
}

#[cfg(feature = "vision")]
impl PlaneDimensionRole {
    fn dimensions(self, width: u32, height: u32) -> (u32, u32) {
        match self {
            Self::Chroma420 => (width.div_ceil(2), height.div_ceil(2)),
        }
    }
}

/// One memory plane inside a raw pixel image.
#[cfg(feature = "vision")]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ImagePlane {
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

/// YUV color conversion matrix for raw YUV camera frames.
#[cfg(feature = "vision")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum YuvColorMatrix {
    /// ITU-R BT.601.
    Bt601,
    /// ITU-R BT.709.
    Bt709,
    /// ITU-R BT.2020.
    Bt2020,
}

/// YUV luma/chroma numeric range.
#[cfg(feature = "vision")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum YuvColorRange {
    /// Video/limited range.
    Limited,
    /// Full range.
    Full,
}

/// Color metadata required for raw YUV camera frames.
#[cfg(feature = "vision")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct YuvColorInfo {
    /// Conversion matrix.
    pub matrix: YuvColorMatrix,
    /// Numeric range.
    pub range: YuvColorRange,
}

/// Validation limits used by image envelope constructors.
#[cfg(feature = "vision")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageValidationLimits {
    /// Maximum encoded payload size in bytes.
    pub max_encoded_bytes: usize,
    /// Maximum decoded pixel count.
    pub max_decoded_pixels: u64,
    /// Reject animated formats before decode.
    pub reject_animated: bool,
}

#[cfg(feature = "vision")]
impl Default for ImageValidationLimits {
    fn default() -> Self {
        Self {
            max_encoded_bytes: DEFAULT_MAX_ENCODED_IMAGE_BYTES,
            max_decoded_pixels: DEFAULT_MAX_DECODED_IMAGE_PIXELS,
            reject_animated: true,
        }
    }
}

#[cfg(feature = "vision")]
impl ImageValidationLimits {
    /// Override the encoded-byte cap.
    pub fn with_max_encoded_bytes(mut self, max_encoded_bytes: usize) -> Self {
        self.max_encoded_bytes = max_encoded_bytes;
        self
    }

    /// Override the decoded-pixel cap.
    pub fn with_max_decoded_pixels(mut self, max_decoded_pixels: u64) -> Self {
        self.max_decoded_pixels = max_decoded_pixels;
        self
    }

    /// Override animated-image handling.
    pub fn with_reject_animated(mut self, reject_animated: bool) -> Self {
        self.reject_animated = reject_animated;
        self
    }
}

/// Image payload source.
///
/// INF-230 only constructs encoded images, but this enum leaves the envelope
/// shape ready for raw camera-frame ingress without another enum-breaking
/// refactor.
#[cfg(feature = "vision")]
#[derive(Clone, PartialEq)]
pub enum ImageSource {
    /// Container-encoded bytes that still need image decoding.
    Encoded {
        /// PNG/JPEG/WebP bytes.
        bytes: Arc<[u8]>,
        /// Declared encoded format.
        format: ImageFormat,
        /// Decoded dimensions validated from the encoded payload.
        dimensions: ImageDimensions,
    },
    /// Pre-decoded raw pixels from a camera frame or canvas.
    Raw {
        /// Backing memory for all planes.
        pixels: Arc<[u8]>,
        /// Raw pixel memory layout.
        pixel_format: PixelFormat,
        /// Image dimensions.
        dimensions: ImageDimensions,
        /// Plane descriptors into `pixels`.
        planes: Vec<ImagePlane>,
        /// Required for YUV formats, absent for RGB-family formats.
        color: Option<YuvColorInfo>,
    },
}

/// Borrowed view of a raw image payload.
#[cfg(feature = "vision")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawImageRef<'a> {
    /// Raw backing pixels.
    pub pixels: &'a [u8],
    /// Raw pixel format.
    pub pixel_format: PixelFormat,
    /// Image dimensions.
    pub dimensions: ImageDimensions,
    /// Plane descriptors into `pixels`.
    pub planes: &'a [ImagePlane],
    /// YUV color metadata when applicable.
    pub color: Option<YuvColorInfo>,
}

/// Byte-free summary of an image envelope.
#[cfg(feature = "vision")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ImageSummary {
    /// Number of bytes carried by the image source.
    pub byte_len: usize,
    /// Validated image dimensions.
    pub dimensions: ImageDimensions,
    /// Encoded or raw source metadata.
    pub source: ImageSummarySource,
}

/// Byte-free source metadata for an image summary.
#[cfg(feature = "vision")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageSummarySource {
    /// Container-encoded image bytes.
    Encoded { format: ImageFormat },
    /// Raw pixel memory layout.
    Raw {
        pixel_format: PixelFormat,
        plane_count: usize,
    },
}

#[cfg(feature = "vision")]
impl ImageSummarySource {
    /// Encoded image format, when this summary describes encoded bytes.
    pub fn as_encoded(self) -> Option<ImageFormat> {
        match self {
            Self::Encoded { format } => Some(format),
            Self::Raw { .. } => None,
        }
    }

    /// Raw pixel format and plane count, when this summary describes raw pixels.
    pub fn as_raw(self) -> Option<(PixelFormat, usize)> {
        match self {
            Self::Raw {
                pixel_format,
                plane_count,
            } => Some((pixel_format, plane_count)),
            Self::Encoded { .. } => None,
        }
    }
}

#[cfg(feature = "vision")]
impl ImageSource {
    /// Number of bytes carried by this image source.
    pub fn byte_len(&self) -> usize {
        match self {
            Self::Encoded { bytes, .. } => bytes.len(),
            Self::Raw { pixels, .. } => pixels.len(),
        }
    }

    /// Declared encoded image format when the source is encoded bytes.
    pub fn encoded_format(&self) -> Option<ImageFormat> {
        match self {
            Self::Encoded { format, .. } => Some(*format),
            Self::Raw { .. } => None,
        }
    }

    /// Encoded bytes and format when the source is encoded bytes.
    pub fn as_encoded(&self) -> Option<(&[u8], ImageFormat)> {
        match self {
            Self::Encoded { bytes, format, .. } => Some((bytes, *format)),
            Self::Raw { .. } => None,
        }
    }

    /// Raw pixel payload and layout when the source is raw pixels.
    pub fn as_raw(&self) -> Option<RawImageRef<'_>> {
        match self {
            Self::Raw {
                pixels,
                pixel_format,
                dimensions,
                planes,
                color,
            } => Some(RawImageRef {
                pixels,
                pixel_format: *pixel_format,
                dimensions: *dimensions,
                planes,
                color: *color,
            }),
            Self::Encoded { .. } => None,
        }
    }

    /// Re-run the shared encoded-image guardrails before handing bytes to
    /// preprocessing. This keeps forged or non-human deserialized image
    /// sources from bypassing the public `Envelope::image` constructors.
    pub(crate) fn validated_encoded(
        &self,
        limits: ImageValidationLimits,
    ) -> EnvelopeResult<(&[u8], ImageFormat, ImageDimensions)> {
        match self {
            Self::Encoded { bytes, format, .. } => {
                let dimensions = validate_encoded_image(bytes, *format, limits)?;
                Ok((bytes.as_ref(), *format, dimensions))
            }
            Self::Raw { .. } => Err(EnvelopeError::ValidationError(
                "encoded image source required".to_string(),
            )),
        }
    }

    /// Re-run raw-image guardrails before raw pixels enter preprocessing.
    /// This mirrors `validated_encoded` for manually constructed or
    /// deserialized image sources.
    pub(crate) fn validated_raw(
        &self,
        limits: ImageValidationLimits,
    ) -> EnvelopeResult<RawImageRef<'_>> {
        match self {
            Self::Raw {
                pixels,
                pixel_format,
                dimensions,
                planes,
                color,
            } => {
                validate_raw_image(
                    pixels,
                    *pixel_format,
                    dimensions.width,
                    dimensions.height,
                    planes,
                    *color,
                    limits,
                )?;
                Ok(RawImageRef {
                    pixels,
                    pixel_format: *pixel_format,
                    dimensions: *dimensions,
                    planes,
                    color: *color,
                })
            }
            Self::Encoded { .. } => Err(EnvelopeError::ValidationError(
                "raw image source required".to_string(),
            )),
        }
    }

    /// Decoded dimensions discovered during validation.
    pub fn dimensions(&self) -> Option<ImageDimensions> {
        match self {
            Self::Encoded { dimensions, .. } => Some(*dimensions),
            Self::Raw { dimensions, .. } => Some(*dimensions),
        }
    }

    /// Return a byte-free summary suitable for diagnostics and telemetry.
    pub fn summary(&self) -> ImageSummary {
        match self {
            Self::Encoded {
                bytes,
                format,
                dimensions,
            } => ImageSummary {
                byte_len: bytes.len(),
                dimensions: *dimensions,
                source: ImageSummarySource::Encoded { format: *format },
            },
            Self::Raw {
                pixels,
                pixel_format,
                dimensions,
                planes,
                ..
            } => ImageSummary {
                byte_len: pixels.len(),
                dimensions: *dimensions,
                source: ImageSummarySource::Raw {
                    pixel_format: *pixel_format,
                    plane_count: planes.len(),
                },
            },
        }
    }
}

#[cfg(feature = "vision")]
impl fmt::Debug for ImageSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encoded {
                bytes,
                format,
                dimensions,
            } => {
                write!(
                    f,
                    "EncodedImage({} bytes, {}, {}x{})",
                    bytes.len(),
                    format,
                    dimensions.width,
                    dimensions.height
                )
            }
            Self::Raw {
                pixels,
                pixel_format,
                dimensions,
                planes,
                ..
            } => {
                write!(
                    f,
                    "RawImage({} bytes, {}, {}x{}, {} planes)",
                    pixels.len(),
                    pixel_format,
                    dimensions.width,
                    dimensions.height,
                    planes.len()
                )
            }
        }
    }
}

#[cfg(feature = "vision")]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum ImageSourceWire {
    Encoded {
        bytes: Vec<u8>,
        format: ImageFormat,
        dimensions: ImageDimensions,
    },
    Raw {
        pixels: Vec<u8>,
        pixel_format: PixelFormat,
        dimensions: ImageDimensions,
        planes: Vec<ImagePlane>,
        color: Option<YuvColorInfo>,
    },
}

#[cfg(feature = "vision")]
impl From<&ImageSource> for ImageSourceWire {
    fn from(source: &ImageSource) -> Self {
        match source {
            ImageSource::Encoded {
                bytes,
                format,
                dimensions,
            } => Self::Encoded {
                bytes: bytes.to_vec(),
                format: *format,
                dimensions: *dimensions,
            },
            ImageSource::Raw {
                pixels,
                pixel_format,
                dimensions,
                planes,
                color,
            } => Self::Raw {
                pixels: pixels.to_vec(),
                pixel_format: *pixel_format,
                dimensions: *dimensions,
                planes: planes.clone(),
                color: *color,
            },
        }
    }
}

#[cfg(feature = "vision")]
impl From<ImageSourceWire> for ImageSource {
    fn from(source: ImageSourceWire) -> Self {
        match source {
            ImageSourceWire::Encoded {
                bytes,
                format,
                dimensions,
            } => Self::Encoded {
                bytes: Arc::from(bytes),
                format,
                dimensions,
            },
            ImageSourceWire::Raw {
                pixels,
                pixel_format,
                dimensions,
                planes,
                color,
            } => Self::Raw {
                pixels: Arc::from(pixels),
                pixel_format,
                dimensions,
                planes,
                color,
            },
        }
    }
}

#[cfg(feature = "vision")]
impl serde::Serialize for ImageSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if serializer.is_human_readable() {
            #[derive(serde::Serialize)]
            #[serde(rename_all = "snake_case")]
            enum HumanImageSource {
                Encoded {
                    byte_len: usize,
                    format: ImageFormat,
                    width: u32,
                    height: u32,
                },
                Raw {
                    byte_len: usize,
                    pixel_format: PixelFormat,
                    width: u32,
                    height: u32,
                    plane_count: usize,
                    color: Option<YuvColorInfo>,
                },
            }

            match self {
                Self::Encoded {
                    bytes,
                    format,
                    dimensions,
                } => serde::Serialize::serialize(
                    &HumanImageSource::Encoded {
                        byte_len: bytes.len(),
                        format: *format,
                        width: dimensions.width,
                        height: dimensions.height,
                    },
                    serializer,
                ),
                Self::Raw {
                    pixels,
                    pixel_format,
                    dimensions,
                    planes,
                    color,
                } => serde::Serialize::serialize(
                    &HumanImageSource::Raw {
                        byte_len: pixels.len(),
                        pixel_format: *pixel_format,
                        width: dimensions.width,
                        height: dimensions.height,
                        plane_count: planes.len(),
                        color: *color,
                    },
                    serializer,
                ),
            }
        } else {
            serde::Serialize::serialize(&ImageSourceWire::from(self), serializer)
        }
    }
}

#[cfg(feature = "vision")]
impl<'de> serde::Deserialize<'de> for ImageSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            #[derive(serde::Deserialize)]
            #[serde(rename_all = "snake_case")]
            enum HumanImageSource {
                Encoded {
                    bytes: Option<Vec<u8>>,
                    format: ImageFormat,
                },
                Raw {
                    pixels: Option<Vec<u8>>,
                    pixel_format: PixelFormat,
                    width: u32,
                    height: u32,
                    planes: Vec<ImagePlane>,
                    color: Option<YuvColorInfo>,
                },
            }

            match <HumanImageSource as serde::Deserialize>::deserialize(deserializer)? {
                HumanImageSource::Encoded {
                    bytes: Some(bytes),
                    format,
                } => {
                    let dimensions =
                        validate_encoded_image(&bytes, format, ImageValidationLimits::default())
                            .map_err(serde::de::Error::custom)?;
                    Ok(Self::Encoded {
                        bytes: Arc::from(bytes),
                        format,
                        dimensions,
                    })
                }
                HumanImageSource::Encoded { bytes: None, .. } => Err(serde::de::Error::custom(
                    "redacted image JSON cannot be deserialized back into image bytes",
                )),
                HumanImageSource::Raw {
                    pixels: Some(pixels),
                    pixel_format,
                    width,
                    height,
                    planes,
                    color,
                } => {
                    let dimensions = validate_raw_image(
                        &pixels,
                        pixel_format,
                        width,
                        height,
                        &planes,
                        color,
                        ImageValidationLimits::default(),
                    )
                    .map_err(serde::de::Error::custom)?;
                    Ok(Self::Raw {
                        pixels: Arc::from(pixels),
                        pixel_format,
                        dimensions,
                        planes,
                        color,
                    })
                }
                HumanImageSource::Raw { pixels: None, .. } => Err(serde::de::Error::custom(
                    "redacted raw image JSON cannot be deserialized back into pixel bytes",
                )),
            }
        } else {
            Ok(<ImageSourceWire as serde::Deserialize>::deserialize(deserializer)?.into())
        }
    }
}

#[cfg(feature = "vision")]
fn validate_encoded_image(
    bytes: &[u8],
    format: ImageFormat,
    limits: ImageValidationLimits,
) -> Result<ImageDimensions, EnvelopeError> {
    if bytes.len() > limits.max_encoded_bytes {
        return Err(EnvelopeError::ImageEncodedTooLarge {
            byte_len: bytes.len(),
            max_bytes: limits.max_encoded_bytes,
        });
    }

    if limits.reject_animated && format == ImageFormat::WebP && webp_declares_animation(bytes) {
        return Err(EnvelopeError::AnimatedImageUnsupported { format });
    }

    let dimensions = read_encoded_image_dimensions(bytes, format)?;
    let pixels = dimensions.pixels();
    if pixels > limits.max_decoded_pixels {
        return Err(EnvelopeError::ImageDimensionsTooLarge {
            width: dimensions.width,
            height: dimensions.height,
            pixels,
            max_pixels: limits.max_decoded_pixels,
        });
    }

    ensure_encoded_image_decodes(bytes, format)?;
    Ok(dimensions)
}

#[cfg(feature = "vision")]
fn validate_raw_image(
    pixels: &[u8],
    pixel_format: PixelFormat,
    width: u32,
    height: u32,
    planes: &[ImagePlane],
    color: Option<YuvColorInfo>,
    limits: ImageValidationLimits,
) -> Result<ImageDimensions, EnvelopeError> {
    if width == 0 || height == 0 {
        return Err(EnvelopeError::ValidationError(
            "raw image dimensions must be non-zero".to_string(),
        ));
    }

    let dimensions = ImageDimensions { width, height };
    let decoded_pixels = dimensions.pixels();
    if decoded_pixels > limits.max_decoded_pixels {
        return Err(EnvelopeError::ImageDimensionsTooLarge {
            width,
            height,
            pixels: decoded_pixels,
            max_pixels: limits.max_decoded_pixels,
        });
    }

    if pixel_format.is_yuv() {
        if color.is_none() {
            return Err(EnvelopeError::RawImageColorMetadataRequired { pixel_format });
        }
    } else if color.is_some() {
        return Err(EnvelopeError::RawImageColorMetadataUnsupported { pixel_format });
    }

    let expected_plane_count = pixel_format.expected_plane_count();
    if planes.len() != expected_plane_count {
        return Err(EnvelopeError::RawImagePlaneCountMismatch {
            pixel_format,
            expected: expected_plane_count,
            actual: planes.len(),
        });
    }

    let expected_specs = pixel_format.expected_plane_specs(width, height);
    for (plane_index, (plane, (expected_width, expected_height, expected_pixel_stride))) in
        planes.iter().zip(expected_specs.iter()).enumerate()
    {
        validate_raw_plane(
            pixels,
            plane,
            plane_index,
            *expected_width,
            *expected_height,
            *expected_pixel_stride,
        )?;
    }

    Ok(dimensions)
}

#[cfg(feature = "vision")]
fn validate_raw_plane(
    pixels: &[u8],
    plane: &ImagePlane,
    plane_index: usize,
    expected_width: u32,
    expected_height: u32,
    expected_pixel_stride: usize,
) -> Result<(), EnvelopeError> {
    if plane.width != expected_width || plane.height != expected_height {
        return Err(EnvelopeError::RawImagePlaneInvalid {
            plane_index,
            reason: format!(
                "expected {}x{} plane, got {}x{}",
                expected_width, expected_height, plane.width, plane.height
            ),
        });
    }

    if plane.pixel_stride != expected_pixel_stride {
        return Err(EnvelopeError::RawImagePlaneInvalid {
            plane_index,
            reason: format!(
                "expected pixel stride {}, got {}",
                expected_pixel_stride, plane.pixel_stride
            ),
        });
    }

    if plane.pixel_stride == 0 {
        return Err(EnvelopeError::RawImagePlaneInvalid {
            plane_index,
            reason: "pixel_stride must be non-zero".to_string(),
        });
    }

    if plane.width == 0 || plane.height == 0 {
        return Err(EnvelopeError::RawImagePlaneInvalid {
            plane_index,
            reason: "plane dimensions must be non-zero".to_string(),
        });
    }

    let width = plane.width as usize;
    let height = plane.height as usize;
    let min_row_stride = width.checked_mul(plane.pixel_stride).ok_or_else(|| {
        EnvelopeError::RawImagePlaneInvalid {
            plane_index,
            reason: "plane row size overflows usize".to_string(),
        }
    })?;

    if plane.row_stride < min_row_stride {
        return Err(EnvelopeError::RawImagePlaneInvalid {
            plane_index,
            reason: format!(
                "row_stride {} is smaller than width * pixel_stride {}",
                plane.row_stride, min_row_stride
            ),
        });
    }

    let last_row_offset = height
        .checked_sub(1)
        .and_then(|rows| rows.checked_mul(plane.row_stride))
        .ok_or_else(|| EnvelopeError::RawImagePlaneInvalid {
            plane_index,
            reason: "plane row extent overflows usize".to_string(),
        })?;
    let plane_extent = last_row_offset.checked_add(min_row_stride).ok_or_else(|| {
        EnvelopeError::RawImagePlaneInvalid {
            plane_index,
            reason: "plane byte extent overflows usize".to_string(),
        }
    })?;
    let end = plane.offset.checked_add(plane_extent).ok_or_else(|| {
        EnvelopeError::RawImagePlaneInvalid {
            plane_index,
            reason: "plane offset plus extent overflows usize".to_string(),
        }
    })?;

    if end > pixels.len() {
        return Err(EnvelopeError::RawImagePlaneInvalid {
            plane_index,
            reason: format!(
                "plane extent ends at byte {}, beyond pixel buffer length {}",
                end,
                pixels.len()
            ),
        });
    }

    Ok(())
}

#[cfg(feature = "vision")]
fn read_encoded_image_dimensions(
    bytes: &[u8],
    format: ImageFormat,
) -> Result<ImageDimensions, EnvelopeError> {
    use image::ImageDecoder as _;

    let reader =
        image::ImageReader::with_format(std::io::Cursor::new(bytes), image_crate_format(format));
    let decoder = reader
        .into_decoder()
        .map_err(|_| EnvelopeError::ImageDecodeFailed { format })?;
    let (width, height) = decoder.dimensions();
    Ok(ImageDimensions { width, height })
}

#[cfg(feature = "vision")]
fn ensure_encoded_image_decodes(bytes: &[u8], format: ImageFormat) -> Result<(), EnvelopeError> {
    image::ImageReader::with_format(std::io::Cursor::new(bytes), image_crate_format(format))
        .decode()
        .map(|_| ())
        .map_err(|_| EnvelopeError::ImageDecodeFailed { format })
}

#[cfg(feature = "vision")]
fn image_crate_format(format: ImageFormat) -> image::ImageFormat {
    match format {
        ImageFormat::Png => image::ImageFormat::Png,
        ImageFormat::Jpeg => image::ImageFormat::Jpeg,
        ImageFormat::WebP => image::ImageFormat::WebP,
    }
}

#[cfg(feature = "vision")]
fn webp_declares_animation(bytes: &[u8]) -> bool {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return false;
    }

    let mut offset = 12usize;
    while offset + 8 <= bytes.len() {
        let chunk = &bytes[offset..offset + 4];
        if chunk == b"ANIM" || chunk == b"ANMF" {
            return true;
        }

        let size = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        let padded_size = size + (size % 2);
        match offset
            .checked_add(8)
            .and_then(|next| next.checked_add(padded_size))
        {
            Some(next) => offset = next,
            None => return false,
        }
    }

    false
}

/// Typed payload variants for envelope data.
///
/// Each variant represents a different data type that can flow through
/// the pipeline stages.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum EnvelopeKind {
    /// Raw audio data (PCM samples, WAV bytes, etc.)
    Audio(Vec<u8>),
    /// Text data (transcriptions, LLM outputs, etc.)
    Text(String),
    /// Embedding vectors (feature vectors, embeddings, etc.)
    Embedding(Vec<f32>),
    /// Image payload for vision-capable models.
    #[cfg(feature = "vision")]
    Image { source: ImageSource },
    /// Ordered envelope fragments that represent one logical message.
    #[cfg(feature = "vision")]
    MultiPart(Vec<Envelope>),
}

impl fmt::Debug for EnvelopeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnvelopeKind::Audio(data) => f.debug_tuple("Audio").field(data).finish(),
            EnvelopeKind::Text(data) => f.debug_tuple("Text").field(data).finish(),
            EnvelopeKind::Embedding(data) => f.debug_tuple("Embedding").field(data).finish(),
            #[cfg(feature = "vision")]
            EnvelopeKind::Image { source } => match source {
                ImageSource::Encoded {
                    bytes,
                    format,
                    dimensions,
                } => {
                    write!(
                        f,
                        "Image({} bytes, {}, {}x{})",
                        bytes.len(),
                        format,
                        dimensions.width,
                        dimensions.height
                    )
                }
                ImageSource::Raw {
                    pixels,
                    pixel_format,
                    dimensions,
                    planes,
                    ..
                } => {
                    write!(
                        f,
                        "Image({} bytes, {}, {}x{}, {} planes)",
                        pixels.len(),
                        pixel_format,
                        dimensions.width,
                        dimensions.height,
                        planes.len()
                    )
                }
            },
            #[cfg(feature = "vision")]
            EnvelopeKind::MultiPart(parts) => f.debug_tuple("MultiPart").field(parts).finish(),
        }
    }
}

impl EnvelopeKind {
    /// Returns a string representation of the envelope kind.
    ///
    /// # Returns
    ///
    /// A string describing the variant (e.g., "Audio", "Text", "Embedding")
    pub fn as_str(&self) -> &'static str {
        match self {
            EnvelopeKind::Audio(_) => "Audio",
            EnvelopeKind::Text(_) => "Text",
            EnvelopeKind::Embedding(_) => "Embedding",
            #[cfg(feature = "vision")]
            EnvelopeKind::Image { .. } => "Image",
            #[cfg(feature = "vision")]
            EnvelopeKind::MultiPart(_) => "MultiPart",
        }
    }

    /// Returns the size of the payload in bytes (approximate).
    ///
    /// For Audio, returns the length of the byte vector.
    /// For Text, returns the byte length of the string.
    /// For Embedding, returns the byte length of the float vector.
    pub fn payload_size(&self) -> usize {
        match self {
            EnvelopeKind::Audio(data) => data.len(),
            EnvelopeKind::Text(data) => data.len(),
            EnvelopeKind::Embedding(data) => data.len() * std::mem::size_of::<f32>(),
            #[cfg(feature = "vision")]
            EnvelopeKind::Image { source } => source.byte_len(),
            #[cfg(feature = "vision")]
            EnvelopeKind::MultiPart(parts) => parts.iter().map(Envelope::payload_size).sum(),
        }
    }
}

/// Data payload envelope containing inference inputs/outputs.
///
/// Envelopes are the primary data structure for passing data between
/// pipeline stages. They encapsulate typed payloads and metadata for
/// routing, telemetry, and processing hints.
///
/// # Serialization
///
/// Envelopes can be serialized to binary format using `bincode` for
/// efficient transmission and storage.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Envelope {
    /// The typed payload data
    pub kind: EnvelopeKind,
    /// Metadata key-value pairs for routing, telemetry, and processing hints
    pub metadata: HashMap<String, String>,
}

impl Envelope {
    /// Metadata key for storing the local unique ID.
    pub const LOCAL_ID_METADATA_KEY: &'static str = "xybrid.local_id";

    /// Creates a new envelope with the specified kind and empty metadata.
    ///
    /// A unique local ID is automatically generated for tracking and
    /// duplicate detection.
    ///
    /// # Arguments
    ///
    /// * `kind` - The envelope kind (Audio, Text, or Embedding)
    ///
    /// # Returns
    ///
    /// A new `Envelope` instance with a unique local ID
    pub fn new(kind: EnvelopeKind) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert(
            Self::LOCAL_ID_METADATA_KEY.to_string(),
            Uuid::new_v4().to_string(),
        );
        Self { kind, metadata }
    }

    /// Creates a new envelope with the specified kind and metadata.
    ///
    /// If the metadata does not contain a local ID, one is automatically generated.
    ///
    /// # Arguments
    ///
    /// * `kind` - The envelope kind (Audio, Text, or Embedding)
    /// * `metadata` - Metadata key-value pairs
    ///
    /// # Returns
    ///
    /// A new `Envelope` instance with a unique local ID
    pub fn with_metadata(kind: EnvelopeKind, mut metadata: HashMap<String, String>) -> Self {
        // Ensure a local ID exists
        if !metadata.contains_key(Self::LOCAL_ID_METADATA_KEY) {
            metadata.insert(
                Self::LOCAL_ID_METADATA_KEY.to_string(),
                Uuid::new_v4().to_string(),
            );
        }
        Self { kind, metadata }
    }

    /// Creates an encoded image envelope.
    #[cfg(feature = "vision")]
    pub fn image(bytes: Vec<u8>, format: impl AsRef<str>) -> EnvelopeResult<Self> {
        Self::image_with_limits(bytes, format, ImageValidationLimits::default())
    }

    /// Creates an encoded image envelope using explicit validation limits.
    #[cfg(feature = "vision")]
    pub fn image_with_limits(
        bytes: Vec<u8>,
        format: impl AsRef<str>,
        limits: ImageValidationLimits,
    ) -> EnvelopeResult<Self> {
        let format = ImageFormat::from_hint(format)?;
        let dimensions = validate_encoded_image(&bytes, format, limits)?;
        Ok(Self::new(EnvelopeKind::Image {
            source: ImageSource::Encoded {
                bytes: Arc::from(bytes),
                format,
                dimensions,
            },
        }))
    }

    /// Creates a raw pixel image envelope using default validation limits.
    #[cfg(feature = "vision")]
    pub fn image_raw(
        pixels: Vec<u8>,
        pixel_format: PixelFormat,
        width: u32,
        height: u32,
        planes: Vec<ImagePlane>,
        color: Option<YuvColorInfo>,
    ) -> EnvelopeResult<Self> {
        Self::image_raw_with_limits(
            pixels,
            pixel_format,
            width,
            height,
            planes,
            color,
            ImageValidationLimits::default(),
        )
    }

    /// Creates a raw pixel image envelope using explicit validation limits.
    #[cfg(feature = "vision")]
    pub fn image_raw_with_limits(
        pixels: Vec<u8>,
        pixel_format: PixelFormat,
        width: u32,
        height: u32,
        planes: Vec<ImagePlane>,
        color: Option<YuvColorInfo>,
        limits: ImageValidationLimits,
    ) -> EnvelopeResult<Self> {
        let dimensions =
            validate_raw_image(&pixels, pixel_format, width, height, &planes, color, limits)?;
        Ok(Self::new(EnvelopeKind::Image {
            source: ImageSource::Raw {
                pixels: Arc::from(pixels),
                pixel_format,
                dimensions,
                planes,
                color,
            },
        }))
    }

    /// Creates a user-role multi-part envelope from text and image attachments.
    #[cfg(feature = "vision")]
    pub fn user_message(text: impl Into<String>, images: Vec<Envelope>) -> EnvelopeResult<Self> {
        if images.iter().any(|image| !image.is_image()) {
            return Err(EnvelopeError::ValidationError(
                "user_message image attachments must all be image envelopes".to_string(),
            ));
        }

        let mut parts = Vec::with_capacity(images.len() + 1);
        parts.push(Self::new(EnvelopeKind::Text(text.into())));
        parts.extend(images);

        Ok(Self::new(EnvelopeKind::MultiPart(parts)).with_role(super::MessageRole::User))
    }

    /// Returns the unique local ID of this envelope.
    ///
    /// Each envelope gets a UUID on creation for tracking and duplicate detection.
    ///
    /// # Returns
    ///
    /// The local ID string, or an empty string if somehow missing
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::ir::{Envelope, EnvelopeKind};
    ///
    /// let e1 = Envelope::new(EnvelopeKind::Text("Hello".to_string()));
    /// let e2 = Envelope::new(EnvelopeKind::Text("Hello".to_string()));
    ///
    /// // Each envelope has a unique ID even with identical content
    /// assert_ne!(e1.local_id(), e2.local_id());
    /// ```
    pub fn local_id(&self) -> &str {
        self.metadata
            .get(Self::LOCAL_ID_METADATA_KEY)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Sets a custom local ID for this envelope (builder pattern).
    ///
    /// Useful for testing or when resuming from serialized state.
    ///
    /// # Arguments
    ///
    /// * `id` - The custom local ID
    ///
    /// # Returns
    ///
    /// Self with the custom local ID set
    pub fn with_local_id(mut self, id: impl Into<String>) -> Self {
        self.metadata
            .insert(Self::LOCAL_ID_METADATA_KEY.to_string(), id.into());
        self
    }

    /// Adds a metadata key-value pair.
    ///
    /// # Arguments
    ///
    /// * `key` - Metadata key
    /// * `value` - Metadata value
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Gets a metadata value by key.
    ///
    /// # Arguments
    ///
    /// * `key` - Metadata key
    ///
    /// # Returns
    ///
    /// `Some(value)` if the key exists, `None` otherwise
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    // =========================================================================
    // Message Role Helpers (for conversation/chat contexts)
    // =========================================================================

    /// Metadata key for storing the message role.
    pub const ROLE_METADATA_KEY: &'static str = "xybrid.role";

    /// Sets the message role for this envelope and returns self (builder pattern).
    ///
    /// Stores the role under the `xybrid.role` metadata key.
    ///
    /// # Arguments
    ///
    /// * `role` - The message role (System, User, or Assistant)
    ///
    /// # Returns
    ///
    /// Self with the role set
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let envelope = Envelope::new(EnvelopeKind::Text("Hello".to_string()))
    ///     .with_role(MessageRole::User);
    /// assert_eq!(envelope.role(), Some(MessageRole::User));
    /// ```
    pub fn with_role(mut self, role: super::MessageRole) -> Self {
        self.metadata.insert(
            Self::ROLE_METADATA_KEY.to_string(),
            role.as_str().to_string(),
        );
        self
    }

    /// Gets the message role of this envelope.
    ///
    /// Reads the role from the `xybrid.role` metadata key.
    ///
    /// # Returns
    ///
    /// `Some(MessageRole)` if a valid role is set, `None` otherwise
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let envelope = Envelope::new(EnvelopeKind::Text("Hello".to_string()))
    ///     .with_role(MessageRole::User);
    /// assert_eq!(envelope.role(), Some(MessageRole::User));
    ///
    /// // Envelopes without a role return None
    /// let plain = Envelope::new(EnvelopeKind::Text("Hello".to_string()));
    /// assert_eq!(plain.role(), None);
    /// ```
    pub fn role(&self) -> Option<super::MessageRole> {
        self.metadata
            .get(Self::ROLE_METADATA_KEY)
            .and_then(|s| match s.as_str() {
                "system" => Some(super::MessageRole::System),
                "user" => Some(super::MessageRole::User),
                "assistant" => Some(super::MessageRole::Assistant),
                _ => None,
            })
    }

    /// Returns `true` if this envelope has the User message role.
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let user_msg = Envelope::new(EnvelopeKind::Text("Hi".to_string()))
    ///     .with_role(MessageRole::User);
    /// assert!(user_msg.is_user_message());
    /// ```
    pub fn is_user_message(&self) -> bool {
        self.role() == Some(super::MessageRole::User)
    }

    /// Returns `true` if this envelope has the Assistant message role.
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let assistant_msg = Envelope::new(EnvelopeKind::Text("Hello!".to_string()))
    ///     .with_role(MessageRole::Assistant);
    /// assert!(assistant_msg.is_assistant_message());
    /// ```
    pub fn is_assistant_message(&self) -> bool {
        self.role() == Some(super::MessageRole::Assistant)
    }

    /// Returns `true` if this envelope has the System message role.
    ///
    /// # Example
    ///
    /// ```rust
    /// use xybrid_core::ir::{Envelope, EnvelopeKind, MessageRole};
    ///
    /// let system_msg = Envelope::new(EnvelopeKind::Text("You are helpful.".to_string()))
    ///     .with_role(MessageRole::System);
    /// assert!(system_msg.is_system_message());
    /// ```
    pub fn is_system_message(&self) -> bool {
        self.role() == Some(super::MessageRole::System)
    }

    /// Returns a string representation of the envelope kind.
    ///
    /// # Returns
    ///
    /// A string describing the variant (e.g., "Audio", "Text", "Embedding")
    pub fn kind_str(&self) -> &'static str {
        self.kind.as_str()
    }

    /// Returns the approximate size of the envelope payload in bytes.
    ///
    /// # Returns
    ///
    /// The size of the payload data
    pub fn payload_size(&self) -> usize {
        self.kind.payload_size()
    }

    /// Returns text content if this is a text envelope.
    pub fn as_text(&self) -> Option<&str> {
        match &self.kind {
            EnvelopeKind::Text(text) => Some(text),
            _ => None,
        }
    }

    /// Returns `true` if this envelope contains an image.
    #[cfg(feature = "vision")]
    pub fn is_image(&self) -> bool {
        matches!(self.kind, EnvelopeKind::Image { .. })
    }

    /// Returns encoded image bytes and format if this is an encoded image.
    #[cfg(feature = "vision")]
    pub fn as_image(&self) -> Option<(&[u8], ImageFormat)> {
        match &self.kind {
            EnvelopeKind::Image { source } => source.as_encoded(),
            _ => None,
        }
    }

    /// Returns `true` if this envelope contains raw pixels.
    #[cfg(feature = "vision")]
    pub fn is_raw_image(&self) -> bool {
        self.as_raw_image().is_some()
    }

    /// Returns raw pixels and layout if this is a raw image.
    #[cfg(feature = "vision")]
    pub fn as_raw_image(&self) -> Option<RawImageRef<'_>> {
        match &self.kind {
            EnvelopeKind::Image { source } => source.as_raw(),
            _ => None,
        }
    }

    /// Returns the image source if this envelope contains an image.
    #[cfg(feature = "vision")]
    pub fn image_source(&self) -> Option<&ImageSource> {
        match &self.kind {
            EnvelopeKind::Image { source } => Some(source),
            _ => None,
        }
    }

    /// Returns decoded image dimensions discovered during validation.
    #[cfg(feature = "vision")]
    pub fn image_dimensions(&self) -> Option<ImageDimensions> {
        self.image_source().and_then(ImageSource::dimensions)
    }

    /// Returns byte-free summaries for every image contained in this envelope.
    ///
    /// For multi-part envelopes, summaries are returned in fragment order and
    /// nested multi-part fragments are traversed depth-first.
    #[cfg(feature = "vision")]
    pub fn image_summaries(&self) -> Vec<ImageSummary> {
        let mut summaries = Vec::new();
        self.collect_image_summaries(&mut summaries);
        summaries
    }

    #[cfg(feature = "vision")]
    fn collect_image_summaries(&self, summaries: &mut Vec<ImageSummary>) {
        match &self.kind {
            EnvelopeKind::Image { source } => summaries.push(source.summary()),
            EnvelopeKind::MultiPart(parts) => {
                for part in parts {
                    part.collect_image_summaries(summaries);
                }
            }
            EnvelopeKind::Audio(_) | EnvelopeKind::Text(_) | EnvelopeKind::Embedding(_) => {}
        }
    }

    /// Revalidates every image contained in this envelope tree.
    ///
    /// This is useful for deserialized or manually constructed envelopes where
    /// callers need the same encoded/raw guardrails as the public constructors.
    #[cfg(feature = "vision")]
    pub fn validate_image_tree(&self) -> EnvelopeResult<()> {
        self.validate_image_tree_with_limits(ImageValidationLimits::default())
    }

    /// Revalidates every image contained in this envelope tree with explicit limits.
    #[cfg(feature = "vision")]
    pub fn validate_image_tree_with_limits(
        &self,
        limits: ImageValidationLimits,
    ) -> EnvelopeResult<()> {
        match &self.kind {
            EnvelopeKind::Image { source } => {
                match source {
                    ImageSource::Encoded { .. } => {
                        source.validated_encoded(limits)?;
                    }
                    ImageSource::Raw { .. } => {
                        source.validated_raw(limits)?;
                    }
                }
                Ok(())
            }
            EnvelopeKind::MultiPart(parts) => {
                for part in parts {
                    part.validate_image_tree_with_limits(limits)?;
                }
                Ok(())
            }
            EnvelopeKind::Audio(_) | EnvelopeKind::Text(_) | EnvelopeKind::Embedding(_) => Ok(()),
        }
    }

    /// Returns ordered fragments if this is a multi-part envelope.
    #[cfg(feature = "vision")]
    pub fn as_multipart(&self) -> Option<&[Envelope]> {
        match &self.kind {
            EnvelopeKind::MultiPart(parts) => Some(parts),
            _ => None,
        }
    }

    /// Serializes the envelope to a byte vector using bincode.
    ///
    /// # Returns
    ///
    /// A `Result` containing the serialized bytes or an error
    pub fn to_bytes(&self) -> Result<Vec<u8>, EnvelopeError> {
        bincode::serialize(self).map_err(|e| {
            EnvelopeError::SerializationError(format!("Failed to serialize envelope: {}", e))
        })
    }

    /// Deserializes an envelope from a byte vector using bincode.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The serialized envelope bytes
    ///
    /// # Returns
    ///
    /// A `Result` containing the deserialized envelope or an error
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, EnvelopeError> {
        bincode::deserialize(bytes).map_err(|e| {
            EnvelopeError::DeserializationError(format!("Failed to deserialize envelope: {}", e))
        })
    }

    /// Serializes the envelope to JSON format (for debugging/telemetry).
    ///
    /// # Returns
    ///
    /// A `Result` containing the JSON string or an error
    pub fn to_json(&self) -> Result<String, EnvelopeError> {
        serde_json::to_string_pretty(self).map_err(|e| {
            EnvelopeError::SerializationError(format!(
                "Failed to serialize envelope to JSON: {}",
                e
            ))
        })
    }

    /// Deserializes an envelope from JSON format.
    ///
    /// # Arguments
    ///
    /// * `json` - The JSON string
    ///
    /// # Returns
    ///
    /// A `Result` containing the deserialized envelope or an error
    pub fn from_json(json: &str) -> Result<Self, EnvelopeError> {
        serde_json::from_str(json).map_err(|e| {
            EnvelopeError::DeserializationError(format!(
                "Failed to deserialize envelope from JSON: {}",
                e
            ))
        })
    }

    /// Extracts audio samples from the envelope based on the `format` metadata.
    ///
    /// Supports the following formats (via `format` metadata):
    /// - `"float32"`: Pre-decoded float32 samples (from AudioEnvelope)
    /// - `"pcm16"`: Raw 16-bit PCM bytes
    /// - `"wav"` or unset: WAV file bytes (caller should use WAV decoder)
    ///
    /// # Returns
    ///
    /// - `Ok(Some(samples))` if audio was successfully extracted
    /// - `Ok(None)` if format is WAV or unknown (caller should decode)
    /// - `Err` if envelope is not Audio type
    ///
    /// # Audio Metadata
    ///
    /// The following metadata keys are used:
    /// - `format`: Audio format ("float32", "pcm16", "wav")
    /// - `sample_rate`: Sample rate in Hz
    /// - `channels`: Number of channels
    pub fn to_audio_samples(&self) -> Result<Option<AudioSamples>, EnvelopeError> {
        let audio_bytes = match &self.kind {
            EnvelopeKind::Audio(bytes) => bytes,
            _ => {
                return Err(EnvelopeError::DeserializationError(
                    "Envelope is not Audio type".to_string(),
                ))
            }
        };

        let format = self
            .get_metadata("format")
            .map(|s| s.as_str())
            .unwrap_or("wav");

        let sample_rate: u32 = self
            .get_metadata("sample_rate")
            .and_then(|s| s.parse().ok())
            .unwrap_or(16000);

        let channels: u32 = self
            .get_metadata("channels")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        match format {
            "float32" => {
                // Pre-decoded float32 samples from AudioEnvelope
                let num_samples = audio_bytes.len() / 4;
                let mut samples = Vec::with_capacity(num_samples);
                for i in 0..num_samples {
                    let idx = i * 4;
                    if idx + 3 < audio_bytes.len() {
                        let sample = f32::from_le_bytes([
                            audio_bytes[idx],
                            audio_bytes[idx + 1],
                            audio_bytes[idx + 2],
                            audio_bytes[idx + 3],
                        ]);
                        samples.push(sample);
                    }
                }
                Ok(Some(AudioSamples {
                    samples,
                    sample_rate,
                    channels,
                }))
            }
            "pcm16" => {
                // Raw 16-bit PCM bytes
                let num_samples = audio_bytes.len() / 2;
                let mut samples = Vec::with_capacity(num_samples);
                for i in 0..num_samples {
                    let idx = i * 2;
                    if idx + 1 < audio_bytes.len() {
                        let sample_i16 =
                            i16::from_le_bytes([audio_bytes[idx], audio_bytes[idx + 1]]);
                        samples.push(sample_i16 as f32 / 32768.0);
                    }
                }
                Ok(Some(AudioSamples {
                    samples,
                    sample_rate,
                    channels,
                }))
            }
            _ => {
                // WAV or unknown format - caller should use WAV decoder
                Ok(None)
            }
        }
    }

    /// Returns the raw audio bytes if this is an Audio envelope.
    pub fn audio_bytes(&self) -> Option<&[u8]> {
        match &self.kind {
            EnvelopeKind::Audio(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// Returns the audio format from metadata.
    pub fn audio_format(&self) -> Option<&str> {
        self.get_metadata("format").map(|s| s.as_str())
    }
}

/// Extracted audio samples with metadata.
#[derive(Debug, Clone)]
pub struct AudioSamples {
    /// Normalized float32 samples (-1.0 to 1.0)
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
}

impl AudioSamples {
    /// Convert to mono by averaging channels.
    pub fn to_mono(&self) -> Self {
        if self.channels <= 1 {
            return self.clone();
        }

        let channels = self.channels as usize;
        let mono_samples: Vec<f32> = self
            .samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect();

        Self {
            samples: mono_samples,
            sample_rate: self.sample_rate,
            channels: 1,
        }
    }

    /// Resample to target sample rate using linear interpolation.
    pub fn resample(&self, target_rate: u32) -> Self {
        if self.sample_rate == target_rate {
            return self.clone();
        }

        let ratio = target_rate as f32 / self.sample_rate as f32;
        let target_len = (self.samples.len() as f32 * ratio) as usize;

        let resampled: Vec<f32> = (0..target_len)
            .map(|i| {
                let source_idx = (i as f32 / ratio) as usize;
                self.samples.get(source_idx).copied().unwrap_or(0.0)
            })
            .collect();

        Self {
            samples: resampled,
            sample_rate: target_rate,
            channels: self.channels,
        }
    }

    /// Prepare for ASR (convert to mono 16kHz).
    pub fn prepare_for_asr(&self) -> Self {
        self.to_mono().resample(16000)
    }
}

/// Error type for envelope operations.
#[derive(Error, Debug)]
pub enum EnvelopeError {
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    #[cfg(feature = "vision")]
    #[error("Unsupported image format '{format}'; expected png, jpeg, or webp")]
    UnsupportedImageFormat { format: String },
    #[cfg(feature = "vision")]
    #[error(
        "Unsupported raw pixel format '{format}'; expected rgb8, rgba8, bgra8, nv12, nv21, or i420"
    )]
    UnsupportedPixelFormat { format: String },
    #[cfg(feature = "vision")]
    #[error("Image payload too large: {byte_len} bytes exceeds max {max_bytes} bytes")]
    ImageEncodedTooLarge { byte_len: usize, max_bytes: usize },
    #[cfg(feature = "vision")]
    #[error(
        "Image dimensions too large: {width}x{height} ({pixels} pixels) exceeds max {max_pixels} pixels"
    )]
    ImageDimensionsTooLarge {
        width: u32,
        height: u32,
        pixels: u64,
        max_pixels: u64,
    },
    #[cfg(feature = "vision")]
    #[error("Image decode failed: invalid or corrupt {format} image bytes")]
    ImageDecodeFailed { format: ImageFormat },
    #[cfg(feature = "vision")]
    #[error("Animated {format} images are not supported")]
    AnimatedImageUnsupported { format: ImageFormat },
    #[cfg(feature = "vision")]
    #[error(
        "Raw image plane count mismatch for {pixel_format}: expected {expected}, got {actual}"
    )]
    RawImagePlaneCountMismatch {
        pixel_format: PixelFormat,
        expected: usize,
        actual: usize,
    },
    #[cfg(feature = "vision")]
    #[error("Raw image plane {plane_index} is invalid: {reason}")]
    RawImagePlaneInvalid { plane_index: usize, reason: String },
    #[cfg(feature = "vision")]
    #[error("Raw YUV image format {pixel_format} requires YUV color metadata")]
    RawImageColorMetadataRequired { pixel_format: PixelFormat },
    #[cfg(feature = "vision")]
    #[error("Raw RGB image format {pixel_format} must not carry YUV color metadata")]
    RawImageColorMetadataUnsupported { pixel_format: PixelFormat },
    #[error("Validation error: {0}")]
    ValidationError(String),
}

/// Result type for envelope operations.
pub type EnvelopeResult<T> = Result<T, EnvelopeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_kind_as_str() {
        assert_eq!(EnvelopeKind::Audio(vec![]).as_str(), "Audio");
        assert_eq!(EnvelopeKind::Text(String::new()).as_str(), "Text");
        assert_eq!(EnvelopeKind::Embedding(vec![]).as_str(), "Embedding");
    }

    #[test]
    fn test_envelope_kind_payload_size() {
        let audio = EnvelopeKind::Audio(vec![0u8; 100]);
        assert_eq!(audio.payload_size(), 100);

        let text = EnvelopeKind::Text("hello".to_string());
        assert_eq!(text.payload_size(), 5);

        let embedding = EnvelopeKind::Embedding(vec![0.0f32; 10]);
        assert_eq!(embedding.payload_size(), 10 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_envelope_new() {
        let envelope = Envelope::new(EnvelopeKind::Text("test".to_string()));
        assert_eq!(envelope.kind, EnvelopeKind::Text("test".to_string()));
        // New envelopes have a local_id automatically generated
        assert!(!envelope.local_id().is_empty());
        assert_eq!(envelope.local_id().len(), 36); // UUID format
    }

    #[test]
    fn test_envelope_unique_local_ids() {
        let e1 = Envelope::new(EnvelopeKind::Text("same text".to_string()));
        let e2 = Envelope::new(EnvelopeKind::Text("same text".to_string()));

        // Each envelope has a unique local ID even with identical content
        assert_ne!(e1.local_id(), e2.local_id());
    }

    #[test]
    fn test_envelope_with_local_id() {
        let envelope =
            Envelope::new(EnvelopeKind::Text("test".to_string())).with_local_id("custom-id-123");

        assert_eq!(envelope.local_id(), "custom-id-123");
    }

    #[test]
    fn test_envelope_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), "value1".to_string());
        let envelope =
            Envelope::with_metadata(EnvelopeKind::Audio(vec![1, 2, 3]), metadata.clone());

        // with_metadata preserves provided metadata AND adds local_id
        assert_eq!(envelope.get_metadata("key1"), Some(&"value1".to_string()));
        assert!(!envelope.local_id().is_empty());
    }

    #[test]
    fn test_envelope_with_metadata_preserves_local_id() {
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), "value1".to_string());
        metadata.insert(
            Envelope::LOCAL_ID_METADATA_KEY.to_string(),
            "my-custom-id".to_string(),
        );
        let envelope = Envelope::with_metadata(EnvelopeKind::Audio(vec![1, 2, 3]), metadata);

        // Custom local_id in metadata is preserved
        assert_eq!(envelope.local_id(), "my-custom-id");
    }

    #[test]
    fn test_envelope_metadata_operations() {
        let mut envelope = Envelope::new(EnvelopeKind::Text("test".to_string()));

        envelope.set_metadata("key1".to_string(), "value1".to_string());
        assert_eq!(envelope.get_metadata("key1"), Some(&"value1".to_string()));
        assert_eq!(envelope.get_metadata("nonexistent"), None);
    }

    #[test]
    fn test_envelope_kind_str() {
        let envelope = Envelope::new(EnvelopeKind::Audio(vec![]));
        assert_eq!(envelope.kind_str(), "Audio");
    }

    #[test]
    fn test_envelope_serialization() -> Result<(), EnvelopeError> {
        let mut envelope = Envelope::new(EnvelopeKind::Text("hello world".to_string()));
        envelope.set_metadata("stage".to_string(), "asr".to_string());

        // Serialize to bytes
        let bytes = envelope.to_bytes()?;
        assert!(!bytes.is_empty());

        // Deserialize from bytes
        let deserialized = Envelope::from_bytes(&bytes)?;
        assert_eq!(deserialized.kind, envelope.kind);
        assert_eq!(deserialized.metadata, envelope.metadata);

        Ok(())
    }

    #[test]
    fn test_envelope_json_serialization() -> Result<(), EnvelopeError> {
        let mut envelope = Envelope::new(EnvelopeKind::Text("hello".to_string()));
        envelope.set_metadata("key".to_string(), "value".to_string());

        // Serialize to JSON
        let json = envelope.to_json()?;
        assert!(json.contains("hello"));
        assert!(json.contains("key"));

        // Deserialize from JSON
        let deserialized = Envelope::from_json(&json)?;
        assert_eq!(deserialized.kind, envelope.kind);
        assert_eq!(deserialized.metadata, envelope.metadata);

        Ok(())
    }

    #[test]
    fn test_envelope_audio_roundtrip() -> Result<(), EnvelopeError> {
        let audio_data = vec![0u8, 1u8, 2u8, 3u8, 4u8];
        let envelope = Envelope::new(EnvelopeKind::Audio(audio_data.clone()));

        let bytes = envelope.to_bytes()?;
        let deserialized = Envelope::from_bytes(&bytes)?;

        match deserialized.kind {
            EnvelopeKind::Audio(data) => assert_eq!(data, audio_data),
            _ => panic!("Expected Audio variant"),
        }

        Ok(())
    }

    #[test]
    fn test_envelope_embedding_roundtrip() -> Result<(), EnvelopeError> {
        let embedding_data = vec![1.0f32, 2.0f32, 3.0f32];
        let envelope = Envelope::new(EnvelopeKind::Embedding(embedding_data.clone()));

        let bytes = envelope.to_bytes()?;
        let deserialized = Envelope::from_bytes(&bytes)?;

        match deserialized.kind {
            EnvelopeKind::Embedding(data) => assert_eq!(data, embedding_data),
            _ => panic!("Expected Embedding variant"),
        }

        Ok(())
    }

    // =========================================================================
    // Message Role Tests
    // =========================================================================

    #[test]
    fn test_envelope_with_role_user() {
        use super::super::MessageRole;

        let envelope =
            Envelope::new(EnvelopeKind::Text("Hello".to_string())).with_role(MessageRole::User);

        assert_eq!(envelope.role(), Some(MessageRole::User));
        assert!(envelope.is_user_message());
        assert!(!envelope.is_assistant_message());
        assert!(!envelope.is_system_message());
    }

    #[test]
    fn test_envelope_with_role_assistant() {
        use super::super::MessageRole;

        let envelope = Envelope::new(EnvelopeKind::Text("Hi there!".to_string()))
            .with_role(MessageRole::Assistant);

        assert_eq!(envelope.role(), Some(MessageRole::Assistant));
        assert!(!envelope.is_user_message());
        assert!(envelope.is_assistant_message());
        assert!(!envelope.is_system_message());
    }

    #[test]
    fn test_envelope_with_role_system() {
        use super::super::MessageRole;

        let envelope = Envelope::new(EnvelopeKind::Text("You are helpful.".to_string()))
            .with_role(MessageRole::System);

        assert_eq!(envelope.role(), Some(MessageRole::System));
        assert!(!envelope.is_user_message());
        assert!(!envelope.is_assistant_message());
        assert!(envelope.is_system_message());
    }

    #[test]
    fn test_envelope_without_role() {
        let envelope = Envelope::new(EnvelopeKind::Text("Plain message".to_string()));

        // Envelopes without a role return None (backwards compatible)
        assert_eq!(envelope.role(), None);
        assert!(!envelope.is_user_message());
        assert!(!envelope.is_assistant_message());
        assert!(!envelope.is_system_message());
    }

    #[test]
    fn test_envelope_role_roundtrip() {
        use super::super::MessageRole;

        // Test round-trip: with_role -> role() returns correct value
        for role in [
            MessageRole::System,
            MessageRole::User,
            MessageRole::Assistant,
        ] {
            let envelope = Envelope::new(EnvelopeKind::Text("test".to_string())).with_role(role);
            assert_eq!(
                envelope.role(),
                Some(role),
                "Round-trip failed for {:?}",
                role
            );
        }
    }

    #[test]
    fn test_envelope_role_metadata_key() {
        use super::super::MessageRole;

        let envelope =
            Envelope::new(EnvelopeKind::Text("test".to_string())).with_role(MessageRole::User);

        // Verify the metadata key is correctly set
        assert_eq!(
            envelope.get_metadata(Envelope::ROLE_METADATA_KEY),
            Some(&"user".to_string())
        );
    }

    #[test]
    fn test_envelope_role_serialization_roundtrip() -> Result<(), EnvelopeError> {
        use super::super::MessageRole;

        let envelope =
            Envelope::new(EnvelopeKind::Text("Hello".to_string())).with_role(MessageRole::User);

        // Binary roundtrip
        let bytes = envelope.to_bytes()?;
        let deserialized = Envelope::from_bytes(&bytes)?;
        assert_eq!(deserialized.role(), Some(MessageRole::User));

        // JSON roundtrip
        let json = envelope.to_json()?;
        let from_json = Envelope::from_json(&json)?;
        assert_eq!(from_json.role(), Some(MessageRole::User));

        Ok(())
    }
}
