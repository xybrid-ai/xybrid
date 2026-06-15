//! Image preprocessing operations.
//!
//! This module provides:
//! - `image_decode_step`: Decode encoded PNG/JPEG/WebP image envelopes
//! - `image_resize_step`: Resize image tensors with stretch/letterbox/center modes
//! - `image_normalize_step`: Apply ImageNet/CLIP/SigLIP/custom normalization
//! - `center_crop_step`: Center crop image tensor to target dimensions
//! - `resize_step`: Resize image tensor using interpolation

use super::super::types::{ExecutorResult, PreprocessedData};
use crate::execution::template::InterpolationMethod;
use crate::execution::template::{ImageNormalizePreset, ImageResizeMode, ImageTensorLayout};
use crate::ir::{
    ImageFormat, ImagePlane, ImageSource, ImageValidationLimits, PixelFormat, RawImageRef,
    YuvColorInfo, YuvColorMatrix, YuvColorRange,
};
use crate::runtime_adapter::AdapterError;
use ndarray::{ArrayD, IxDyn};
use yuvutils_rs::{
    yuv420_to_rgb, yuv_nv12_to_rgb, yuv_nv21_to_rgb, YuvBiPlanarImage, YuvConversionMode,
    YuvPlanarImage, YuvRange, YuvStandardMatrix,
};

/// Decode encoded image bytes into a float tensor in [0, 1].
pub fn image_decode_step(
    data: PreprocessedData,
    channels: usize,
    layout: ImageTensorLayout,
) -> ExecutorResult<PreprocessedData> {
    let source = match data {
        PreprocessedData::Image { source } => source,
        _ => {
            return Err(AdapterError::InvalidInput(
                "ImageDecode requires image input".to_string(),
            ))
        }
    };
    let (bytes, format, _dimensions) =
        source
            .validated_encoded(ImageValidationLimits::default())
            .map_err(|err| AdapterError::InvalidInput(err.to_string()))?;

    let decoded =
        image::ImageReader::with_format(std::io::Cursor::new(bytes), image_crate_format(format))
            .decode()
            .map_err(|_| {
                AdapterError::InvalidInput(format!("ImageDecode failed for {} input", format))
            })?;

    match channels {
        1 => Ok(decode_grayscale(decoded, layout)),
        3 => Ok(decode_rgb(decoded, layout)),
        other => Err(AdapterError::InvalidInput(format!(
            "ImageDecode supports 1 or 3 output channels (got {})",
            other
        ))),
    }
}

/// Ingress encoded or raw image sources into a float tensor in [0, 1].
pub fn image_ingress_step(
    data: PreprocessedData,
    channels: usize,
    layout: ImageTensorLayout,
) -> ExecutorResult<PreprocessedData> {
    let source = match data {
        PreprocessedData::Image { source } => source,
        _ => {
            return Err(AdapterError::InvalidInput(
                "ImageIngress requires image input".to_string(),
            ))
        }
    };

    match source {
        ImageSource::Encoded { .. } => {
            image_decode_step(PreprocessedData::Image { source }, channels, layout)
        }
        ImageSource::Raw { .. } => {
            let raw = source
                .validated_raw(ImageValidationLimits::default())
                .map_err(|err| AdapterError::InvalidInput(err.to_string()))?;
            let conversion_buffer_allocations = match raw.pixel_format {
                PixelFormat::Nv12 | PixelFormat::Nv21 | PixelFormat::I420 => "1",
                PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Bgra8 => "0",
            };
            crate::tracing::add_metadata("raw_image_source_vec_allocations", "1");
            crate::tracing::add_metadata("raw_image_source_extra_pixel_copies", "0");
            crate::tracing::add_metadata(
                "raw_image_conversion_buffer_allocations",
                conversion_buffer_allocations,
            );
            crate::tracing::add_metadata("raw_image_pixel_bytes", raw.pixels.len().to_string());
            raw_image_to_tensor(raw, channels, layout)
        }
    }
}

fn raw_image_to_tensor(
    raw: RawImageRef<'_>,
    channels: usize,
    layout: ImageTensorLayout,
) -> ExecutorResult<PreprocessedData> {
    match raw.pixel_format {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Bgra8 => {
            raw_packed_to_tensor(raw, channels, layout)
        }
        PixelFormat::Nv12 | PixelFormat::Nv21 | PixelFormat::I420 => {
            let rgb = raw_yuv_to_rgb(raw)?;
            rgb_bytes_to_tensor(
                &rgb,
                raw.dimensions.width as usize,
                raw.dimensions.height as usize,
                channels,
                layout,
            )
        }
    }
}

fn raw_packed_to_tensor(
    raw: RawImageRef<'_>,
    channels: usize,
    layout: ImageTensorLayout,
) -> ExecutorResult<PreprocessedData> {
    if channels != 1 && channels != 3 {
        return Err(AdapterError::InvalidInput(format!(
            "ImageIngress supports 1 or 3 output channels (got {})",
            channels
        )));
    }

    let plane = raw.planes.first().ok_or_else(|| {
        AdapterError::InvalidInput(format!("{} raw image is missing plane 0", raw.pixel_format))
    })?;
    let width = raw.dimensions.width as usize;
    let height = raw.dimensions.height as usize;
    let mut tensor = match (channels, layout) {
        (1, ImageTensorLayout::Nchw) => ArrayD::<f32>::zeros(IxDyn(&[1, 1, height, width])),
        (1, ImageTensorLayout::Nhwc) => ArrayD::<f32>::zeros(IxDyn(&[1, height, width, 1])),
        (3, ImageTensorLayout::Nchw) => ArrayD::<f32>::zeros(IxDyn(&[1, 3, height, width])),
        (3, ImageTensorLayout::Nhwc) => ArrayD::<f32>::zeros(IxDyn(&[1, height, width, 3])),
        _ => unreachable!(),
    };

    for y in 0..height {
        for x in 0..width {
            let source_offset = sample_offset(plane, x, y)?;
            let pixel = raw
                .pixels
                .get(source_offset..source_offset + plane.pixel_stride)
                .ok_or_else(|| {
                    AdapterError::InvalidInput(format!(
                        "{} raw image plane 0 is out of bounds",
                        raw.pixel_format
                    ))
                })?;
            let [red, green, blue] = packed_pixel_to_rgb(raw.pixel_format, pixel);
            let red = red as f32 / 255.0;
            let green = green as f32 / 255.0;
            let blue = blue as f32 / 255.0;

            if channels == 1 {
                let luma = (0.299 * red) + (0.587 * green) + (0.114 * blue);
                match layout {
                    ImageTensorLayout::Nchw => tensor[IxDyn(&[0, 0, y, x])] = luma,
                    ImageTensorLayout::Nhwc => tensor[IxDyn(&[0, y, x, 0])] = luma,
                }
            } else {
                for (channel, value) in [red, green, blue].into_iter().enumerate() {
                    match layout {
                        ImageTensorLayout::Nchw => tensor[IxDyn(&[0, channel, y, x])] = value,
                        ImageTensorLayout::Nhwc => tensor[IxDyn(&[0, y, x, channel])] = value,
                    }
                }
            }
        }
    }

    Ok(PreprocessedData::Tensor(tensor))
}

fn raw_yuv_to_rgb(raw: RawImageRef<'_>) -> ExecutorResult<Vec<u8>> {
    let color = raw.color.ok_or_else(|| {
        AdapterError::InvalidInput(format!(
            "{} raw image requires YUV color metadata",
            raw.pixel_format
        ))
    })?;
    let width = raw.dimensions.width;
    let height = raw.dimensions.height;
    let mut rgb = vec![0_u8; width as usize * height as usize * 3];
    let rgb_stride = width * 3;
    let range = yuv_range(color);
    let matrix = yuv_matrix(color);

    match raw.pixel_format {
        PixelFormat::Nv12 | PixelFormat::Nv21 => {
            let y_plane = raw.planes.first().ok_or_else(|| {
                AdapterError::InvalidInput(format!(
                    "{} raw image is missing Y plane",
                    raw.pixel_format
                ))
            })?;
            let uv_plane = raw.planes.get(1).ok_or_else(|| {
                AdapterError::InvalidInput(format!(
                    "{} raw image is missing UV plane",
                    raw.pixel_format
                ))
            })?;
            let image = YuvBiPlanarImage {
                y_plane: plane_tail(raw.pixels, y_plane, "Y")?,
                y_stride: y_plane.row_stride as u32,
                uv_plane: plane_tail(raw.pixels, uv_plane, "UV")?,
                uv_stride: uv_plane.row_stride as u32,
                width,
                height,
            };
            let result = match raw.pixel_format {
                PixelFormat::Nv12 => yuv_nv12_to_rgb(
                    &image,
                    &mut rgb,
                    rgb_stride,
                    range,
                    matrix,
                    YuvConversionMode::Balanced,
                ),
                PixelFormat::Nv21 => yuv_nv21_to_rgb(
                    &image,
                    &mut rgb,
                    rgb_stride,
                    range,
                    matrix,
                    YuvConversionMode::Balanced,
                ),
                PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Bgra8 | PixelFormat::I420 => {
                    unreachable!()
                }
            };
            result.map_err(|err| {
                AdapterError::InvalidInput(format!(
                    "{} raw image YUV conversion failed: {err}",
                    raw.pixel_format
                ))
            })?;
        }
        PixelFormat::I420 => {
            let y_plane = raw.planes.first().ok_or_else(|| {
                AdapterError::InvalidInput("i420 raw image is missing Y plane".to_string())
            })?;
            let u_plane = raw.planes.get(1).ok_or_else(|| {
                AdapterError::InvalidInput("i420 raw image is missing U plane".to_string())
            })?;
            let v_plane = raw.planes.get(2).ok_or_else(|| {
                AdapterError::InvalidInput("i420 raw image is missing V plane".to_string())
            })?;
            let image = YuvPlanarImage {
                y_plane: plane_tail(raw.pixels, y_plane, "Y")?,
                y_stride: y_plane.row_stride as u32,
                u_plane: plane_tail(raw.pixels, u_plane, "U")?,
                u_stride: u_plane.row_stride as u32,
                v_plane: plane_tail(raw.pixels, v_plane, "V")?,
                v_stride: v_plane.row_stride as u32,
                width,
                height,
            };
            yuv420_to_rgb(&image, &mut rgb, rgb_stride, range, matrix).map_err(|err| {
                AdapterError::InvalidInput(format!("i420 raw image YUV conversion failed: {err}"))
            })?;
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Bgra8 => unreachable!(),
    }

    Ok(rgb)
}

/// Channel-swap a single packed pixel into RGB order.
///
/// Shared by the tensor path ([`raw_packed_to_tensor`]) and the packed-RGB
/// byte path ([`raw_packed_to_rgb`]) so the per-format channel ordering and
/// alpha stripping live in exactly one place.
#[inline]
fn packed_pixel_to_rgb(format: PixelFormat, pixel: &[u8]) -> [u8; 3] {
    match format {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 => [pixel[0], pixel[1], pixel[2]],
        PixelFormat::Bgra8 => [pixel[2], pixel[1], pixel[0]],
        PixelFormat::Nv12 | PixelFormat::Nv21 | PixelFormat::I420 => unreachable!(),
    }
}

/// Convert any [`ImageSource::Raw`] view into a tightly-packed RGB byte buffer
/// of exactly `width * height * 3` bytes (RGBRGB... order, no stride padding,
/// no alpha).
///
/// This is the ingress point for the raw-frame mtmd path: the returned buffer
/// is exactly the layout `mtmd_bitmap_init` expects, so camera frames feed the
/// VLM without a per-frame JPEG encode/decode round-trip. clip (inside mtmd)
/// performs its own resize and normalization, so the full-resolution pixels
/// are passed through unscaled.
///
/// YUV formats reuse the existing BT.601/BT.709 conversion in
/// [`raw_yuv_to_rgb`]; RGB-family formats strip row stride and alpha while
/// honoring each plane's `pixel_stride`.
pub fn raw_image_to_packed_rgb(raw: RawImageRef<'_>) -> ExecutorResult<Vec<u8>> {
    match raw.pixel_format {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Bgra8 => raw_packed_to_rgb(raw),
        PixelFormat::Nv12 | PixelFormat::Nv21 | PixelFormat::I420 => raw_yuv_to_rgb(raw),
    }
}

/// Strip stride/alpha from a packed RGB-family raw image into tightly-packed RGB.
fn raw_packed_to_rgb(raw: RawImageRef<'_>) -> ExecutorResult<Vec<u8>> {
    let plane = raw.planes.first().ok_or_else(|| {
        AdapterError::InvalidInput(format!("{} raw image is missing plane 0", raw.pixel_format))
    })?;
    let width = raw.dimensions.width as usize;
    let height = raw.dimensions.height as usize;
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| AdapterError::InvalidInput("raw image size overflow".to_string()))?;
    let mut rgb = Vec::with_capacity(pixel_count * 3);

    for y in 0..height {
        for x in 0..width {
            let source_offset = sample_offset(plane, x, y)?;
            let pixel = raw
                .pixels
                .get(source_offset..source_offset + plane.pixel_stride)
                .ok_or_else(|| {
                    AdapterError::InvalidInput(format!(
                        "{} raw image plane 0 is out of bounds",
                        raw.pixel_format
                    ))
                })?;
            rgb.extend_from_slice(&packed_pixel_to_rgb(raw.pixel_format, pixel));
        }
    }

    Ok(rgb)
}

fn rgb_bytes_to_tensor(
    rgb: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    layout: ImageTensorLayout,
) -> ExecutorResult<PreprocessedData> {
    if channels != 1 && channels != 3 {
        return Err(AdapterError::InvalidInput(format!(
            "ImageIngress supports 1 or 3 output channels (got {})",
            channels
        )));
    }
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| AdapterError::InvalidInput("image tensor size overflow".to_string()))?;
    let expected_rgb_len = pixel_count
        .checked_mul(3)
        .ok_or_else(|| AdapterError::InvalidInput("RGB byte length overflow".to_string()))?;
    if rgb.len() != expected_rgb_len {
        return Err(AdapterError::InvalidInput(format!(
            "RGB byte length mismatch: expected {}, got {}",
            expected_rgb_len,
            rgb.len()
        )));
    }

    let mut tensor = match (channels, layout) {
        (1, ImageTensorLayout::Nchw) => ArrayD::<f32>::zeros(IxDyn(&[1, 1, height, width])),
        (1, ImageTensorLayout::Nhwc) => ArrayD::<f32>::zeros(IxDyn(&[1, height, width, 1])),
        (3, ImageTensorLayout::Nchw) => ArrayD::<f32>::zeros(IxDyn(&[1, 3, height, width])),
        (3, ImageTensorLayout::Nhwc) => ArrayD::<f32>::zeros(IxDyn(&[1, height, width, 3])),
        _ => unreachable!(),
    };
    let output = tensor.as_slice_memory_order_mut().ok_or_else(|| {
        AdapterError::RuntimeError("image tensor storage is not contiguous".to_string())
    })?;
    let scale = 1.0 / 255.0;

    match (channels, layout) {
        (1, _) => {
            for (dst, pixel) in output.iter_mut().zip(rgb.chunks_exact(3)) {
                let red = pixel[0] as f32 * scale;
                let green = pixel[1] as f32 * scale;
                let blue = pixel[2] as f32 * scale;
                *dst = (0.299 * red) + (0.587 * green) + (0.114 * blue);
            }
        }
        (3, ImageTensorLayout::Nchw) => {
            let (red_plane, remaining) = output.split_at_mut(pixel_count);
            let (green_plane, blue_plane) = remaining.split_at_mut(pixel_count);
            for (index, pixel) in rgb.chunks_exact(3).enumerate() {
                red_plane[index] = pixel[0] as f32 * scale;
                green_plane[index] = pixel[1] as f32 * scale;
                blue_plane[index] = pixel[2] as f32 * scale;
            }
        }
        (3, ImageTensorLayout::Nhwc) => {
            for (dst, pixel) in output.chunks_exact_mut(3).zip(rgb.chunks_exact(3)) {
                for channel in 0..3 {
                    dst[channel] = pixel[channel] as f32 * scale;
                }
            }
        }
        _ => unreachable!(),
    }

    Ok(PreprocessedData::Tensor(tensor))
}

fn sample_offset(plane: &ImagePlane, x: usize, y: usize) -> ExecutorResult<usize> {
    let row_offset = y
        .checked_mul(plane.row_stride)
        .ok_or_else(|| AdapterError::InvalidInput("raw image row offset overflow".to_string()))?;
    let column_offset = x.checked_mul(plane.pixel_stride).ok_or_else(|| {
        AdapterError::InvalidInput("raw image column offset overflow".to_string())
    })?;
    plane
        .offset
        .checked_add(row_offset)
        .and_then(|offset| offset.checked_add(column_offset))
        .ok_or_else(|| AdapterError::InvalidInput("raw image plane offset overflow".to_string()))
}

fn plane_tail<'a>(
    pixels: &'a [u8],
    plane: &ImagePlane,
    label: &'static str,
) -> ExecutorResult<&'a [u8]> {
    pixels.get(plane.offset..).ok_or_else(|| {
        AdapterError::InvalidInput(format!("raw image {label} plane offset is out of bounds"))
    })
}

fn yuv_range(color: YuvColorInfo) -> YuvRange {
    match color.range {
        YuvColorRange::Limited => YuvRange::Limited,
        YuvColorRange::Full => YuvRange::Full,
    }
}

fn yuv_matrix(color: YuvColorInfo) -> YuvStandardMatrix {
    match color.matrix {
        YuvColorMatrix::Bt601 => YuvStandardMatrix::Bt601,
        YuvColorMatrix::Bt709 => YuvStandardMatrix::Bt709,
        YuvColorMatrix::Bt2020 => YuvStandardMatrix::Bt2020,
    }
}

fn decode_rgb(decoded: image::DynamicImage, layout: ImageTensorLayout) -> PreprocessedData {
    let image = decoded.to_rgb8();
    let (width, height) = image.dimensions();
    let width = width as usize;
    let height = height as usize;

    let mut tensor = match layout {
        ImageTensorLayout::Nchw => ArrayD::<f32>::zeros(IxDyn(&[1, 3, height, width])),
        ImageTensorLayout::Nhwc => ArrayD::<f32>::zeros(IxDyn(&[1, height, width, 3])),
    };

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x as u32, y as u32);
            for channel in 0..3 {
                let value = pixel[channel] as f32 / 255.0;
                match layout {
                    ImageTensorLayout::Nchw => tensor[IxDyn(&[0, channel, y, x])] = value,
                    ImageTensorLayout::Nhwc => tensor[IxDyn(&[0, y, x, channel])] = value,
                }
            }
        }
    }

    PreprocessedData::Tensor(tensor)
}

fn decode_grayscale(decoded: image::DynamicImage, layout: ImageTensorLayout) -> PreprocessedData {
    let image = decoded.to_luma8();
    let (width, height) = image.dimensions();
    let width = width as usize;
    let height = height as usize;

    let mut tensor = match layout {
        ImageTensorLayout::Nchw => ArrayD::<f32>::zeros(IxDyn(&[1, 1, height, width])),
        ImageTensorLayout::Nhwc => ArrayD::<f32>::zeros(IxDyn(&[1, height, width, 1])),
    };

    for y in 0..height {
        for x in 0..width {
            let value = image.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
            match layout {
                ImageTensorLayout::Nchw => tensor[IxDyn(&[0, 0, y, x])] = value,
                ImageTensorLayout::Nhwc => tensor[IxDyn(&[0, y, x, 0])] = value,
            }
        }
    }

    PreprocessedData::Tensor(tensor)
}

fn image_crate_format(format: ImageFormat) -> image::ImageFormat {
    match format {
        ImageFormat::Png => image::ImageFormat::Png,
        ImageFormat::Jpeg => image::ImageFormat::Jpeg,
        ImageFormat::WebP => image::ImageFormat::WebP,
    }
}

/// Resize an image tensor with explicit aspect-ratio handling.
pub fn image_resize_step(
    data: PreprocessedData,
    width: usize,
    height: usize,
    mode: ImageResizeMode,
    interpolation: &InterpolationMethod,
    fill: &[f32],
    layout: ImageTensorLayout,
) -> ExecutorResult<PreprocessedData> {
    let tensor = match data {
        PreprocessedData::Tensor(tensor) => tensor,
        _ => {
            return Err(AdapterError::InvalidInput(
                "ImageResize requires tensor input".to_string(),
            ))
        }
    };

    let spec = ImageTensorSpec::from_shape(tensor.shape(), layout)?;
    let filter_type = filter_type(interpolation);
    let output_shape = spec.output_shape(width, height);
    let mut output = ArrayD::<f32>::zeros(IxDyn(&output_shape));

    for batch in 0..spec.batch {
        match spec.channels {
            1 => {
                let source = tensor_to_gray_image(&tensor, &spec, batch);
                let (resized, offset_x, offset_y) =
                    resize_gray_with_mode(&source, width, height, mode, filter_type);
                fill_output(&mut output, &spec, batch, width, height, fill);
                write_gray_image_to_tensor(&resized, &mut output, &spec, batch, offset_x, offset_y);
            }
            3 => {
                let source = tensor_to_rgb_image(&tensor, &spec, batch);
                let (resized, offset_x, offset_y) =
                    resize_rgb_with_mode(&source, width, height, mode, filter_type);
                fill_output(&mut output, &spec, batch, width, height, fill);
                write_rgb_image_to_tensor(&resized, &mut output, &spec, batch, offset_x, offset_y);
            }
            channels => {
                return Err(AdapterError::InvalidInput(format!(
                    "ImageResize supports 1 or 3 channels (got {})",
                    channels
                )))
            }
        }
    }

    Ok(PreprocessedData::Tensor(output))
}

/// Normalize image tensors with a named or custom preset.
pub fn image_normalize_step(
    data: PreprocessedData,
    preset: &ImageNormalizePreset,
    layout: ImageTensorLayout,
) -> ExecutorResult<PreprocessedData> {
    let mut tensor = match data {
        PreprocessedData::Tensor(tensor) => tensor,
        _ => {
            return Err(AdapterError::InvalidInput(
                "ImageNormalize requires tensor input".to_string(),
            ))
        }
    };

    let spec = ImageTensorSpec::from_shape(tensor.shape(), layout)?;
    let (mean, std) = normalize_params(preset);
    if mean.len() != spec.channels || std.len() != spec.channels {
        return Err(AdapterError::InvalidInput(format!(
            "ImageNormalize preset has {} mean values and {} std values for {} channel tensor",
            mean.len(),
            std.len(),
            spec.channels
        )));
    }
    if std.contains(&0.0) {
        return Err(AdapterError::InvalidInput(
            "ImageNormalize std values must be non-zero".to_string(),
        ));
    }

    for batch in 0..spec.batch {
        for channel in 0..spec.channels {
            for y in 0..spec.height {
                for x in 0..spec.width {
                    let coords = spec.coords(batch, channel, y, x);
                    tensor[IxDyn(&coords)] =
                        (tensor[IxDyn(&coords)] - mean[channel]) / std[channel];
                }
            }
        }
    }

    Ok(PreprocessedData::Tensor(tensor))
}

#[derive(Debug, Clone, Copy)]
struct ImageTensorSpec {
    layout: ImageTensorLayout,
    rank: usize,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
}

impl ImageTensorSpec {
    fn from_shape(shape: &[usize], layout: ImageTensorLayout) -> ExecutorResult<Self> {
        match (shape.len(), layout) {
            (4, ImageTensorLayout::Nchw) => Ok(Self {
                layout,
                rank: 4,
                batch: shape[0],
                channels: shape[1],
                height: shape[2],
                width: shape[3],
            }),
            (4, ImageTensorLayout::Nhwc) => Ok(Self {
                layout,
                rank: 4,
                batch: shape[0],
                channels: shape[3],
                height: shape[1],
                width: shape[2],
            }),
            (3, ImageTensorLayout::Nchw) => Ok(Self {
                layout,
                rank: 3,
                batch: 1,
                channels: shape[0],
                height: shape[1],
                width: shape[2],
            }),
            (3, ImageTensorLayout::Nhwc) => Ok(Self {
                layout,
                rank: 3,
                batch: 1,
                channels: shape[2],
                height: shape[0],
                width: shape[1],
            }),
            _ => Err(AdapterError::InvalidInput(format!(
                "Image tensor requires 3D or 4D input (got {:?})",
                shape
            ))),
        }
    }

    fn output_shape(self, width: usize, height: usize) -> Vec<usize> {
        match (self.rank, self.layout) {
            (4, ImageTensorLayout::Nchw) => vec![self.batch, self.channels, height, width],
            (4, ImageTensorLayout::Nhwc) => vec![self.batch, height, width, self.channels],
            (3, ImageTensorLayout::Nchw) => vec![self.channels, height, width],
            (3, ImageTensorLayout::Nhwc) => vec![height, width, self.channels],
            _ => unreachable!("ImageTensorSpec rank is validated at construction"),
        }
    }

    fn coords(self, batch: usize, channel: usize, y: usize, x: usize) -> Vec<usize> {
        match (self.rank, self.layout) {
            (4, ImageTensorLayout::Nchw) => vec![batch, channel, y, x],
            (4, ImageTensorLayout::Nhwc) => vec![batch, y, x, channel],
            (3, ImageTensorLayout::Nchw) => vec![channel, y, x],
            (3, ImageTensorLayout::Nhwc) => vec![y, x, channel],
            _ => unreachable!("ImageTensorSpec rank is validated at construction"),
        }
    }
}

fn normalize_params(preset: &ImageNormalizePreset) -> (Vec<f32>, Vec<f32>) {
    match preset {
        ImageNormalizePreset::ImageNet => (vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225]),
        ImageNormalizePreset::Clip => (
            vec![0.48145466, 0.4578275, 0.40821073],
            vec![0.26862954, 0.261_302_6, 0.275_777_1],
        ),
        ImageNormalizePreset::SigLip => (vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]),
        ImageNormalizePreset::Custom { mean, std } => (mean.clone(), std.clone()),
    }
}

fn filter_type(interpolation: &InterpolationMethod) -> image::imageops::FilterType {
    match interpolation {
        InterpolationMethod::Nearest => image::imageops::FilterType::Nearest,
        InterpolationMethod::Bilinear => image::imageops::FilterType::Triangle,
        InterpolationMethod::Bicubic => image::imageops::FilterType::CatmullRom,
    }
}

fn resize_dimensions(
    src_width: usize,
    src_height: usize,
    target_width: usize,
    target_height: usize,
    mode: ImageResizeMode,
) -> (usize, usize, usize, usize) {
    match mode {
        ImageResizeMode::Stretch => (target_width, target_height, 0, 0),
        ImageResizeMode::Letterbox => {
            let scale = (target_width as f32 / src_width as f32)
                .min(target_height as f32 / src_height as f32);
            let width = ((src_width as f32 * scale).round() as usize).clamp(1, target_width);
            let height = ((src_height as f32 * scale).round() as usize).clamp(1, target_height);
            (
                (width),
                height,
                (target_width - width) / 2,
                (target_height - height) / 2,
            )
        }
        ImageResizeMode::Center => {
            let scale = (target_width as f32 / src_width as f32)
                .max(target_height as f32 / src_height as f32);
            let width = ((src_width as f32 * scale).ceil() as usize).max(target_width);
            let height = ((src_height as f32 * scale).ceil() as usize).max(target_height);
            (width, height, 0, 0)
        }
    }
}

fn resize_rgb_with_mode(
    source: &image::RgbImage,
    target_width: usize,
    target_height: usize,
    mode: ImageResizeMode,
    filter_type: image::imageops::FilterType,
) -> (image::RgbImage, usize, usize) {
    let (resize_width, resize_height, offset_x, offset_y) = resize_dimensions(
        source.width() as usize,
        source.height() as usize,
        target_width,
        target_height,
        mode,
    );
    let resized = image::imageops::resize(
        source,
        resize_width as u32,
        resize_height as u32,
        filter_type,
    );

    if mode == ImageResizeMode::Center {
        let crop_x = (resize_width - target_width) / 2;
        let crop_y = (resize_height - target_height) / 2;
        (
            image::imageops::crop_imm(
                &resized,
                crop_x as u32,
                crop_y as u32,
                target_width as u32,
                target_height as u32,
            )
            .to_image(),
            0,
            0,
        )
    } else {
        (resized, offset_x, offset_y)
    }
}

fn resize_gray_with_mode(
    source: &image::GrayImage,
    target_width: usize,
    target_height: usize,
    mode: ImageResizeMode,
    filter_type: image::imageops::FilterType,
) -> (image::GrayImage, usize, usize) {
    let (resize_width, resize_height, offset_x, offset_y) = resize_dimensions(
        source.width() as usize,
        source.height() as usize,
        target_width,
        target_height,
        mode,
    );
    let resized = image::imageops::resize(
        source,
        resize_width as u32,
        resize_height as u32,
        filter_type,
    );

    if mode == ImageResizeMode::Center {
        let crop_x = (resize_width - target_width) / 2;
        let crop_y = (resize_height - target_height) / 2;
        (
            image::imageops::crop_imm(
                &resized,
                crop_x as u32,
                crop_y as u32,
                target_width as u32,
                target_height as u32,
            )
            .to_image(),
            0,
            0,
        )
    } else {
        (resized, offset_x, offset_y)
    }
}

fn tensor_to_rgb_image(
    tensor: &ArrayD<f32>,
    spec: &ImageTensorSpec,
    batch: usize,
) -> image::RgbImage {
    let mut image = image::RgbImage::new(spec.width as u32, spec.height as u32);
    for y in 0..spec.height {
        for x in 0..spec.width {
            let pixel = image::Rgb([
                tensor_to_u8(tensor[IxDyn(&spec.coords(batch, 0, y, x))]),
                tensor_to_u8(tensor[IxDyn(&spec.coords(batch, 1, y, x))]),
                tensor_to_u8(tensor[IxDyn(&spec.coords(batch, 2, y, x))]),
            ]);
            image.put_pixel(x as u32, y as u32, pixel);
        }
    }
    image
}

fn tensor_to_gray_image(
    tensor: &ArrayD<f32>,
    spec: &ImageTensorSpec,
    batch: usize,
) -> image::GrayImage {
    let mut image = image::GrayImage::new(spec.width as u32, spec.height as u32);
    for y in 0..spec.height {
        for x in 0..spec.width {
            image.put_pixel(
                x as u32,
                y as u32,
                image::Luma([tensor_to_u8(tensor[IxDyn(&spec.coords(batch, 0, y, x))])]),
            );
        }
    }
    image
}

fn fill_output(
    output: &mut ArrayD<f32>,
    spec: &ImageTensorSpec,
    batch: usize,
    width: usize,
    height: usize,
    fill: &[f32],
) {
    for channel in 0..spec.channels {
        let value = fill
            .get(channel)
            .copied()
            .or_else(|| fill.first().copied())
            .unwrap_or(0.0);
        for y in 0..height {
            for x in 0..width {
                output[IxDyn(&spec.coords(batch, channel, y, x))] = value;
            }
        }
    }
}

fn write_rgb_image_to_tensor(
    image: &image::RgbImage,
    output: &mut ArrayD<f32>,
    spec: &ImageTensorSpec,
    batch: usize,
    offset_x: usize,
    offset_y: usize,
) {
    for y in 0..image.height() as usize {
        for x in 0..image.width() as usize {
            let pixel = image.get_pixel(x as u32, y as u32);
            for channel in 0..3 {
                output[IxDyn(&spec.coords(batch, channel, offset_y + y, offset_x + x))] =
                    pixel[channel] as f32 / 255.0;
            }
        }
    }
}

fn write_gray_image_to_tensor(
    image: &image::GrayImage,
    output: &mut ArrayD<f32>,
    spec: &ImageTensorSpec,
    batch: usize,
    offset_x: usize,
    offset_y: usize,
) {
    for y in 0..image.height() as usize {
        for x in 0..image.width() as usize {
            output[IxDyn(&spec.coords(batch, 0, offset_y + y, offset_x + x))] =
                image.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
        }
    }
}

fn tensor_to_u8(value: f32) -> u8 {
    (value * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Center crop image tensor to target dimensions.
///
/// # Arguments
/// - `data`: Input data (Tensor with shape [batch, channels, h, w] or [channels, h, w])
/// - `width`: Target crop width
/// - `height`: Target crop height
pub fn center_crop_step(
    data: PreprocessedData,
    width: usize,
    height: usize,
) -> ExecutorResult<PreprocessedData> {
    let tensor = match data {
        PreprocessedData::Tensor(t) => t,
        _ => {
            return Err(AdapterError::InvalidInput(
                "CenterCrop requires tensor input".to_string(),
            ))
        }
    };

    let shape = tensor.shape();
    if shape.len() < 3 {
        return Err(AdapterError::InvalidInput(format!(
            "CenterCrop requires at least 3D tensor (got {:?})",
            shape
        )));
    }

    let (batch_size, channels, src_h, src_w) = if shape.len() == 4 {
        (shape[0], shape[1], shape[2], shape[3])
    } else {
        (1, shape[0], shape[1], shape[2])
    };

    if height > src_h || width > src_w {
        return Err(AdapterError::InvalidInput(format!(
            "Cannot crop {}x{} from {}x{} image",
            width, height, src_w, src_h
        )));
    }

    let offset_h = (src_h - height) / 2;
    let offset_w = (src_w - width) / 2;

    let out_shape = if shape.len() == 4 {
        vec![batch_size, channels, height, width]
    } else {
        vec![channels, height, width]
    };

    let mut cropped = ArrayD::<f32>::zeros(IxDyn(&out_shape));

    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let src_coords = if shape.len() == 4 {
                        vec![b, c, offset_h + h, offset_w + w]
                    } else {
                        vec![c, offset_h + h, offset_w + w]
                    };
                    let dst_coords = if shape.len() == 4 {
                        vec![b, c, h, w]
                    } else {
                        vec![c, h, w]
                    };

                    cropped[IxDyn(&dst_coords)] = tensor[IxDyn(&src_coords)];
                }
            }
        }
    }

    Ok(PreprocessedData::Tensor(cropped))
}

/// Resize image tensor using interpolation.
///
/// # Arguments
/// - `data`: Input data (Tensor with shape [batch, channels, h, w] or [channels, h, w])
/// - `width`: Target width
/// - `height`: Target height
/// - `interpolation`: Interpolation method (Nearest, Bilinear, Bicubic)
pub fn resize_step(
    data: PreprocessedData,
    width: usize,
    height: usize,
    interpolation: &InterpolationMethod,
) -> ExecutorResult<PreprocessedData> {
    let tensor = match data {
        PreprocessedData::Tensor(t) => t,
        _ => {
            return Err(AdapterError::InvalidInput(
                "Resize requires tensor input".to_string(),
            ))
        }
    };

    let shape = tensor.shape();
    if shape.len() < 3 {
        return Err(AdapterError::InvalidInput(format!(
            "Resize requires at least 3D tensor (got {:?})",
            shape
        )));
    }

    let (batch_size, channels, src_h, src_w) = if shape.len() == 4 {
        (shape[0], shape[1], shape[2], shape[3])
    } else {
        (1, shape[0], shape[1], shape[2])
    };

    if channels != 3 && channels != 1 {
        return Err(AdapterError::InvalidInput(format!(
            "Resize only supports 1 or 3 channels (got {})",
            channels
        )));
    }

    let filter_type = match interpolation {
        InterpolationMethod::Nearest => image::imageops::FilterType::Nearest,
        InterpolationMethod::Bilinear => image::imageops::FilterType::Triangle,
        InterpolationMethod::Bicubic => image::imageops::FilterType::CatmullRom,
    };

    let out_shape = if shape.len() == 4 {
        vec![batch_size, channels, height, width]
    } else {
        vec![channels, height, width]
    };

    let mut resized_tensor = ArrayD::<f32>::zeros(IxDyn(&out_shape));

    for b in 0..batch_size {
        if channels == 3 {
            resized_tensor = resize_rgb_image(
                &tensor,
                resized_tensor,
                shape,
                b,
                src_h,
                src_w,
                width,
                height,
                filter_type,
            )?;
        } else {
            resized_tensor = resize_grayscale_image(
                &tensor,
                resized_tensor,
                shape,
                b,
                src_h,
                src_w,
                width,
                height,
                filter_type,
            )?;
        }
    }

    Ok(PreprocessedData::Tensor(resized_tensor))
}

/// Helper: Resize an RGB image within a tensor batch.
fn resize_rgb_image(
    tensor: &ArrayD<f32>,
    mut resized_tensor: ArrayD<f32>,
    shape: &[usize],
    b: usize,
    src_h: usize,
    src_w: usize,
    width: usize,
    height: usize,
    filter_type: image::imageops::FilterType,
) -> ExecutorResult<ArrayD<f32>> {
    use image::{ImageBuffer, Rgb, RgbImage};

    let mut img: RgbImage = ImageBuffer::new(src_w as u32, src_h as u32);
    for h in 0..src_h {
        for w in 0..src_w {
            let (r_idx, g_idx, b_idx) = if shape.len() == 4 {
                (vec![b, 0, h, w], vec![b, 1, h, w], vec![b, 2, h, w])
            } else {
                (vec![0, h, w], vec![1, h, w], vec![2, h, w])
            };

            let r = (tensor[IxDyn(&r_idx)] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (tensor[IxDyn(&g_idx)] * 255.0).clamp(0.0, 255.0) as u8;
            let b_val = (tensor[IxDyn(&b_idx)] * 255.0).clamp(0.0, 255.0) as u8;

            img.put_pixel(w as u32, h as u32, Rgb([r, g, b_val]));
        }
    }

    let resized = image::imageops::resize(&img, width as u32, height as u32, filter_type);

    for h in 0..height {
        for w in 0..width {
            let pixel = resized.get_pixel(w as u32, h as u32);
            let (r_idx, g_idx, b_idx) = if shape.len() == 4 {
                (vec![b, 0, h, w], vec![b, 1, h, w], vec![b, 2, h, w])
            } else {
                (vec![0, h, w], vec![1, h, w], vec![2, h, w])
            };

            resized_tensor[IxDyn(&r_idx)] = pixel[0] as f32 / 255.0;
            resized_tensor[IxDyn(&g_idx)] = pixel[1] as f32 / 255.0;
            resized_tensor[IxDyn(&b_idx)] = pixel[2] as f32 / 255.0;
        }
    }

    Ok(resized_tensor)
}

/// Helper: Resize a grayscale image within a tensor batch.
fn resize_grayscale_image(
    tensor: &ArrayD<f32>,
    mut resized_tensor: ArrayD<f32>,
    shape: &[usize],
    b: usize,
    src_h: usize,
    src_w: usize,
    width: usize,
    height: usize,
    filter_type: image::imageops::FilterType,
) -> ExecutorResult<ArrayD<f32>> {
    use image::{GrayImage, ImageBuffer, Luma};

    let mut img: GrayImage = ImageBuffer::new(src_w as u32, src_h as u32);
    for h in 0..src_h {
        for w in 0..src_w {
            let idx = if shape.len() == 4 {
                vec![b, 0, h, w]
            } else {
                vec![0, h, w]
            };
            let val = (tensor[IxDyn(&idx)] * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(w as u32, h as u32, Luma([val]));
        }
    }

    let resized = image::imageops::resize(&img, width as u32, height as u32, filter_type);

    for h in 0..height {
        for w in 0..width {
            let pixel = resized.get_pixel(w as u32, h as u32);
            let idx = if shape.len() == 4 {
                vec![b, 0, h, w]
            } else {
                vec![0, h, w]
            };
            resized_tensor[IxDyn(&idx)] = pixel[0] as f32 / 255.0;
        }
    }

    Ok(resized_tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution::{
            preprocessing::apply_preprocessing_step,
            template::{
                ImageNormalizePreset, ImageResizeMode, ImageTensorLayout, PreprocessingStep,
            },
        },
        ir::{Envelope, ImagePlane, PixelFormat, YuvColorInfo, YuvColorMatrix, YuvColorRange},
    };

    fn rgb_encoded_2x1(format: image::ImageFormat) -> Vec<u8> {
        let mut image = image::RgbImage::new(2, 1);
        image.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        image.put_pixel(1, 0, image::Rgb([0, 128, 255]));

        let mut encoded = std::io::Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, format)
            .expect("test image encodes");
        encoded.into_inner()
    }

    fn solid_rgb_encoded(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
        let image = image::RgbImage::from_pixel(width, height, image::Rgb(rgb));
        let mut encoded = std::io::Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, image::ImageFormat::Png)
            .expect("test image encodes");
        encoded.into_inner()
    }

    fn gradient_rgba_raw(width: u32, height: u32) -> (Vec<u8>, Vec<ImagePlane>) {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                pixels.push((x * 40 + y * 7) as u8);
                pixels.push((x * 5 + y * 50) as u8);
                pixels.push((x * 20 + y * 11) as u8);
                pixels.push(255);
            }
        }
        let planes = vec![ImagePlane {
            offset: 0,
            row_stride: width as usize * 4,
            pixel_stride: 4,
            width,
            height,
        }];
        (pixels, planes)
    }

    fn gradient_rgb_png(width: u32, height: u32) -> Vec<u8> {
        let mut image = image::RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                image.put_pixel(
                    x,
                    y,
                    image::Rgb([
                        (x * 40 + y * 7) as u8,
                        (x * 5 + y * 50) as u8,
                        (x * 20 + y * 11) as u8,
                    ]),
                );
            }
        }
        let mut encoded = std::io::Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, image::ImageFormat::Png)
            .expect("test image encodes");
        encoded.into_inner()
    }

    fn packed_raw_envelope(
        format: PixelFormat,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    ) -> Envelope {
        let pixel_stride = match format {
            PixelFormat::Rgb8 => 3,
            PixelFormat::Rgba8 | PixelFormat::Bgra8 => 4,
            PixelFormat::Nv12 | PixelFormat::Nv21 | PixelFormat::I420 => unreachable!(),
        };
        Envelope::image_raw(
            pixels,
            format,
            width,
            height,
            vec![ImagePlane {
                offset: 0,
                row_stride: width as usize * pixel_stride,
                pixel_stride,
                width,
                height,
            }],
            None,
        )
        .unwrap()
    }

    fn yuv_color() -> YuvColorInfo {
        YuvColorInfo {
            matrix: YuvColorMatrix::Bt709,
            range: YuvColorRange::Full,
        }
    }

    fn yuv420_envelope(format: PixelFormat) -> Envelope {
        let (pixels, planes) = match format {
            PixelFormat::Nv12 => (
                vec![128, 128, 128, 128, 255, 128],
                vec![
                    ImagePlane {
                        offset: 0,
                        row_stride: 2,
                        pixel_stride: 1,
                        width: 2,
                        height: 2,
                    },
                    ImagePlane {
                        offset: 4,
                        row_stride: 2,
                        pixel_stride: 2,
                        width: 1,
                        height: 1,
                    },
                ],
            ),
            PixelFormat::Nv21 => (
                vec![128, 128, 128, 128, 128, 255],
                vec![
                    ImagePlane {
                        offset: 0,
                        row_stride: 2,
                        pixel_stride: 1,
                        width: 2,
                        height: 2,
                    },
                    ImagePlane {
                        offset: 4,
                        row_stride: 2,
                        pixel_stride: 2,
                        width: 1,
                        height: 1,
                    },
                ],
            ),
            PixelFormat::I420 => (
                vec![128, 128, 128, 128, 255, 128],
                vec![
                    ImagePlane {
                        offset: 0,
                        row_stride: 2,
                        pixel_stride: 1,
                        width: 2,
                        height: 2,
                    },
                    ImagePlane {
                        offset: 4,
                        row_stride: 1,
                        pixel_stride: 1,
                        width: 1,
                        height: 1,
                    },
                    ImagePlane {
                        offset: 5,
                        row_stride: 1,
                        pixel_stride: 1,
                        width: 1,
                        height: 1,
                    },
                ],
            ),
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Bgra8 => unreachable!(),
        };

        Envelope::image_raw(pixels, format, 2, 2, planes, Some(yuv_color())).unwrap()
    }

    fn assert_nchw_pixel(tensor: &ArrayD<f32>, x: usize, y: usize, expected: [f32; 3]) {
        for (channel, expected) in expected.iter().enumerate() {
            let actual = tensor[[0, channel, y, x]];
            assert!(
                (actual - expected).abs() < 1e-6,
                "channel {channel} at ({x},{y}) expected {expected}, got {actual}"
            );
        }
    }

    fn assert_tensors_close(left: &ArrayD<f32>, right: &ArrayD<f32>) {
        assert_eq!(left.shape(), right.shape());
        for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
            assert!(
                (left - right).abs() < 1e-6,
                "tensor mismatch at flat index {index}: {left} vs {right}"
            );
        }
    }

    #[test]
    fn mobilenet_fixture_uses_metadata_driven_image_preprocessing() {
        use crate::execution::template::ModelMetadata;

        let model_dir = crate::testing::model_fixtures::model_path("mobilenet")
            .expect("mobilenet fixture directory exists");
        let metadata_content =
            std::fs::read_to_string(model_dir.join("model_metadata.json")).unwrap();
        let metadata: ModelMetadata = serde_json::from_str(&metadata_content).unwrap();

        assert!(
            matches!(
                metadata.preprocessing.first(),
                Some(PreprocessingStep::ImageDecode {
                    channels: 3,
                    layout: ImageTensorLayout::Nchw
                })
            ),
            "MobileNetV2 should start from encoded image envelopes"
        );
        assert!(
            metadata.preprocessing.iter().any(|step| matches!(
                step,
                PreprocessingStep::ImageNormalize {
                    preset: ImageNormalizePreset::ImageNet,
                    layout: ImageTensorLayout::Nchw
                }
            )),
            "MobileNetV2 should use the ImageNet normalization preset"
        );

        let envelope = Envelope::image(solid_rgb_encoded(320, 256, [255, 128, 0]), "png").unwrap();
        let mut data = PreprocessedData::from_envelope(&envelope).unwrap();
        for step in &metadata.preprocessing {
            data = apply_preprocessing_step(
                step,
                data,
                &envelope,
                model_dir.to_str().expect("fixture path is UTF-8"),
            )
            .unwrap();
        }

        match data {
            PreprocessedData::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
                let expected_r = (1.0 - 0.485) / 0.229;
                let expected_g = ((128.0 / 255.0) - 0.456) / 0.224;
                let expected_b = (0.0 - 0.406) / 0.225;
                assert!((tensor[[0, 0, 112, 112]] - expected_r).abs() < 1e-6);
                assert!((tensor[[0, 1, 112, 112]] - expected_g).abs() < 1e-6);
                assert!((tensor[[0, 2, 112, 112]] - expected_b).abs() < 1e-6);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn mobilenet_download_manifest_preserves_image_preprocessing_metadata() {
        use crate::execution::template::ModelMetadata;

        let models_dir =
            crate::testing::model_fixtures::models_dir().expect("fixtures model directory exists");
        let manifest_content = std::fs::read_to_string(models_dir.join("models.json")).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(&manifest_content).unwrap();
        let metadata_value = manifest
            .pointer("/models/mobilenet/model_metadata")
            .expect("mobilenet manifest metadata exists")
            .clone();
        let metadata: ModelMetadata = serde_json::from_value(metadata_value).unwrap();

        assert!(
            matches!(
                metadata.preprocessing.first(),
                Some(PreprocessingStep::ImageDecode {
                    channels: 3,
                    layout: ImageTensorLayout::Nchw
                })
            ),
            "download.sh mobilenet should regenerate image-envelope metadata"
        );
        assert!(
            metadata.preprocessing.iter().any(|step| matches!(
                step,
                PreprocessingStep::ImageNormalize {
                    preset: ImageNormalizePreset::ImageNet,
                    layout: ImageTensorLayout::Nchw
                }
            )),
            "download.sh mobilenet should preserve the ImageNet normalization preset"
        );
    }

    #[test]
    fn image_decode_step_emits_nchw_tensor_from_envelope_image() {
        let envelope = Envelope::image(rgb_encoded_2x1(image::ImageFormat::Png), "png").unwrap();
        let data = PreprocessedData::from_envelope(&envelope).unwrap();

        let result = image_decode_step(data, 3, ImageTensorLayout::Nchw).unwrap();

        match result {
            PreprocessedData::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[1, 3, 1, 2]);
                assert!((tensor[[0, 0, 0, 0]] - 1.0).abs() < 1e-6);
                assert!((tensor[[0, 1, 0, 0]] - 0.0).abs() < 1e-6);
                assert!((tensor[[0, 2, 0, 0]] - 0.0).abs() < 1e-6);
                assert!((tensor[[0, 0, 0, 1]] - 0.0).abs() < 1e-6);
                assert!((tensor[[0, 1, 0, 1]] - (128.0 / 255.0)).abs() < 1e-6);
                assert!((tensor[[0, 2, 0, 1]] - 1.0).abs() < 1e-6);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn image_decode_step_emits_nhwc_tensor_from_envelope_image() {
        let envelope = Envelope::image(rgb_encoded_2x1(image::ImageFormat::Png), "png").unwrap();
        let data = PreprocessedData::from_envelope(&envelope).unwrap();

        let result = image_decode_step(data, 3, ImageTensorLayout::Nhwc).unwrap();

        match result {
            PreprocessedData::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[1, 1, 2, 3]);
                assert!((tensor[[0, 0, 0, 0]] - 1.0).abs() < 1e-6);
                assert!((tensor[[0, 0, 0, 1]] - 0.0).abs() < 1e-6);
                assert!((tensor[[0, 0, 0, 2]] - 0.0).abs() < 1e-6);
                assert!((tensor[[0, 0, 1, 0]] - 0.0).abs() < 1e-6);
                assert!((tensor[[0, 0, 1, 1]] - (128.0 / 255.0)).abs() < 1e-6);
                assert!((tensor[[0, 0, 1, 2]] - 1.0).abs() < 1e-6);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn image_decode_preprocessing_step_is_dispatched() {
        let envelope = Envelope::image(rgb_encoded_2x1(image::ImageFormat::Png), "png").unwrap();
        let data = PreprocessedData::from_envelope(&envelope).unwrap();
        let step = PreprocessingStep::ImageDecode {
            channels: 3,
            layout: ImageTensorLayout::Nchw,
        };

        let result = apply_preprocessing_step(&step, data, &envelope, "").unwrap();

        match result {
            PreprocessedData::Tensor(tensor) => assert_eq!(tensor.shape(), &[1, 3, 1, 2]),
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn image_ingress_matches_image_decode_for_encoded_image_sources() {
        let envelope = Envelope::image(rgb_encoded_2x1(image::ImageFormat::Png), "png").unwrap();
        let decoded = image_decode_step(
            PreprocessedData::from_envelope(&envelope).unwrap(),
            3,
            ImageTensorLayout::Nchw,
        )
        .unwrap();
        let ingressed = image_ingress_step(
            PreprocessedData::from_envelope(&envelope).unwrap(),
            3,
            ImageTensorLayout::Nchw,
        )
        .unwrap();

        match (decoded, ingressed) {
            (PreprocessedData::Tensor(decoded), PreprocessedData::Tensor(ingressed)) => {
                assert_tensors_close(&decoded, &ingressed);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn image_ingress_step_accepts_raw_rgb_family_sources() {
        let cases = [
            (PixelFormat::Rgb8, vec![255, 0, 0, 0, 128, 255]),
            (PixelFormat::Rgba8, vec![255, 0, 0, 99, 0, 128, 255, 100]),
            (PixelFormat::Bgra8, vec![0, 0, 255, 99, 255, 128, 0, 100]),
        ];

        for (format, pixels) in cases {
            let envelope = packed_raw_envelope(format, 2, 1, pixels);
            let data = PreprocessedData::from_envelope(&envelope).unwrap();

            let result = image_ingress_step(data, 3, ImageTensorLayout::Nchw).unwrap();

            match result {
                PreprocessedData::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), &[1, 3, 1, 2]);
                    assert_nchw_pixel(&tensor, 0, 0, [1.0, 0.0, 0.0]);
                    assert_nchw_pixel(&tensor, 1, 0, [0.0, 128.0 / 255.0, 1.0]);
                }
                _ => panic!("Expected Tensor output"),
            }
        }
    }

    #[test]
    fn image_ingress_step_accepts_raw_yuv420_sources() {
        for format in [PixelFormat::Nv12, PixelFormat::Nv21, PixelFormat::I420] {
            let envelope = yuv420_envelope(format);
            let data = PreprocessedData::from_envelope(&envelope).unwrap();

            let result = image_ingress_step(data, 3, ImageTensorLayout::Nchw).unwrap();

            match result {
                PreprocessedData::Tensor(tensor) => {
                    assert_eq!(tensor.shape(), &[1, 3, 2, 2]);
                    for y in 0..2 {
                        for x in 0..2 {
                            let expected = [128.0 / 255.0, 104.0 / 255.0, 1.0];
                            for (channel, expected) in expected.iter().enumerate() {
                                let actual = tensor[[0, channel, y, x]];
                                assert!(
                                    (actual - expected).abs() <= (2.0 / 255.0),
                                    "{format} channel {channel} at ({x},{y}) expected about {expected}, got {actual}"
                                );
                            }
                        }
                    }
                }
                _ => panic!("Expected Tensor output"),
            }
        }
    }

    #[test]
    fn image_ingress_preprocessing_step_is_dispatched() {
        let envelope = packed_raw_envelope(
            PixelFormat::Rgba8,
            2,
            1,
            vec![255, 0, 0, 255, 0, 128, 255, 255],
        );
        let data = PreprocessedData::from_envelope(&envelope).unwrap();
        let step = PreprocessingStep::ImageIngress {
            channels: 3,
            layout: ImageTensorLayout::Nchw,
        };

        let result = apply_preprocessing_step(&step, data, &envelope, "").unwrap();

        match result {
            PreprocessedData::Tensor(tensor) => assert_eq!(tensor.shape(), &[1, 3, 1, 2]),
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn raw_image_ingress_matches_encoded_png_preprocessing_chain() {
        let (raw_pixels, raw_planes) = gradient_rgba_raw(4, 4);
        let raw =
            Envelope::image_raw(raw_pixels, PixelFormat::Rgba8, 4, 4, raw_planes, None).unwrap();
        let encoded = Envelope::image(gradient_rgb_png(4, 4), "png").unwrap();
        let steps = [
            PreprocessingStep::ImageIngress {
                channels: 3,
                layout: ImageTensorLayout::Nchw,
            },
            PreprocessingStep::ImageResize {
                width: 2,
                height: 2,
                mode: ImageResizeMode::Stretch,
                interpolation: InterpolationMethod::Nearest,
                fill: vec![0.0, 0.0, 0.0],
                layout: ImageTensorLayout::Nchw,
            },
            PreprocessingStep::ImageNormalize {
                preset: ImageNormalizePreset::Custom {
                    mean: vec![0.1, 0.2, 0.3],
                    std: vec![0.9, 0.8, 0.7],
                },
                layout: ImageTensorLayout::Nchw,
            },
        ];

        let mut raw_data = PreprocessedData::from_envelope(&raw).unwrap();
        for step in &steps {
            raw_data = apply_preprocessing_step(step, raw_data, &raw, "").unwrap();
        }

        let mut encoded_data = PreprocessedData::from_envelope(&encoded).unwrap();
        for step in &steps {
            encoded_data = apply_preprocessing_step(step, encoded_data, &encoded, "").unwrap();
        }

        match (raw_data, encoded_data) {
            (PreprocessedData::Tensor(raw), PreprocessedData::Tensor(encoded)) => {
                assert_tensors_close(&raw, &encoded);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn raw_image_ingress_trace_preserves_one_pixel_buffer_copy() {
        let _trace_lock = crate::tracing::test_lock();
        let (raw_pixels, raw_planes) = gradient_rgba_raw(4, 4);
        let raw =
            Envelope::image_raw(raw_pixels, PixelFormat::Rgba8, 4, 4, raw_planes, None).unwrap();
        let envelope_pixel_ptr = raw.as_raw_image().unwrap().pixels.as_ptr();

        crate::tracing::init_tracing(true);
        crate::tracing::reset_tracing();
        {
            let _span = crate::tracing::SpanGuard::new("raw_image_copy_budget");
            let data = PreprocessedData::from_envelope(&raw).unwrap();
            let preprocessed_pixel_ptr = match &data {
                PreprocessedData::Image { source } => source.as_raw().unwrap().pixels.as_ptr(),
                other => panic!("expected preprocessed image, got {other:?}"),
            };
            assert_eq!(
                preprocessed_pixel_ptr, envelope_pixel_ptr,
                "raw image preprocessing must not clone the pixel buffer before ImageIngress"
            );

            image_ingress_step(data, 3, ImageTensorLayout::Nchw).unwrap();
        }

        let trace = crate::tracing::get_stages_json();
        crate::tracing::reset_tracing();
        let span = trace["spans"]
            .as_array()
            .unwrap()
            .iter()
            .find(|span| span["name"] == "raw_image_copy_budget")
            .expect("copy budget span");
        let metadata = &span["metadata"];
        assert_eq!(metadata["raw_image_source_vec_allocations"], "1");
        assert_eq!(metadata["raw_image_source_extra_pixel_copies"], "0");
        assert_eq!(metadata["raw_image_conversion_buffer_allocations"], "0");
        assert_eq!(metadata["raw_image_pixel_bytes"], "64");
    }

    #[test]
    fn raw_yuv_image_ingress_trace_reports_conversion_buffer_allocation() {
        let _trace_lock = crate::tracing::test_lock();
        for format in [PixelFormat::Nv12, PixelFormat::Nv21, PixelFormat::I420] {
            let raw = yuv420_envelope(format);

            crate::tracing::init_tracing(true);
            crate::tracing::reset_tracing();
            {
                let _span = crate::tracing::SpanGuard::new("raw_yuv_image_copy_budget");
                let data = PreprocessedData::from_envelope(&raw).unwrap();
                image_ingress_step(data, 3, ImageTensorLayout::Nchw).unwrap();
            }

            let trace = crate::tracing::get_stages_json();
            crate::tracing::reset_tracing();
            let span = trace["spans"]
                .as_array()
                .unwrap()
                .iter()
                .find(|span| span["name"] == "raw_yuv_image_copy_budget")
                .unwrap_or_else(|| panic!("copy budget span for {format}"));
            let metadata = &span["metadata"];
            assert_eq!(metadata["raw_image_source_vec_allocations"], "1");
            assert_eq!(metadata["raw_image_source_extra_pixel_copies"], "0");
            assert_eq!(
                metadata["raw_image_conversion_buffer_allocations"], "1",
                "{format} currently uses one yuvutils RGB conversion buffer before tensor emission"
            );
            assert_eq!(metadata["raw_image_pixel_bytes"], "6");
        }
    }

    #[test]
    fn raw_1920x1080_image_ingress_matches_encoded_png_preprocessing_chain() {
        let (raw_pixels, raw_planes) = gradient_rgba_raw(1920, 1080);
        let raw = Envelope::image_raw(raw_pixels, PixelFormat::Rgba8, 1920, 1080, raw_planes, None)
            .unwrap();
        let encoded = Envelope::image(gradient_rgb_png(1920, 1080), "png").unwrap();
        let steps = [
            PreprocessingStep::ImageIngress {
                channels: 3,
                layout: ImageTensorLayout::Nchw,
            },
            PreprocessingStep::ImageResize {
                width: 224,
                height: 224,
                mode: ImageResizeMode::Stretch,
                interpolation: InterpolationMethod::Nearest,
                fill: vec![0.0, 0.0, 0.0],
                layout: ImageTensorLayout::Nchw,
            },
            PreprocessingStep::ImageNormalize {
                preset: ImageNormalizePreset::Custom {
                    mean: vec![0.1, 0.2, 0.3],
                    std: vec![0.9, 0.8, 0.7],
                },
                layout: ImageTensorLayout::Nchw,
            },
        ];

        let mut raw_data = PreprocessedData::from_envelope(&raw).unwrap();
        for step in &steps {
            raw_data = apply_preprocessing_step(step, raw_data, &raw, "").unwrap();
        }

        let mut encoded_data = PreprocessedData::from_envelope(&encoded).unwrap();
        for step in &steps {
            encoded_data = apply_preprocessing_step(step, encoded_data, &encoded, "").unwrap();
        }

        match (raw_data, encoded_data) {
            (PreprocessedData::Tensor(raw), PreprocessedData::Tensor(encoded)) => {
                assert_tensors_close(&raw, &encoded);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn image_decode_step_accepts_jpeg_and_webp() {
        let cases = [
            ("jpeg", image::ImageFormat::Jpeg),
            ("webp", image::ImageFormat::WebP),
        ];

        for (hint, format) in cases {
            let envelope = Envelope::image(rgb_encoded_2x1(format), hint).unwrap();
            let data = PreprocessedData::from_envelope(&envelope).unwrap();
            let result = image_decode_step(data, 3, ImageTensorLayout::Nchw).unwrap();

            match result {
                PreprocessedData::Tensor(tensor) => assert_eq!(tensor.shape(), &[1, 3, 1, 2]),
                _ => panic!("Expected Tensor output"),
            }
        }
    }

    #[test]
    fn image_resize_letterbox_preserves_aspect_ratio_and_pads() {
        let mut data = ndarray::Array4::<f32>::zeros((1, 3, 2, 4));
        for y in 0..2 {
            for x in 0..4 {
                data[[0, 0, y, x]] = 1.0;
                data[[0, 1, y, x]] = 0.0;
                data[[0, 2, y, x]] = 0.0;
            }
        }

        let result = image_resize_step(
            PreprocessedData::Tensor(data.into_dyn()),
            4,
            4,
            ImageResizeMode::Letterbox,
            &InterpolationMethod::Nearest,
            &[0.25, 0.5, 0.75],
            ImageTensorLayout::Nchw,
        )
        .unwrap();

        match result {
            PreprocessedData::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[1, 3, 4, 4]);
                assert!((tensor[[0, 0, 0, 0]] - 0.25).abs() < 1e-6);
                assert!((tensor[[0, 1, 0, 0]] - 0.5).abs() < 1e-6);
                assert!((tensor[[0, 2, 0, 0]] - 0.75).abs() < 1e-6);
                assert!((tensor[[0, 0, 1, 0]] - 1.0).abs() < 1e-6);
                assert!((tensor[[0, 1, 1, 0]] - 0.0).abs() < 1e-6);
                assert!((tensor[[0, 2, 1, 0]] - 0.0).abs() < 1e-6);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn image_resize_center_preserves_aspect_ratio_and_crops_center() {
        let mut data = ndarray::Array4::<f32>::zeros((1, 3, 2, 4));
        for x in 0..4 {
            data[[0, 0, 0, x]] = x as f32 / 3.0;
            data[[0, 1, 0, x]] = 0.0;
            data[[0, 2, 0, x]] = 0.0;
            data[[0, 0, 1, x]] = x as f32 / 3.0;
            data[[0, 1, 1, x]] = 0.0;
            data[[0, 2, 1, x]] = 0.0;
        }

        let result = image_resize_step(
            PreprocessedData::Tensor(data.into_dyn()),
            2,
            2,
            ImageResizeMode::Center,
            &InterpolationMethod::Nearest,
            &[0.0, 0.0, 0.0],
            ImageTensorLayout::Nchw,
        )
        .unwrap();

        match result {
            PreprocessedData::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[1, 3, 2, 2]);
                assert!((tensor[[0, 0, 0, 0]] - (1.0 / 3.0)).abs() < 1e-6);
                assert!((tensor[[0, 0, 0, 1]] - (2.0 / 3.0)).abs() < 1e-6);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn image_resize_preprocessing_step_dispatches_letterbox() {
        let data = ndarray::Array4::<f32>::zeros((1, 3, 2, 4)).into_dyn();
        let input = PreprocessedData::Tensor(data);
        let step = PreprocessingStep::ImageResize {
            width: 4,
            height: 4,
            mode: ImageResizeMode::Letterbox,
            interpolation: InterpolationMethod::Nearest,
            fill: vec![0.25, 0.5, 0.75],
            layout: ImageTensorLayout::Nchw,
        };

        let result = apply_preprocessing_step(
            &step,
            input,
            &Envelope::new(crate::ir::EnvelopeKind::Text(String::new())),
            "",
        )
        .unwrap();

        match result {
            PreprocessedData::Tensor(tensor) => assert_eq!(tensor.shape(), &[1, 3, 4, 4]),
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn image_normalize_presets_match_reference_outputs() {
        let cases = [
            (
                ImageNormalizePreset::ImageNet,
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
            (
                ImageNormalizePreset::Clip,
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.261_302_6, 0.275_777_1],
            ),
            (
                ImageNormalizePreset::SigLip,
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ];

        for (preset, mean, std) in cases {
            let data = ndarray::Array4::from_shape_vec((1, 3, 1, 1), vec![1.0, 0.5, 0.0])
                .unwrap()
                .into_dyn();
            let result = image_normalize_step(
                PreprocessedData::Tensor(data),
                &preset,
                ImageTensorLayout::Nchw,
            )
            .unwrap();

            match result {
                PreprocessedData::Tensor(tensor) => {
                    for channel in 0..3 {
                        let input = [1.0, 0.5, 0.0][channel];
                        let expected = (input - mean[channel]) / std[channel];
                        assert!((tensor[[0, channel, 0, 0]] - expected).abs() < 1e-6);
                    }
                }
                _ => panic!("Expected Tensor output"),
            }
        }
    }

    #[test]
    fn image_normalize_custom_preset_and_dispatcher_work() {
        let data = ndarray::Array4::from_shape_vec((1, 3, 1, 1), vec![0.2, 0.4, 0.6])
            .unwrap()
            .into_dyn();
        let step = PreprocessingStep::ImageNormalize {
            preset: ImageNormalizePreset::Custom {
                mean: vec![0.1, 0.2, 0.3],
                std: vec![0.1, 0.2, 0.3],
            },
            layout: ImageTensorLayout::Nchw,
        };

        let result = apply_preprocessing_step(
            &step,
            PreprocessedData::Tensor(data),
            &Envelope::new(crate::ir::EnvelopeKind::Text(String::new())),
            "",
        )
        .unwrap();

        match result {
            PreprocessedData::Tensor(tensor) => {
                assert!((tensor[[0, 0, 0, 0]] - 1.0).abs() < 1e-6);
                assert!((tensor[[0, 1, 0, 0]] - 1.0).abs() < 1e-6);
                assert!((tensor[[0, 2, 0, 0]] - 1.0).abs() < 1e-6);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_center_crop_step_basic() {
        let data = ndarray::Array4::<f32>::zeros((1, 3, 100, 100)).into_dyn();
        let input = PreprocessedData::Tensor(data);

        let result = center_crop_step(input, 50, 50);

        assert!(result.is_ok());
        match result.unwrap() {
            PreprocessedData::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[1, 3, 50, 50]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_center_crop_step_3d_tensor() {
        let data = ndarray::Array3::<f32>::zeros((3, 64, 64)).into_dyn();
        let input = PreprocessedData::Tensor(data);

        let result = center_crop_step(input, 32, 32);

        assert!(result.is_ok());
        match result.unwrap() {
            PreprocessedData::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[3, 32, 32]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_center_crop_step_larger_than_input() {
        let data = ndarray::Array4::<f32>::zeros((1, 3, 50, 50)).into_dyn();
        let input = PreprocessedData::Tensor(data);

        let result = center_crop_step(input, 100, 100);

        assert!(result.is_err());
    }

    #[test]
    fn test_resize_step_rgb_upscale() {
        let mut data = ndarray::Array4::<f32>::zeros((1, 3, 4, 4));
        data[[0, 0, 0, 0]] = 1.0;
        data[[0, 1, 0, 0]] = 0.5;
        data[[0, 2, 0, 0]] = 0.0;

        let input = PreprocessedData::Tensor(data.into_dyn());

        let result = resize_step(input, 8, 8, &InterpolationMethod::Bilinear);

        assert!(result.is_ok());
        match result.unwrap() {
            PreprocessedData::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[1, 3, 8, 8]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_resize_step_grayscale() {
        let data = ndarray::Array4::<f32>::zeros((1, 1, 32, 32)).into_dyn();
        let input = PreprocessedData::Tensor(data);

        let result = resize_step(input, 64, 64, &InterpolationMethod::Bicubic);

        assert!(result.is_ok());
        match result.unwrap() {
            PreprocessedData::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[1, 1, 64, 64]);
            }
            _ => panic!("Expected Tensor output"),
        }
    }

    #[test]
    fn test_resize_step_invalid_channels() {
        let data = ndarray::Array4::<f32>::zeros((1, 5, 10, 10)).into_dyn();
        let input = PreprocessedData::Tensor(data);

        let result = resize_step(input, 20, 20, &InterpolationMethod::Nearest);

        assert!(result.is_err());
    }

    fn bt601_full_color() -> YuvColorInfo {
        YuvColorInfo {
            matrix: YuvColorMatrix::Bt601,
            range: YuvColorRange::Full,
        }
    }

    /// 2x2 packed RGB-family raw envelope with an intentionally padded row
    /// stride (one trailing byte per row) so the packed converter must honor
    /// `row_stride > width * pixel_stride`.
    fn padded_packed_raw_envelope(format: PixelFormat, rows: [[u8; 8]; 2]) -> Envelope {
        let pixel_stride = match format {
            PixelFormat::Rgb8 => 3,
            PixelFormat::Rgba8 | PixelFormat::Bgra8 => 4,
            PixelFormat::Nv12 | PixelFormat::Nv21 | PixelFormat::I420 => unreachable!(),
        };
        let row_stride = pixel_stride * 2 + 1; // pad each row by one byte
        let mut pixels = Vec::new();
        for row in rows {
            pixels.extend_from_slice(&row[..pixel_stride * 2]);
            pixels.push(0xAA); // padding byte the converter must skip
        }
        Envelope::image_raw(
            pixels,
            format,
            2,
            2,
            vec![ImagePlane {
                offset: 0,
                row_stride,
                pixel_stride,
                width: 2,
                height: 2,
            }],
            None,
        )
        .unwrap()
    }

    #[test]
    fn raw_image_to_packed_rgb_strips_stride_for_rgb8() {
        // Two rows of two RGB pixels, each row padded by one trailing byte.
        let rows = [
            [255, 0, 0, 0, 255, 0, 0, 0], // (255,0,0) (0,255,0)
            [0, 0, 255, 9, 8, 7, 0, 0],   // (0,0,255) (9,8,7)
        ];
        let envelope = padded_packed_raw_envelope(PixelFormat::Rgb8, rows);
        let raw = envelope.as_raw_image().unwrap();

        let rgb = raw_image_to_packed_rgb(raw).unwrap();

        assert_eq!(rgb.len(), 2 * 2 * 3);
        assert_eq!(&rgb[0..3], &[255, 0, 0]);
        assert_eq!(&rgb[3..6], &[0, 255, 0]);
        assert_eq!(&rgb[6..9], &[0, 0, 255]);
        assert_eq!(&rgb[9..12], &[9, 8, 7]);
    }

    #[test]
    fn raw_image_to_packed_rgb_swaps_channels_and_strips_alpha_for_bgra8() {
        // BGRA pixels: stored B,G,R,A; expect channel-swapped RGB with alpha dropped.
        let rows = [
            [0, 0, 255, 99, 0, 255, 0, 88], // -> (255,0,0) (0,255,0)
            [255, 0, 0, 77, 7, 8, 9, 66],   // -> (0,0,255) (9,8,7)
        ];
        let envelope = padded_packed_raw_envelope(PixelFormat::Bgra8, rows);
        let raw = envelope.as_raw_image().unwrap();

        let rgb = raw_image_to_packed_rgb(raw).unwrap();

        assert_eq!(rgb.len(), 2 * 2 * 3);
        assert_eq!(&rgb[0..3], &[255, 0, 0]);
        assert_eq!(&rgb[3..6], &[0, 255, 0]);
        assert_eq!(&rgb[6..9], &[0, 0, 255]);
        assert_eq!(&rgb[9..12], &[9, 8, 7]);
    }

    #[test]
    fn raw_image_to_packed_rgb_decodes_neutral_gray_yuv_exactly() {
        // Y=U=V=128 is neutral gray; BT.601 full-range maps it back to (128,128,128)
        // for every 4:2:0 layout. Chroma planes carry pixel_stride 2 (NV12/NV21).
        for format in [PixelFormat::Nv12, PixelFormat::Nv21, PixelFormat::I420] {
            let (pixels, planes) = match format {
                PixelFormat::Nv12 | PixelFormat::Nv21 => (
                    vec![128, 128, 128, 128, 128, 128],
                    vec![
                        ImagePlane {
                            offset: 0,
                            row_stride: 2,
                            pixel_stride: 1,
                            width: 2,
                            height: 2,
                        },
                        ImagePlane {
                            offset: 4,
                            row_stride: 2,
                            pixel_stride: 2,
                            width: 1,
                            height: 1,
                        },
                    ],
                ),
                PixelFormat::I420 => (
                    vec![128, 128, 128, 128, 128, 128],
                    vec![
                        ImagePlane {
                            offset: 0,
                            row_stride: 2,
                            pixel_stride: 1,
                            width: 2,
                            height: 2,
                        },
                        ImagePlane {
                            offset: 4,
                            row_stride: 1,
                            pixel_stride: 1,
                            width: 1,
                            height: 1,
                        },
                        ImagePlane {
                            offset: 5,
                            row_stride: 1,
                            pixel_stride: 1,
                            width: 1,
                            height: 1,
                        },
                    ],
                ),
                _ => unreachable!(),
            };
            let envelope =
                Envelope::image_raw(pixels, format, 2, 2, planes, Some(bt601_full_color()))
                    .unwrap();
            let raw = envelope.as_raw_image().unwrap();

            let rgb = raw_image_to_packed_rgb(raw).unwrap();

            assert_eq!(rgb.len(), 2 * 2 * 3, "{format} packed length");
            for (i, value) in rgb.iter().enumerate() {
                assert!(
                    value.abs_diff(128) <= 1,
                    "{format} byte {i} expected ~128, got {value}"
                );
            }
        }
    }

    #[test]
    fn raw_image_to_packed_rgb_applies_bt601_full_range_chroma() {
        // Y=128, U=128, V=200 (NV12 stores U then V). BT.601 full-range:
        //   R = Y + 1.402 * (V-128)          = 128 + 1.402*72  ≈ 229
        //   G = Y - 0.344*(U-128) - 0.714*(V-128) ≈ 128 - 51.4  ≈ 77
        //   B = Y + 1.772 * (U-128)          = 128
        let pixels = vec![128, 128, 128, 128, 128, 200];
        let planes = vec![
            ImagePlane {
                offset: 0,
                row_stride: 2,
                pixel_stride: 1,
                width: 2,
                height: 2,
            },
            ImagePlane {
                offset: 4,
                row_stride: 2,
                pixel_stride: 2,
                width: 1,
                height: 1,
            },
        ];
        let envelope = Envelope::image_raw(
            pixels,
            PixelFormat::Nv12,
            2,
            2,
            planes,
            Some(bt601_full_color()),
        )
        .unwrap();
        let raw = envelope.as_raw_image().unwrap();

        let rgb = raw_image_to_packed_rgb(raw).unwrap();

        assert_eq!(rgb.len(), 2 * 2 * 3);
        // Every pixel shares the single chroma sample.
        for pixel in rgb.chunks_exact(3) {
            assert!(
                pixel[0].abs_diff(229) <= 2,
                "R expected ~229, got {}",
                pixel[0]
            );
            assert!(
                pixel[1].abs_diff(77) <= 2,
                "G expected ~77, got {}",
                pixel[1]
            );
            assert!(
                pixel[2].abs_diff(128) <= 2,
                "B expected ~128, got {}",
                pixel[2]
            );
        }
    }

    #[test]
    fn raw_image_to_packed_rgb_matches_tensor_path_for_packed_sources() {
        // The packed-RGB byte path must agree with the tensor path's per-pixel
        // channel handling for the shared RGB-family formats.
        let cases = [
            (PixelFormat::Rgb8, vec![255, 0, 0, 0, 128, 255]),
            (PixelFormat::Rgba8, vec![255, 0, 0, 99, 0, 128, 255, 100]),
            (PixelFormat::Bgra8, vec![0, 0, 255, 99, 255, 128, 0, 100]),
        ];

        for (format, pixels) in cases {
            let envelope = packed_raw_envelope(format, 2, 1, pixels);
            let raw = envelope.as_raw_image().unwrap();
            let rgb = raw_image_to_packed_rgb(raw).unwrap();

            assert_eq!(rgb.len(), 2 * 3, "{format} packed length");
            assert_eq!(&rgb[0..3], &[255, 0, 0], "{format} pixel 0");
            assert_eq!(&rgb[3..6], &[0, 128, 255], "{format} pixel 1");
        }
    }
}
