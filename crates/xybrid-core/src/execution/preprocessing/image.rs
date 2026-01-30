//! Image preprocessing operations.
//!
//! This module provides:
//! - `center_crop_step`: Center crop image tensor to target dimensions
//! - `resize_step`: Resize image tensor using interpolation

use super::super::types::{ExecutorResult, PreprocessedData};
use crate::execution::template::InterpolationMethod;
use crate::runtime_adapter::AdapterError;
use ndarray::{ArrayD, IxDyn};

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
}
