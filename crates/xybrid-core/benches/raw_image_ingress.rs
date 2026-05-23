//! Criterion bench for raw camera-frame image ingress (INF-248).
//!
//! SLO defended:
//!   * 1920x1080 NV12 `ImageIngress` snapshot: <= 8 ms on the 2024 M3 MacBook Air baseline.
//!
//! This measures the real raw-envelope -> `ImageIngress` -> tensor path,
//! including validation, YUV conversion, and tensor emission.
//!
//! Run with:
//!   cargo bench -p xybrid-core --bench raw_image_ingress --features vision,dev-tools

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use xybrid_core::execution::{test_seams::image_ingress_tensor, ImageTensorLayout};
use xybrid_core::ir::{
    Envelope, ImagePlane, PixelFormat, YuvColorInfo, YuvColorMatrix, YuvColorRange,
};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

fn nv12_1920x1080_frame() -> Envelope {
    let width = WIDTH as usize;
    let height = HEIGHT as usize;
    let y_bytes = width * height;
    let uv_bytes = y_bytes / 2;

    let mut pixels = vec![128_u8; y_bytes + uv_bytes];
    for uv in pixels[y_bytes..].chunks_exact_mut(2) {
        uv[0] = 128;
        uv[1] = 128;
    }

    Envelope::image_raw(
        pixels,
        PixelFormat::Nv12,
        WIDTH,
        HEIGHT,
        vec![
            ImagePlane {
                offset: 0,
                row_stride: width,
                pixel_stride: 1,
                width: WIDTH,
                height: HEIGHT,
            },
            ImagePlane {
                offset: y_bytes,
                row_stride: width,
                pixel_stride: 2,
                width: WIDTH / 2,
                height: HEIGHT / 2,
            },
        ],
        Some(YuvColorInfo {
            matrix: YuvColorMatrix::Bt709,
            range: YuvColorRange::Full,
        }),
    )
    .expect("valid NV12 fixture")
}

fn bench_nv12_image_ingress(c: &mut Criterion) {
    let envelope = nv12_1920x1080_frame();
    let tensor = image_ingress_tensor(&envelope, 3, ImageTensorLayout::Nchw)
        .expect("NV12 fixture ingresses");
    assert_eq!(tensor.shape(), &[1, 3, HEIGHT as usize, WIDTH as usize]);

    c.bench_function("image_ingress::nv12_1920x1080_nchw", |b| {
        b.iter(|| {
            let tensor = image_ingress_tensor(
                black_box(&envelope),
                black_box(3),
                black_box(ImageTensorLayout::Nchw),
            )
            .expect("NV12 fixture ingresses");
            black_box(tensor);
        })
    });
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(20)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_nv12_image_ingress
}
criterion_main!(benches);
