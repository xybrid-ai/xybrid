use xybrid_core::{
    conversation::ConversationContext,
    ir::{
        envelope::EnvelopeError, Envelope, ImageDimensions, ImageFormat, ImagePlane,
        ImageValidationLimits, MessageRole, PixelFormat, YuvColorInfo, YuvColorMatrix,
        YuvColorRange,
    },
};

fn encoded_test_image(width: u32, height: u32, format: image::ImageFormat) -> Vec<u8> {
    let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        width,
        height,
        image::Rgb([17, 34, 51]),
    ));
    let mut encoded = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut encoded, format)
        .expect("test image encodes");
    encoded.into_inner()
}

fn png_image(width: u32, height: u32) -> Vec<u8> {
    encoded_test_image(width, height, image::ImageFormat::Png)
}

fn jpeg_image(width: u32, height: u32) -> Vec<u8> {
    encoded_test_image(width, height, image::ImageFormat::Jpeg)
}

fn webp_image(width: u32, height: u32) -> Vec<u8> {
    encoded_test_image(width, height, image::ImageFormat::WebP)
}

fn yuv_color() -> YuvColorInfo {
    YuvColorInfo {
        matrix: YuvColorMatrix::Bt709,
        range: YuvColorRange::Full,
    }
}

fn raw_case(
    format: PixelFormat,
    width: u32,
    height: u32,
) -> (Vec<u8>, Vec<ImagePlane>, Option<YuvColorInfo>) {
    let width_usize = width as usize;
    let height_usize = height as usize;
    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let chroma_width_usize = chroma_width as usize;
    let chroma_height_usize = chroma_height as usize;

    match format {
        PixelFormat::Rgb8 => {
            let row_stride = width_usize * 3;
            let pixels = vec![7; row_stride * height_usize];
            let planes = vec![ImagePlane {
                offset: 0,
                row_stride,
                pixel_stride: 3,
                width,
                height,
            }];
            (pixels, planes, None)
        }
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => {
            let row_stride = width_usize * 4;
            let pixels = vec![7; row_stride * height_usize];
            let planes = vec![ImagePlane {
                offset: 0,
                row_stride,
                pixel_stride: 4,
                width,
                height,
            }];
            (pixels, planes, None)
        }
        PixelFormat::Nv12 | PixelFormat::Nv21 => {
            let y_stride = width_usize;
            let uv_stride = chroma_width_usize * 2;
            let uv_offset = y_stride * height_usize;
            let pixels = vec![7; uv_offset + uv_stride * chroma_height_usize];
            let planes = vec![
                ImagePlane {
                    offset: 0,
                    row_stride: y_stride,
                    pixel_stride: 1,
                    width,
                    height,
                },
                ImagePlane {
                    offset: uv_offset,
                    row_stride: uv_stride,
                    pixel_stride: 2,
                    width: chroma_width,
                    height: chroma_height,
                },
            ];
            (pixels, planes, Some(yuv_color()))
        }
        PixelFormat::I420 => {
            let y_stride = width_usize;
            let u_stride = chroma_width_usize;
            let y_len = y_stride * height_usize;
            let u_len = u_stride * chroma_height_usize;
            let v_offset = y_len + u_len;
            let pixels = vec![7; v_offset + u_len];
            let planes = vec![
                ImagePlane {
                    offset: 0,
                    row_stride: y_stride,
                    pixel_stride: 1,
                    width,
                    height,
                },
                ImagePlane {
                    offset: y_len,
                    row_stride: u_stride,
                    pixel_stride: 1,
                    width: chroma_width,
                    height: chroma_height,
                },
                ImagePlane {
                    offset: v_offset,
                    row_stride: u_stride,
                    pixel_stride: 1,
                    width: chroma_width,
                    height: chroma_height,
                },
            ];
            (pixels, planes, Some(yuv_color()))
        }
    }
}

#[test]
fn encoded_image_envelope_round_trips() {
    let image_bytes = png_image(2, 3);
    let envelope = Envelope::image(image_bytes.clone(), "png").expect("valid image envelope");

    assert!(envelope.is_image());
    assert_eq!(envelope.kind_str(), "Image");
    assert_eq!(envelope.payload_size(), image_bytes.len());
    assert_eq!(
        envelope.image_dimensions(),
        Some(ImageDimensions {
            width: 2,
            height: 3
        })
    );
    assert_eq!(
        envelope.as_image(),
        Some((image_bytes.as_slice(), ImageFormat::Png))
    );

    let restored = Envelope::from_bytes(&envelope.to_bytes().unwrap()).unwrap();
    assert_eq!(
        restored.as_image(),
        Some((image_bytes.as_slice(), ImageFormat::Png))
    );
}

#[test]
fn raw_image_envelopes_round_trip_all_supported_formats() {
    for format in [
        PixelFormat::Rgb8,
        PixelFormat::Rgba8,
        PixelFormat::Bgra8,
        PixelFormat::Nv12,
        PixelFormat::Nv21,
        PixelFormat::I420,
    ] {
        let (pixels, planes, color) = raw_case(format, 4, 4);
        let envelope = Envelope::image_raw(pixels.clone(), format, 4, 4, planes.clone(), color)
            .expect("valid raw image envelope");

        assert!(envelope.is_image());
        assert!(envelope.is_raw_image());
        assert_eq!(envelope.kind_str(), "Image");
        assert_eq!(envelope.payload_size(), pixels.len());
        assert_eq!(
            envelope.image_dimensions(),
            Some(ImageDimensions {
                width: 4,
                height: 4
            })
        );

        let raw = envelope.as_raw_image().expect("raw image view");
        assert_eq!(raw.pixels, pixels.as_slice());
        assert_eq!(raw.pixel_format, format);
        assert_eq!(
            raw.dimensions,
            ImageDimensions {
                width: 4,
                height: 4
            }
        );
        assert_eq!(raw.planes, planes.as_slice());
        assert_eq!(raw.color, color);

        let restored = Envelope::from_bytes(&envelope.to_bytes().unwrap()).unwrap();
        let restored_raw = restored.as_raw_image().expect("restored raw image view");
        assert_eq!(restored_raw.pixels, pixels.as_slice());
        assert_eq!(restored_raw.pixel_format, format);
        assert_eq!(restored_raw.planes, planes.as_slice());
        assert_eq!(restored_raw.color, color);
    }
}

#[test]
fn pixel_format_from_hint_accepts_supported_camera_formats() {
    let cases = [
        ("rgb8", PixelFormat::Rgb8),
        (" RGB8 ", PixelFormat::Rgb8),
        ("rgba8", PixelFormat::Rgba8),
        ("bgra8", PixelFormat::Bgra8),
        ("nv12", PixelFormat::Nv12),
        ("nv21", PixelFormat::Nv21),
        ("i420", PixelFormat::I420),
    ];

    for (hint, expected) in cases {
        assert_eq!(PixelFormat::from_hint(hint).unwrap(), expected);
    }
}

#[test]
fn unsupported_pixel_format_hints_are_typed_errors() {
    for hint in ["p010", "yuv420p10", "rgba8_premultiplied", "opaque_handle"] {
        let err = PixelFormat::from_hint(hint).unwrap_err();
        assert!(
            matches!(&err, EnvelopeError::UnsupportedPixelFormat { format } if format == hint),
            "expected UnsupportedPixelFormat for {hint}, got {err:?}"
        );
        assert!(err.to_string().contains("Unsupported raw pixel format"));
        assert!(!err.to_string().contains("expected png"));
    }
}

#[test]
fn jpeg_and_webp_image_envelopes_validate_dimensions() {
    let jpeg = Envelope::image(jpeg_image(4, 5), "jpg").expect("valid jpeg");
    let webp = Envelope::image(webp_image(6, 7), "webp").expect("valid webp");

    assert_eq!(
        jpeg.image_dimensions(),
        Some(ImageDimensions {
            width: 4,
            height: 5
        })
    );
    assert_eq!(
        webp.image_dimensions(),
        Some(ImageDimensions {
            width: 6,
            height: 7
        })
    );
}

#[test]
fn unsupported_image_format_is_rejected() {
    let err = Envelope::image(vec![1, 2, 3], "gif").unwrap_err();

    assert!(matches!(err, EnvelopeError::UnsupportedImageFormat { .. }));
}

#[test]
fn encoded_byte_limit_rejects_before_decode() {
    let limits = ImageValidationLimits::default().with_max_encoded_bytes(4);
    let err = Envelope::image_with_limits(vec![42; 16], "png", limits).unwrap_err();

    assert!(matches!(
        err,
        EnvelopeError::ImageEncodedTooLarge {
            byte_len: 16,
            max_bytes: 4
        }
    ));
}

#[test]
fn decoded_pixel_limit_rejects_small_compressed_images() {
    let limits = ImageValidationLimits::default().with_max_decoded_pixels(8);
    let err = Envelope::image_with_limits(png_image(4, 4), "png", limits).unwrap_err();

    assert!(matches!(
        err,
        EnvelopeError::ImageDimensionsTooLarge {
            width: 4,
            height: 4,
            pixels: 16,
            max_pixels: 8
        }
    ));
}

#[test]
fn raw_image_rejects_invalid_multiplane_layouts_before_reading_pixels() {
    for format in [PixelFormat::Nv12, PixelFormat::Nv21, PixelFormat::I420] {
        let (pixels, mut planes, color) = raw_case(format, 4, 4);
        planes[0].row_stride = 3;
        let err = Envelope::image_raw(pixels.clone(), format, 4, 4, planes, color).unwrap_err();
        assert!(
            matches!(
                err,
                EnvelopeError::RawImagePlaneInvalid { plane_index: 0, .. }
            ),
            "expected row-stride validation for {format:?}, got {err:?}"
        );

        let (pixels, mut planes, color) = raw_case(format, 4, 4);
        let last = planes.len() - 1;
        planes[last].offset = pixels.len();
        let err = Envelope::image_raw(pixels, format, 4, 4, planes, color).unwrap_err();
        assert!(
            matches!(err, EnvelopeError::RawImagePlaneInvalid { plane_index, .. } if plane_index == last),
            "expected extent validation for {format:?}, got {err:?}"
        );
    }
}

#[test]
fn raw_image_enforces_yuv_color_metadata_and_rgb_absence() {
    let (pixels, planes, _) = raw_case(PixelFormat::Nv12, 4, 4);
    let err = Envelope::image_raw(pixels, PixelFormat::Nv12, 4, 4, planes, None).unwrap_err();
    assert!(matches!(
        err,
        EnvelopeError::RawImageColorMetadataRequired {
            pixel_format: PixelFormat::Nv12
        }
    ));

    let (pixels, planes, _) = raw_case(PixelFormat::Rgba8, 4, 4);
    let err = Envelope::image_raw(pixels, PixelFormat::Rgba8, 4, 4, planes, Some(yuv_color()))
        .unwrap_err();
    assert!(matches!(
        err,
        EnvelopeError::RawImageColorMetadataUnsupported {
            pixel_format: PixelFormat::Rgba8
        }
    ));
}

#[test]
fn corrupt_image_bytes_are_rejected_with_stable_redacted_error() {
    let err = Envelope::image(vec![42, 42, 42, 42], "jpeg").unwrap_err();
    let message = err.to_string();

    assert!(matches!(
        err,
        EnvelopeError::ImageDecodeFailed {
            format: ImageFormat::Jpeg
        }
    ));
    assert!(message.contains("invalid or corrupt jpeg image bytes"));
    assert!(!message.contains("42"));
}

#[test]
fn animated_webp_is_rejected_before_decode() {
    let animated_webp_header = b"RIFF\x0c\x00\x00\x00WEBPANIM\x00\x00\x00\x00".to_vec();
    let err = Envelope::image(animated_webp_header, "webp").unwrap_err();

    assert!(matches!(
        err,
        EnvelopeError::AnimatedImageUnsupported {
            format: ImageFormat::WebP
        }
    ));
}

#[test]
fn user_message_preserves_ordered_text_and_images() {
    let first = Envelope::image(png_image(1, 1), "png").unwrap();
    let second = Envelope::image(jpeg_image(1, 1), "jpeg").unwrap();

    let message = Envelope::user_message("describe these", vec![first.clone(), second.clone()])
        .expect("multipart user message");

    assert_eq!(message.role(), Some(MessageRole::User));
    let parts = message.as_multipart().expect("multipart envelope");
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].as_text(), Some("describe these"));
    assert_eq!(parts[1].as_image(), first.as_image());
    assert_eq!(parts[2].as_image(), second.as_image());

    let restored = Envelope::from_bytes(&message.to_bytes().unwrap()).unwrap();
    let restored_parts = restored
        .as_multipart()
        .expect("restored multipart envelope");
    assert_eq!(restored.role(), Some(MessageRole::User));
    assert_eq!(restored_parts[0].as_text(), Some("describe these"));
    assert_eq!(restored_parts[1].as_image(), first.as_image());
    assert_eq!(restored_parts[2].as_image(), second.as_image());

    let mut context = ConversationContext::new();
    context.push(restored);
    let history = context.history();
    let replayed_parts = history[0]
        .as_multipart()
        .expect("history multipart envelope");
    assert_eq!(history[0].role(), Some(MessageRole::User));
    assert_eq!(replayed_parts[0].as_text(), Some("describe these"));
    assert_eq!(replayed_parts[1].as_image(), first.as_image());
    assert_eq!(replayed_parts[2].as_image(), second.as_image());
}

#[test]
fn user_message_rejects_non_image_attachments() {
    let text_attachment = Envelope::new(xybrid_core::ir::EnvelopeKind::Text("not image".into()));

    let err = Envelope::user_message("describe this", vec![text_attachment]).unwrap_err();

    assert!(err.to_string().contains("image attachments"));
}

#[test]
fn user_message_accepts_mixed_encoded_and_raw_image_parts() {
    let encoded = Envelope::image(png_image(1, 1), "png").unwrap();
    let (pixels, planes, color) = raw_case(PixelFormat::Rgba8, 2, 2);
    let raw = Envelope::image_raw(pixels, PixelFormat::Rgba8, 2, 2, planes, color).unwrap();

    let message =
        Envelope::user_message("describe these", vec![encoded.clone(), raw.clone()]).unwrap();
    let parts = message.as_multipart().expect("multipart envelope");

    assert_eq!(parts.len(), 3);
    assert_eq!(parts[1].as_image(), encoded.as_image());
    assert!(parts[2].as_raw_image().is_some());
}

#[test]
fn image_summaries_preserve_order_for_mixed_encoded_and_raw_parts() {
    let encoded_bytes = png_image(1, 1);
    let encoded_len = encoded_bytes.len();
    let encoded = Envelope::image(encoded_bytes, "png").unwrap();
    let (pixels, planes, color) = raw_case(PixelFormat::Nv12, 4, 2);
    let raw_len = pixels.len();
    let raw = Envelope::image_raw(pixels, PixelFormat::Nv12, 4, 2, planes, color).unwrap();
    let message = Envelope::user_message("describe these", vec![encoded, raw]).unwrap();

    let summaries = message.image_summaries();

    assert_eq!(summaries.len(), 2);
    assert_eq!(summaries[0].byte_len, encoded_len);
    assert_eq!(
        summaries[0].dimensions,
        ImageDimensions {
            width: 1,
            height: 1
        }
    );
    assert_eq!(summaries[0].source.as_encoded(), Some(ImageFormat::Png));
    assert_eq!(summaries[1].byte_len, raw_len);
    assert_eq!(
        summaries[1].dimensions,
        ImageDimensions {
            width: 4,
            height: 2
        }
    );
    assert_eq!(summaries[1].source.as_raw(), Some((PixelFormat::Nv12, 2)));
}

#[test]
fn image_summaries_walk_nested_multipart_without_pixel_bytes() {
    let raw_pixels = vec![99; 4 * 4 * 4];
    let raw = Envelope::image_raw(
        raw_pixels.clone(),
        PixelFormat::Rgba8,
        4,
        4,
        vec![ImagePlane {
            offset: 0,
            row_stride: 16,
            pixel_stride: 4,
            width: 4,
            height: 4,
        }],
        None,
    )
    .unwrap();
    let nested = Envelope::new(xybrid_core::ir::EnvelopeKind::MultiPart(vec![
        Envelope::new(xybrid_core::ir::EnvelopeKind::Text("ignore text".into())),
        raw,
    ]));

    let summaries = nested.image_summaries();

    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].byte_len, raw_pixels.len());
    assert_eq!(summaries[0].source.as_raw(), Some((PixelFormat::Rgba8, 1)));
    assert!(!format!("{summaries:?}").contains(&format!("{:?}", raw_pixels)));
}

#[test]
fn validate_image_tree_revalidates_nested_raw_image_sources() {
    let invalid_raw = Envelope::new(xybrid_core::ir::EnvelopeKind::Image {
        source: xybrid_core::ir::ImageSource::Raw {
            pixels: vec![1, 2, 3, 4].into(),
            pixel_format: PixelFormat::Nv12,
            dimensions: ImageDimensions {
                width: 4,
                height: 4,
            },
            planes: vec![
                ImagePlane {
                    offset: 0,
                    row_stride: 4,
                    pixel_stride: 1,
                    width: 4,
                    height: 4,
                },
                ImagePlane {
                    offset: 4,
                    row_stride: 4,
                    pixel_stride: 2,
                    width: 2,
                    height: 2,
                },
            ],
            color: Some(yuv_color()),
        },
    });
    let nested = Envelope::new(xybrid_core::ir::EnvelopeKind::MultiPart(vec![
        Envelope::new(xybrid_core::ir::EnvelopeKind::Text("ignore text".into())),
        invalid_raw,
    ]));

    let err = nested.validate_image_tree().unwrap_err();

    assert!(
        matches!(err, EnvelopeError::RawImagePlaneInvalid { .. }),
        "expected raw plane validation error, got {err:?}"
    );
    assert!(!err.to_string().contains("1, 2, 3, 4"));
}

#[test]
fn validate_image_tree_revalidates_nested_encoded_image_sources() {
    let invalid_encoded = Envelope::new(xybrid_core::ir::EnvelopeKind::Image {
        source: xybrid_core::ir::ImageSource::Encoded {
            bytes: vec![42, 42, 42, 42].into(),
            format: ImageFormat::Png,
            dimensions: ImageDimensions {
                width: 1,
                height: 1,
            },
        },
    });
    let nested = Envelope::new(xybrid_core::ir::EnvelopeKind::MultiPart(vec![
        invalid_encoded,
    ]));

    let err = nested.validate_image_tree().unwrap_err();

    assert!(matches!(
        err,
        EnvelopeError::ImageDecodeFailed {
            format: ImageFormat::Png
        }
    ));
    assert!(!err.to_string().contains("42"));
}

#[test]
fn debug_output_redacts_image_bytes() {
    let image_bytes = png_image(1, 1);
    let debug_sentinel = format!("{:?}", image_bytes);
    let envelope = Envelope::image(image_bytes, "png").unwrap();
    let debug = format!("{:?}", envelope);

    assert!(debug.len() < 220);
    assert!(debug.contains("Image"));
    assert!(debug.contains("png"));
    assert!(debug.contains("1x1"));
    assert!(!debug.contains(&debug_sentinel));
}

#[test]
fn raw_image_debug_and_json_redact_pixel_bytes() {
    let width = 3840;
    let height = 2160;
    let (pixels, planes, color) = raw_case(PixelFormat::Rgba8, width, height);
    let debug_sentinel = format!("{:?}", &pixels[..16]);
    let envelope =
        Envelope::image_raw(pixels, PixelFormat::Rgba8, width, height, planes, color).unwrap();

    let debug = format!("{:?}", envelope);
    assert!(debug.len() < 220);
    assert!(debug.contains("Image"));
    assert!(debug.contains("rgba8"));
    assert!(debug.contains("3840x2160"));
    assert!(!debug.contains(&debug_sentinel));

    let json = envelope.to_json().expect("raw image JSON");
    assert!(json.contains("\"byte_len\""));
    assert!(json.contains("\"pixel_format\""));
    assert!(json.contains("\"width\": 3840"));
    assert!(json.contains("\"height\": 2160"));
    assert!(!json.contains("\"pixels\""));
    assert!(!json.contains(&debug_sentinel));
}

#[test]
fn human_readable_serialization_redacts_image_bytes() {
    let image_bytes = webp_image(1, 1);
    let encoded_len = image_bytes.len();
    let envelope = Envelope::image(image_bytes, "webp").unwrap();
    let json = serde_json::to_string(&envelope).expect("image envelope json");

    assert!(json.contains("\"byte_len\""));
    assert!(json.contains(&encoded_len.to_string()));
    assert!(json.contains("\"width\":1"));
    assert!(json.contains("\"height\":1"));
    assert!(!json.contains("\"bytes\""));
}

#[test]
fn to_json_redacts_nested_multipart_images() {
    let image = Envelope::image(png_image(1, 1), "png").unwrap();
    let message = Envelope::user_message("describe this", vec![image]).unwrap();
    let json = message.to_json().expect("multipart json");

    assert!(json.contains("\"byte_len\""));
    assert!(json.contains("\"width\""));
    assert!(!json.contains("\"bytes\""));
}
