use xybrid_core::{
    conversation::ConversationContext,
    ir::{Envelope, EnvelopeKind, ImageDimensions, ImageFormat, MessageRole},
    runtime_adapter::{MultimodalChatMessage, MultimodalMessagePart},
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

#[test]
fn context_to_multimodal_messages_preserves_ordered_parts_and_roles() {
    let first_image = Envelope::image(png_image(2, 3), "png").unwrap();
    let second_image = Envelope::image(jpeg_image(4, 5), "jpeg").unwrap();
    let mixed_turn = Envelope::new(EnvelopeKind::MultiPart(vec![
        Envelope::new(EnvelopeKind::Text("first ".to_string())),
        first_image.clone(),
        Envelope::new(EnvelopeKind::Text(" then ".to_string())),
        second_image.clone(),
    ]))
    .with_role(MessageRole::User);

    let mut context = ConversationContext::new().with_system(
        Envelope::new(EnvelopeKind::Text("be concise".to_string())).with_role(MessageRole::System),
    );
    context
        .push(Envelope::new(EnvelopeKind::Text("hello".to_string())).with_role(MessageRole::User));
    context.push(mixed_turn);
    context.push(
        Envelope::new(EnvelopeKind::Text("looks good".to_string()))
            .with_role(MessageRole::Assistant),
    );

    let messages = MultimodalChatMessage::from_context(&context).unwrap();

    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0].role, MessageRole::System);
    assert_eq!(messages[0].parts[0].as_text(), Some("be concise"));
    assert_eq!(messages[1].role, MessageRole::User);
    assert_eq!(messages[1].parts[0].as_text(), Some("hello"));
    assert_eq!(messages[2].role, MessageRole::User);
    assert_eq!(messages[2].parts.len(), 4);
    assert_eq!(messages[2].parts[0].as_text(), Some("first "));
    assert_eq!(messages[2].parts[2].as_text(), Some(" then "));

    match &messages[2].parts[1] {
        MultimodalMessagePart::Image(image) => {
            assert_eq!(image.format(), Some(ImageFormat::Png));
            assert_eq!(
                image.dimensions(),
                Some(ImageDimensions {
                    width: 2,
                    height: 3
                })
            );
            assert_eq!(image.byte_len(), first_image.payload_size());
        }
        other => panic!("expected first image part, got {other:?}"),
    }

    match &messages[2].parts[3] {
        MultimodalMessagePart::Image(image) => {
            assert_eq!(image.format(), Some(ImageFormat::Jpeg));
            assert_eq!(
                image.dimensions(),
                Some(ImageDimensions {
                    width: 4,
                    height: 5
                })
            );
            assert_eq!(image.byte_len(), second_image.payload_size());
        }
        other => panic!("expected second image part, got {other:?}"),
    }
}

#[test]
fn marker_prompt_preserves_part_order_and_rejects_marker_collisions() {
    let image = Envelope::image(png_image(1, 1), "png").unwrap();
    let turn = Envelope::new(EnvelopeKind::MultiPart(vec![
        Envelope::new(EnvelopeKind::Text("look ".to_string())),
        image,
        Envelope::new(EnvelopeKind::Text(" now".to_string())),
    ]))
    .with_role(MessageRole::User);

    let message = MultimodalChatMessage::from_envelope(&turn).unwrap();

    assert_eq!(message.image_count(), 1);
    assert_eq!(
        message.marker_prompt("<__media__>").unwrap(),
        "look <__media__> now"
    );

    let collision = Envelope::new(EnvelopeKind::Text("literal <__media__>".to_string()))
        .with_role(MessageRole::User);
    let err = MultimodalChatMessage::from_envelope(&collision)
        .unwrap()
        .marker_prompt("<__media__>")
        .unwrap_err();

    assert!(err.to_string().contains("media marker"));
}

#[test]
fn multimodal_contract_rejects_non_text_or_image_parts_without_leaking_bytes() {
    let sentinel = vec![1_u8, 2, 3, 4, 5];
    let debug_sentinel = format!("{sentinel:?}");
    let turn = Envelope::new(EnvelopeKind::MultiPart(vec![Envelope::new(
        EnvelopeKind::Audio(sentinel),
    )]))
    .with_role(MessageRole::User);

    let err = MultimodalChatMessage::from_envelope(&turn).unwrap_err();
    let message = err.to_string();

    assert!(message.contains("unsupported multimodal part"));
    assert!(!message.contains(&debug_sentinel));
}

#[test]
fn multimodal_debug_redacts_image_bytes() {
    let image_bytes = png_image(8, 8);
    let debug_sentinel = format!("{image_bytes:?}");
    let image = Envelope::image(image_bytes, "png").unwrap();
    let message =
        MultimodalChatMessage::from_envelope(&image.with_role(MessageRole::User)).unwrap();
    let debug = format!("{message:?}");

    assert!(debug.len() < 300);
    assert!(debug.contains("Image"));
    assert!(debug.contains("png"));
    assert!(debug.contains("8x8"));
    assert!(!debug.contains(&debug_sentinel));
}
