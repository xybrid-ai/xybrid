// Xybrid SDK - Envelope vision EditMode tests.
// Requires the native xybrid library to be loadable (DllImport "xybrid").

using System;
using NUnit.Framework;
using UnityEngine;
using Xybrid;

namespace Xybrid.Tests.Editor
{
    [TestFixture]
    public class EnvelopeVisionTests
    {
        [Test]
        public void Image_NormalizesJpegAlias()
        {
            byte[] jpegBytes = TinyJpegBytes();

            using (var image = Envelope.Image(jpegBytes, "JPG"))
            {
                Assert.IsFalse(image.IsDisposed);
            }
        }

        [Test]
        public void Image_RejectsUnsupportedFormat()
        {
            Assert.Throws<ArgumentException>(() => Envelope.Image(new byte[] { 1, 2, 3 }, "gif"));
        }

        [Test]
        public void UserMessage_AcceptsImageAttachments()
        {
            using (var image = Envelope.Image(TinyPngBytes(), "png"))
            using (var message = Envelope.UserMessage("Describe this image", new[] { image }))
            {
                Assert.IsFalse(message.IsDisposed);
            }
        }

        [Test]
        public void UserMessage_RejectsNonImageAttachments()
        {
            using (var text = Envelope.Text("not an image"))
            {
                Assert.Throws<ArgumentException>(() =>
                    Envelope.UserMessage("Describe this image", new[] { text }));
            }
        }

        private static byte[] TinyPngBytes()
        {
            var texture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            texture.SetPixel(0, 0, Color.black);
            texture.Apply();
            return texture.EncodeToPNG();
        }

        private static byte[] TinyJpegBytes()
        {
            var texture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            texture.SetPixel(0, 0, Color.black);
            texture.Apply();
            return texture.EncodeToJPG();
        }
    }
}
