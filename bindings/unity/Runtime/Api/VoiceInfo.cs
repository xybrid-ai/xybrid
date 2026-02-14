// Xybrid SDK - VoiceInfo
// Metadata for a single voice available in a TTS model.

namespace Xybrid
{
    /// <summary>
    /// Metadata for a single voice available in a TTS model's voice catalog.
    /// </summary>
    /// <remarks>
    /// Use <see cref="Model.Voices"/> to get all available voices,
    /// or <see cref="Model.GetVoice"/> to look up a specific voice by ID.
    /// Pass the <see cref="Id"/> to <see cref="Envelope.Text(string, string, double)"/>
    /// to select a voice for TTS inference.
    /// </remarks>
    public sealed class VoiceInfo
    {
        /// <summary>
        /// Unique voice identifier (e.g., "af_bella").
        /// Pass this to <see cref="Envelope.Text(string, string, double)"/> for voice selection.
        /// </summary>
        public string Id { get; }

        /// <summary>
        /// Human-readable display name (e.g., "Bella").
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Gender: "male", "female", or "neutral". May be null.
        /// </summary>
        public string Gender { get; }

        /// <summary>
        /// BCP-47 language tag (e.g., "en-US", "en-GB"). May be null.
        /// </summary>
        public string Language { get; }

        /// <summary>
        /// Style descriptor (e.g., "neutral", "cheerful"). May be null.
        /// </summary>
        public string Style { get; }

        /// <summary>
        /// Returns true if the voice gender is male.
        /// </summary>
        public bool IsMale => Gender == "male";

        /// <summary>
        /// Returns true if the voice gender is female.
        /// </summary>
        public bool IsFemale => Gender == "female";

        internal VoiceInfo(string id, string name, string gender, string language, string style)
        {
            Id = id;
            Name = name;
            Gender = gender;
            Language = language;
            Style = style;
        }

        /// <summary>
        /// Returns a string representation of the voice.
        /// </summary>
        public override string ToString()
        {
            return $"VoiceInfo({Id}: {Name})";
        }
    }
}
