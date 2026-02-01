final class AppConstants {
  static const registryUrl = 'http://localhost:8080';
  static const pipelinesDir = 'assets/pipelines';

  // Local backend URL for development
  // This is used for both telemetry and LLM gateway calls
  // Note: The SDK automatically appends /v1 for gateway endpoints
  static const backendUrl = 'http://127.0.0.1:8000';

  // Gateway URL with /v1 path for OpenAI-compatible API
  // Used when explicitly configuring LlmClientConfig (which requires full URL)
  static const gatewayUrl = '$backendUrl/v1';
}
