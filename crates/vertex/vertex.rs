use std::mem;

use anyhow::{Result, anyhow, bail};
use futures::{AsyncBufReadExt, AsyncReadExt, StreamExt, io::BufReader, stream::BoxStream};
use http_client::{AsyncBody, HttpClient, Method, Request as HttpRequest};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

pub const VERTEX_API_VERSION: &str = "v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum Publisher {
    Google,
    Anthropic,
}

impl Publisher {
    pub fn detect_from_model(model_id: &str) -> Option<Self> {
        if model_id.starts_with("gemini-") {
            Some(Publisher::Google)
        } else if model_id.starts_with("claude-") {
            Some(Publisher::Anthropic)
        } else {
            None
        }
    }

    pub fn api_name(&self) -> &'static str {
        match self {
            Publisher::Google => "google",
            Publisher::Anthropic => "anthropic",
        }
    }
}

/// Returns true if this model requires the global region endpoint.
/// Gemini 3 models require the global region.
pub fn requires_global_region(model_id: &str) -> bool {
    model_id.starts_with("gemini-3")
}

/// Returns true if this is a Gemini 3 model.
/// Used to determine global region requirement and thinking config type.
pub fn is_gemini_3(model_id: &str) -> bool {
    model_id.starts_with("gemini-3")
}

/// Build the Vertex AI API URL for a given model.
/// Handles the special case where Gemini 3 models require the global region.
pub fn build_vertex_api_url(
    location: &str,
    project: &str,
    publisher: Publisher,
    model_id: &str,
) -> String {
    let method = match publisher {
        Publisher::Google => "streamGenerateContent",
        Publisher::Anthropic => "streamRawPredict",
    };
    let query_params = match publisher {
        Publisher::Google => "?alt=sse",
        Publisher::Anthropic => "",
    };

    // Gemini 3 models require the global region endpoint
    if requires_global_region(model_id) {
        format!(
            "https://aiplatform.googleapis.com/{VERTEX_API_VERSION}/projects/{project}/locations/global/publishers/{publisher}/models/{model_id}:{method}{query_params}",
            publisher = publisher.api_name(),
        )
    } else {
        format!(
            "https://{location}-aiplatform.googleapis.com/{VERTEX_API_VERSION}/projects/{project}/locations/{location}/publishers/{publisher}/models/{model_id}:{method}{query_params}",
            publisher = publisher.api_name(),
        )
    }
}

pub mod gemini {
    use super::*;

    pub async fn stream_generate_content(
        client: &dyn HttpClient,
        location: &str,
        project: &str,
        access_token: &str,
        mut request: GenerateContentRequest,
    ) -> Result<BoxStream<'static, Result<GenerateContentResponse>>> {
        validate_generate_content_request(&request)?;

        let model_id = mem::take(&mut request.model.model_id);
        let uri = build_vertex_api_url(location, project, Publisher::Google, &model_id);

        let request_builder = HttpRequest::builder()
            .method(Method::POST)
            .uri(uri)
            .header("Authorization", format!("Bearer {}", access_token.trim()))
            .header("Content-Type", "application/json");

        let request = request_builder.body(AsyncBody::from(serde_json::to_string(&request)?))?;
        let mut response = client.send(request).await?;

        if response.status().is_success() {
            let reader = BufReader::new(response.into_body());
            Ok(reader
                .lines()
                .filter_map(|line| async move {
                    match line {
                        Ok(line) => {
                            if let Some(line) = line.strip_prefix("data: ") {
                                match serde_json::from_str(line) {
                                    Ok(response) => Some(Ok(response)),
                                    Err(error) => Some(Err(anyhow!(
                                        "Error parsing JSON: {error:?}\n{line:?}"
                                    ))),
                                }
                            } else {
                                None
                            }
                        }
                        Err(error) => Some(Err(anyhow!(error))),
                    }
                })
                .boxed())
        } else {
            let mut text = String::new();
            response.body_mut().read_to_string(&mut text).await?;
            Err(anyhow!(
                "error during streamGenerateContent, status code: {:?}, body: {}",
                response.status(),
                text
            ))
        }
    }

    pub fn validate_generate_content_request(request: &GenerateContentRequest) -> Result<()> {
        if request.model.is_empty() {
            bail!("Model must be specified");
        }

        if request.contents.is_empty() {
            bail!("Request must contain at least one content item");
        }

        if let Some(user_content) = request
            .contents
            .iter()
            .find(|content| content.role == Role::User)
            && user_content.parts.is_empty()
        {
            bail!("User content must contain at least one part");
        }

        Ok(())
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GenerateContentRequest {
        #[serde(default, skip_serializing_if = "ModelName::is_empty")]
        pub model: ModelName,
        pub contents: Vec<Content>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub system_instruction: Option<SystemInstruction>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub generation_config: Option<GenerationConfig>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub safety_settings: Option<Vec<SafetySetting>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tools: Option<Vec<Tool>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_config: Option<ToolConfig>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GenerateContentResponse {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub candidates: Option<Vec<GenerateContentCandidate>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub prompt_feedback: Option<PromptFeedback>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub usage_metadata: Option<UsageMetadata>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GenerateContentCandidate {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub index: Option<usize>,
        pub content: Content,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub finish_reason: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub finish_message: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub safety_ratings: Option<Vec<SafetyRating>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub citation_metadata: Option<CitationMetadata>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Content {
        #[serde(default)]
        pub parts: Vec<Part>,
        pub role: Role,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct SystemInstruction {
        pub parts: Vec<Part>,
    }

    #[derive(Debug, PartialEq, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub enum Role {
        User,
        Model,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum Part {
        TextPart(TextPart),
        InlineDataPart(InlineDataPart),
        FunctionCallPart(FunctionCallPart),
        FunctionResponsePart(FunctionResponsePart),
        ThoughtPart(ThoughtPart),
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct TextPart {
        pub text: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct InlineDataPart {
        pub inline_data: GenerativeContentBlob,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GenerativeContentBlob {
        pub mime_type: String,
        pub data: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct FunctionCallPart {
        pub function_call: FunctionCall,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thought_signature: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct FunctionResponsePart {
        pub function_response: FunctionResponse,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ThoughtPart {
        pub thought: bool,
        pub thought_signature: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct CitationSource {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub start_index: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub end_index: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub license: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct CitationMetadata {
        pub citation_sources: Vec<CitationSource>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct PromptFeedback {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub block_reason: Option<String>,
        pub safety_ratings: Option<Vec<SafetyRating>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub block_reason_message: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize, Default)]
    #[serde(rename_all = "camelCase")]
    pub struct UsageMetadata {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub prompt_token_count: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cached_content_token_count: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub candidates_token_count: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_use_prompt_token_count: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thoughts_token_count: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub total_token_count: Option<u64>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ThinkingConfig {
        /// For Gemini 2.5 models: token budget for thinking (0-24576)
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thinking_budget: Option<u32>,
        /// For Gemini 2.5 models: whether to include thought summaries
        #[serde(skip_serializing_if = "Option::is_none")]
        pub include_thoughts: Option<bool>,
        /// For Gemini 3 models: thinking depth level (MINIMAL, LOW, MEDIUM, HIGH)
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thinking_level: Option<ThinkingLevel>,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum ThinkingLevel {
        Minimal,
        Low,
        Medium,
        High,
    }

    impl ThinkingConfig {
        /// Create thinking config for Gemini 2.5 models
        pub fn for_gemini_2_5(include_thoughts: bool) -> Self {
            Self {
                thinking_budget: None,
                include_thoughts: Some(include_thoughts),
                thinking_level: None,
            }
        }

        /// Create thinking config for Gemini 2.5 models with budget
        pub fn for_gemini_2_5_with_budget(budget: u32) -> Self {
            Self {
                thinking_budget: Some(budget),
                include_thoughts: Some(true),
                thinking_level: None,
            }
        }

        /// Create thinking config for Gemini 3 models
        pub fn for_gemini_3(level: ThinkingLevel) -> Self {
            Self {
                thinking_budget: None,
                include_thoughts: None,
                thinking_level: Some(level),
            }
        }
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GenerationConfig {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub candidate_count: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stop_sequences: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_output_tokens: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_k: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thinking_config: Option<ThinkingConfig>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct SafetySetting {
        pub category: HarmCategory,
        pub threshold: HarmBlockThreshold,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub enum HarmCategory {
        #[serde(rename = "HARM_CATEGORY_UNSPECIFIED")]
        Unspecified,
        #[serde(rename = "HARM_CATEGORY_DEROGATORY")]
        Derogatory,
        #[serde(rename = "HARM_CATEGORY_TOXICITY")]
        Toxicity,
        #[serde(rename = "HARM_CATEGORY_VIOLENCE")]
        Violence,
        #[serde(rename = "HARM_CATEGORY_SEXUAL")]
        Sexual,
        #[serde(rename = "HARM_CATEGORY_MEDICAL")]
        Medical,
        #[serde(rename = "HARM_CATEGORY_DANGEROUS")]
        Dangerous,
        #[serde(rename = "HARM_CATEGORY_HARASSMENT")]
        Harassment,
        #[serde(rename = "HARM_CATEGORY_HATE_SPEECH")]
        HateSpeech,
        #[serde(rename = "HARM_CATEGORY_SEXUALLY_EXPLICIT")]
        SexuallyExplicit,
        #[serde(rename = "HARM_CATEGORY_DANGEROUS_CONTENT")]
        DangerousContent,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum HarmBlockThreshold {
        #[serde(rename = "HARM_BLOCK_THRESHOLD_UNSPECIFIED")]
        Unspecified,
        BlockLowAndAbove,
        BlockMediumAndAbove,
        BlockOnlyHigh,
        BlockNone,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum HarmProbability {
        #[serde(rename = "HARM_PROBABILITY_UNSPECIFIED")]
        Unspecified,
        Negligible,
        Low,
        Medium,
        High,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct SafetyRating {
        pub category: HarmCategory,
        pub probability: HarmProbability,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FunctionCall {
        pub name: String,
        pub args: serde_json::Value,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FunctionResponse {
        pub name: String,
        pub response: serde_json::Value,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Tool {
        pub function_declarations: Vec<FunctionDeclaration>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ToolConfig {
        pub function_calling_config: FunctionCallingConfig,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct FunctionCallingConfig {
        pub mode: FunctionCallingMode,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub allowed_function_names: Option<Vec<String>>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum FunctionCallingMode {
        Auto,
        Any,
        None,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FunctionDeclaration {
        pub name: String,
        pub description: String,
        pub parameters: serde_json::Value,
    }

    #[derive(Debug, Default)]
    pub struct ModelName {
        pub model_id: String,
    }

    impl ModelName {
        pub fn is_empty(&self) -> bool {
            self.model_id.is_empty()
        }
    }

    impl Serialize for ModelName {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serializer.serialize_str(&self.model_id)
        }
    }

    impl<'de> Deserialize<'de> for ModelName {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let string = String::deserialize(deserializer)?;
            Ok(Self { model_id: string })
        }
    }
}

pub mod anthropic {
    use super::*;

    pub const ANTHROPIC_VERTEX_VERSION: &str = "vertex-2023-10-16";

    pub async fn stream_completion(
        client: &dyn HttpClient,
        location: &str,
        project: &str,
        access_token: &str,
        request: Request,
    ) -> Result<BoxStream<'static, Result<Event>>> {
        let uri = build_vertex_api_url(location, project, Publisher::Anthropic, &request.model);

        let streaming_request = StreamingRequest {
            base: request,
            stream: true,
        };

        let request_builder = HttpRequest::builder()
            .method(Method::POST)
            .uri(uri)
            .header("Authorization", format!("Bearer {}", access_token.trim()))
            .header("Content-Type", "application/json");

        let http_request =
            request_builder.body(AsyncBody::from(serde_json::to_string(&streaming_request)?))?;
        let mut response = client.send(http_request).await?;

        if response.status().is_success() {
            let reader = BufReader::new(response.into_body());
            let stream = reader
                .lines()
                .filter_map(|line| async move {
                    match line {
                        Ok(line) => {
                            let line = line.strip_prefix("data: ")?;
                            match serde_json::from_str(line) {
                                Ok(response) => Some(Ok(response)),
                                Err(error) => Some(Err(anyhow!(
                                    "Error parsing JSON: {error:?}\n{line:?}"
                                ))),
                            }
                        }
                        Err(error) => Some(Err(anyhow!(error))),
                    }
                })
                .boxed();
            Ok(stream)
        } else {
            let mut text = String::new();
            response.body_mut().read_to_string(&mut text).await?;

            match serde_json::from_str::<Event>(&text) {
                Ok(Event::Error { error }) => {
                    Err(anyhow!("Anthropic API error: {}: {}", error.error_type, error.message))
                }
                _ => Err(anyhow!(
                    "error during streamRawPredict, status code: {:?}, body: {}",
                    response.status(),
                    text
                )),
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize, Copy, Clone)]
    #[serde(rename_all = "lowercase")]
    pub enum CacheControlType {
        Ephemeral,
    }

    #[derive(Debug, Serialize, Deserialize, Copy, Clone)]
    pub struct CacheControl {
        #[serde(rename = "type")]
        pub cache_type: CacheControlType,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Message {
        pub role: Role,
        pub content: Vec<RequestContent>,
    }

    #[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
    #[serde(rename_all = "lowercase")]
    pub enum Role {
        User,
        Assistant,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum RequestContent {
        #[serde(rename = "text")]
        Text {
            text: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            cache_control: Option<CacheControl>,
        },
        #[serde(rename = "thinking")]
        Thinking {
            thinking: String,
            signature: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            cache_control: Option<CacheControl>,
        },
        #[serde(rename = "redacted_thinking")]
        RedactedThinking { data: String },
        #[serde(rename = "image")]
        Image {
            source: ImageSource,
            #[serde(skip_serializing_if = "Option::is_none")]
            cache_control: Option<CacheControl>,
        },
        #[serde(rename = "tool_use")]
        ToolUse {
            id: String,
            name: String,
            input: serde_json::Value,
            #[serde(skip_serializing_if = "Option::is_none")]
            cache_control: Option<CacheControl>,
        },
        #[serde(rename = "tool_result")]
        ToolResult {
            tool_use_id: String,
            is_error: bool,
            content: ToolResultContent,
            #[serde(skip_serializing_if = "Option::is_none")]
            cache_control: Option<CacheControl>,
        },
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum ToolResultContent {
        Plain(String),
        Multipart(Vec<ToolResultPart>),
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(tag = "type", rename_all = "lowercase")]
    pub enum ToolResultPart {
        Text { text: String },
        Image { source: ImageSource },
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum ResponseContent {
        #[serde(rename = "text")]
        Text { text: String },
        #[serde(rename = "thinking")]
        Thinking { thinking: String },
        #[serde(rename = "redacted_thinking")]
        RedactedThinking { data: String },
        #[serde(rename = "tool_use")]
        ToolUse {
            id: String,
            name: String,
            input: serde_json::Value,
        },
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ImageSource {
        #[serde(rename = "type")]
        pub source_type: String,
        pub media_type: String,
        pub data: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Tool {
        pub name: String,
        pub description: String,
        pub input_schema: serde_json::Value,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(tag = "type", rename_all = "lowercase")]
    pub enum ToolChoice {
        Auto,
        Any,
        Tool { name: String },
        None,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(tag = "type", rename_all = "lowercase")]
    pub enum Thinking {
        Enabled { budget_tokens: Option<u32> },
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum StringOrContents {
        String(String),
        Content(Vec<RequestContent>),
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Request {
        #[serde(rename = "anthropic_version")]
        pub anthropic_version: String,
        /// Model is passed in the URL for Vertex AI, not in the request body
        #[serde(skip_serializing)]
        pub model: String,
        pub max_tokens: u64,
        pub messages: Vec<Message>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub tools: Vec<Tool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub thinking: Option<Thinking>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub tool_choice: Option<ToolChoice>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub system: Option<StringOrContents>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub stop_sequences: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub top_k: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f32>,
    }

    impl Request {
        pub fn new(model: String, max_tokens: u64, messages: Vec<Message>) -> Self {
            Self {
                anthropic_version: ANTHROPIC_VERTEX_VERSION.to_string(),
                model,
                max_tokens,
                messages,
                tools: Vec::new(),
                thinking: None,
                tool_choice: None,
                system: None,
                stop_sequences: Vec::new(),
                temperature: None,
                top_k: None,
                top_p: None,
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct StreamingRequest {
        #[serde(flatten)]
        pub base: Request,
        pub stream: bool,
    }

    #[derive(Debug, Serialize, Deserialize, Default)]
    pub struct Usage {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub input_tokens: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub output_tokens: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub cache_creation_input_tokens: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub cache_read_input_tokens: Option<u64>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Response {
        pub id: String,
        #[serde(rename = "type")]
        pub response_type: String,
        pub role: Role,
        pub content: Vec<ResponseContent>,
        pub model: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub stop_reason: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub stop_sequence: Option<String>,
        pub usage: Usage,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum Event {
        #[serde(rename = "message_start")]
        MessageStart { message: Response },
        #[serde(rename = "content_block_start")]
        ContentBlockStart {
            index: usize,
            content_block: ResponseContent,
        },
        #[serde(rename = "content_block_delta")]
        ContentBlockDelta { index: usize, delta: ContentDelta },
        #[serde(rename = "content_block_stop")]
        ContentBlockStop { index: usize },
        #[serde(rename = "message_delta")]
        MessageDelta { delta: MessageDelta, usage: Usage },
        #[serde(rename = "message_stop")]
        MessageStop,
        #[serde(rename = "ping")]
        Ping,
        #[serde(rename = "error")]
        Error { error: ApiError },
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum ContentDelta {
        #[serde(rename = "text_delta")]
        TextDelta { text: String },
        #[serde(rename = "thinking_delta")]
        ThinkingDelta { thinking: String },
        #[serde(rename = "signature_delta")]
        SignatureDelta { signature: String },
        #[serde(rename = "input_json_delta")]
        InputJsonDelta { partial_json: String },
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct MessageDelta {
        pub stop_reason: Option<String>,
        pub stop_sequence: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ApiError {
        #[serde(rename = "type")]
        pub error_type: String,
        pub message: String,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_publisher_detection() {
        assert_eq!(
            Publisher::detect_from_model("gemini-2.5-flash"),
            Some(Publisher::Google)
        );
        assert_eq!(
            Publisher::detect_from_model("gemini-2.5-pro"),
            Some(Publisher::Google)
        );
        assert_eq!(
            Publisher::detect_from_model("gemini-3-pro-preview"),
            Some(Publisher::Google)
        );
        assert_eq!(
            Publisher::detect_from_model("claude-sonnet-4-5@20250929"),
            Some(Publisher::Anthropic)
        );
        assert_eq!(
            Publisher::detect_from_model("claude-opus-4-5@20251101"),
            Some(Publisher::Anthropic)
        );
        assert_eq!(Publisher::detect_from_model("unknown-model"), None);
    }

    #[test]
    fn test_gemini_3_detection() {
        assert!(is_gemini_3("gemini-3-pro-preview"));
        assert!(is_gemini_3("gemini-3-flash"));
        assert!(!is_gemini_3("gemini-2.5-flash"));
        assert!(!is_gemini_3("gemini-2.5-pro"));
    }

    #[test]
    fn test_global_region_requirement() {
        assert!(requires_global_region("gemini-3-pro-preview"));
        assert!(requires_global_region("gemini-3-flash"));
        assert!(!requires_global_region("gemini-2.5-flash"));
        assert!(!requires_global_region("gemini-2.5-pro"));
        assert!(!requires_global_region("claude-sonnet-4-5@20250929"));
    }

    #[test]
    fn test_vertex_api_url_gemini_2_5() {
        let url = build_vertex_api_url("us-east5", "my-project", Publisher::Google, "gemini-2.5-flash");
        assert_eq!(
            url,
            "https://us-east5-aiplatform.googleapis.com/v1/projects/my-project/locations/us-east5/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn test_vertex_api_url_gemini_3_uses_global() {
        let url = build_vertex_api_url("us-east5", "my-project", Publisher::Google, "gemini-3-pro-preview");
        assert_eq!(
            url,
            "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/publishers/google/models/gemini-3-pro-preview:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn test_vertex_api_url_anthropic() {
        let url = build_vertex_api_url(
            "us-east5",
            "my-project",
            Publisher::Anthropic,
            "claude-sonnet-4-5@20250929",
        );
        assert_eq!(
            url,
            "https://us-east5-aiplatform.googleapis.com/v1/projects/my-project/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929:streamRawPredict"
        );
    }

    #[test]
    fn test_anthropic_request_has_version() {
        let request = anthropic::Request::new(
            "claude-sonnet-4-5@20250929".to_string(),
            4096,
            vec![],
        );
        assert_eq!(request.anthropic_version, "vertex-2023-10-16");
    }

    #[test]
    fn test_anthropic_request_model_not_serialized() {
        let request = anthropic::Request::new(
            "claude-sonnet-4-5@20250929".to_string(),
            4096,
            vec![],
        );
        let json = serde_json::to_string(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        
        // The model field should NOT be in the serialized output
        // because for Vertex AI, the model is specified in the URL, not the body
        assert!(parsed.get("model").is_none(), "model field should not be serialized, but got: {}", json);
    }

    #[test]
    fn test_anthropic_streaming_request_model_not_serialized() {
        use anthropic::StreamingRequest;
        
        let request = anthropic::Request::new(
            "claude-sonnet-4-5@20250929".to_string(),
            4096,
            vec![],
        );
        let streaming_request = StreamingRequest {
            base: request,
            stream: true,
        };
        let json = serde_json::to_string(&streaming_request).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        
        // The model field should NOT be in the serialized output even when flattened
        assert!(parsed.get("model").is_none(), "model field should not be serialized in streaming request, but got: {}", json);
        // But stream should be present
        assert!(parsed.get("stream").is_some(), "stream field should be present");
        assert_eq!(parsed.get("stream").unwrap(), true);
    }

    #[test]
    fn test_thinking_config_gemini_2_5() {
        let config = gemini::ThinkingConfig::for_gemini_2_5(true);
        assert_eq!(config.include_thoughts, Some(true));
        assert!(config.thinking_level.is_none());
        assert!(config.thinking_budget.is_none());
    }

    #[test]
    fn test_thinking_config_gemini_3() {
        let config = gemini::ThinkingConfig::for_gemini_3(gemini::ThinkingLevel::High);
        assert!(config.thinking_level.is_some());
        assert!(config.include_thoughts.is_none());
        assert!(config.thinking_budget.is_none());
    }
}
