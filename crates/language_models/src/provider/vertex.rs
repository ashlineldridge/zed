use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow};
use collections::{BTreeMap, HashMap};
use futures::{FutureExt, Stream, StreamExt, future::BoxFuture};
use gpui::{AnyView, App, AsyncApp, Context, Entity, Subscription, Task, Window};
use http_client::HttpClient;
use language_model::{
    AuthenticateError, LanguageModel, LanguageModelCacheConfiguration,
    LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelId, LanguageModelName,
    LanguageModelProvider, LanguageModelProviderId, LanguageModelProviderName,
    LanguageModelProviderState, LanguageModelRequest, LanguageModelToolChoice,
    LanguageModelToolResultContent, LanguageModelToolSchemaFormat, LanguageModelToolUse,
    LanguageModelToolUseId, MessageContent, RateLimiter, Role, StopReason, TokenUsage,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::{Settings, SettingsStore, VertexAvailableModel, VertexPublisher};
use smol::process::Command;
use std::sync::atomic::{self, AtomicU64};
use ui::{ButtonLink, ConfiguredApiCard, List, ListBulletItem, prelude::*};
use ui_input::InputField;
use util::ResultExt;
use vertex::{Publisher, gemini, anthropic as vertex_anthropic};

use crate::AllLanguageModelSettings;

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("google-vertex-ai");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Google Vertex AI");

const DEFAULT_CREDENTIALS_COMMAND: &str = "gcloud auth print-access-token";
const DEFAULT_LOCATION: &str = "us-east5";

#[derive(Default, Clone, Debug, PartialEq)]
pub struct VertexSettings {
    pub project: Option<String>,
    pub location: Option<String>,
    pub credentials_command: Option<String>,
    pub credentials_refresh_interval: Option<u64>,
    pub available_models: Vec<VertexAvailableModel>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ModelMode {
    #[default]
    Default,
    Thinking {
        budget_tokens: Option<u32>,
    },
}

pub struct State {
    project: Option<String>,
    location: Option<String>,
    settings: VertexSettings,
    _subscription: Subscription,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.project.is_some()
    }

    fn get_location(&self) -> String {
        self.location
            .clone()
            .or_else(|| self.settings.location.clone())
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string())
    }

    fn get_project(&self) -> Option<String> {
        self.project
            .clone()
            .or_else(|| self.settings.project.clone())
    }

    fn get_credentials_command(&self) -> String {
        self.settings
            .credentials_command
            .clone()
            .unwrap_or_else(|| DEFAULT_CREDENTIALS_COMMAND.to_string())
    }

    fn set_project(&mut self, project: Option<String>, cx: &mut Context<Self>) {
        self.project = project;
        cx.notify();
    }

    fn set_location(&mut self, location: Option<String>, cx: &mut Context<Self>) {
        self.location = location;
        cx.notify();
    }

    fn reset_credentials(&mut self, cx: &mut Context<Self>) {
        self.project = None;
        self.location = None;
        cx.notify();
    }
}

pub struct VertexLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
}

impl VertexLanguageModelProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        let settings = AllLanguageModelSettings::get_global(cx).vertex.clone();
        let state = cx.new(|cx| {
            let subscription = cx.observe_global::<SettingsStore>(|this: &mut State, cx| {
                this.settings = AllLanguageModelSettings::get_global(cx).vertex.clone();
                cx.notify();
            });
            State {
                project: settings.project.clone(),
                location: settings.location.clone(),
                settings,
                _subscription: subscription,
            }
        });

        Self { http_client, state }
    }

    fn create_language_model(&self, model: VertexModel) -> Arc<dyn LanguageModel> {
        Arc::new(VertexLanguageModel {
            id: LanguageModelId::from(model.id().to_string()),
            model,
            http_client: self.http_client.clone(),
            state: self.state.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }
}

impl LanguageModelProvider for VertexLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> language_model::IconOrSvg {
        language_model::IconOrSvg::Icon(IconName::AiGoogle)
    }

    fn default_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(VertexModel::default()))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(VertexModel::default_fast()))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let mut models = BTreeMap::default();

        // Add built-in models
        for model in VertexModel::all() {
            models.insert(model.id().to_string(), model);
        }

        // Override with available models from settings
        for model in &AllLanguageModelSettings::get_global(cx).vertex.available_models {
            let publisher = model
                .publisher
                .map(|p| match p {
                    VertexPublisher::Google => Publisher::Google,
                    VertexPublisher::Anthropic => Publisher::Anthropic,
                })
                .or_else(|| Publisher::detect_from_model(&model.name));

            models.insert(
                model.name.clone(),
                VertexModel::Custom {
                    name: model.name.clone(),
                    display_name: model.display_name.clone(),
                    max_tokens: model.max_tokens,
                    max_output_tokens: model.max_output_tokens,
                    publisher,
                    default_temperature: model.default_temperature,
                    mode: model.mode.map(|m| match m {
                        settings::ModelMode::Default => ModelMode::Default,
                        settings::ModelMode::Thinking { budget_tokens } => {
                            ModelMode::Thinking { budget_tokens }
                        }
                    }),
                },
            );
        }

        models
            .into_values()
            .map(|model| self.create_language_model(model))
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        if self.is_authenticated(cx) {
            return Task::ready(Ok(()));
        }

        // Check if project is configured in settings
        let project = self.state.read(cx).get_project();
        if project.is_some() {
            self.state.update(cx, |state, cx| {
                state.project = project;
                cx.notify();
            });
            return Task::ready(Ok(()));
        }

        Task::ready(Err(AuthenticateError::CredentialsNotFound))
    }

    fn configuration_view(
        &self,
        _target_agent: language_model::ConfigurationViewTargetAgent,
        window: &mut Window,
        cx: &mut App,
    ) -> AnyView {
        cx.new(|cx| ConfigurationView::new(self.state.clone(), window, cx))
            .into()
    }

    fn reset_credentials(&self, cx: &mut App) -> Task<Result<()>> {
        self.state.update(cx, |state, cx| {
            state.reset_credentials(cx);
        });
        Task::ready(Ok(()))
    }
}

impl LanguageModelProviderState for VertexLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

#[derive(Clone, Debug, Default)]
pub enum VertexModel {
    // Anthropic Claude 4.5 models
    #[default]
    ClaudeOpus4_5,
    ClaudeSonnet4_5,
    ClaudeHaiku4_5,
    // Google Gemini 3 models (require global region)
    Gemini3ProPreview,
    Gemini3FlashPreview,
    // Google Gemini 2.5 models
    Gemini2_5Pro,
    Gemini2_5Flash,
    Gemini2_5FlashLite,
    // Custom model from settings
    Custom {
        name: String,
        display_name: Option<String>,
        max_tokens: u64,
        max_output_tokens: Option<u64>,
        publisher: Option<Publisher>,
        default_temperature: Option<f32>,
        mode: Option<ModelMode>,
    },
}

impl VertexModel {
    pub fn all() -> Vec<Self> {
        vec![
            // Claude 4.5 models
            Self::ClaudeOpus4_5,
            Self::ClaudeSonnet4_5,
            Self::ClaudeHaiku4_5,
            // Gemini 3 models
            Self::Gemini3ProPreview,
            Self::Gemini3FlashPreview,
            // Gemini 2.5 models
            Self::Gemini2_5Pro,
            Self::Gemini2_5Flash,
            Self::Gemini2_5FlashLite,
        ]
    }

    pub fn default_fast() -> Self {
        Self::ClaudeSonnet4_5
    }

    pub fn id(&self) -> &str {
        match self {
            // Claude 4.5
            Self::ClaudeOpus4_5 => "claude-opus-4-5@20251101",
            Self::ClaudeSonnet4_5 => "claude-sonnet-4-5@20250929",
            Self::ClaudeHaiku4_5 => "claude-haiku-4-5@20251001",
            // Gemini 3
            Self::Gemini3ProPreview => "gemini-3-pro-preview",
            Self::Gemini3FlashPreview => "gemini-3-flash-preview",
            // Gemini 2.5
            Self::Gemini2_5Pro => "gemini-2.5-pro",
            Self::Gemini2_5Flash => "gemini-2.5-flash",
            Self::Gemini2_5FlashLite => "gemini-2.5-flash-lite",
            Self::Custom { name, .. } => name,
        }
    }

    /// Returns the actual model ID to use in API calls
    pub fn api_model_id(&self) -> &str {
        self.id()
    }

    pub fn display_name(&self) -> &str {
        match self {
            // Claude 4.5
            Self::ClaudeOpus4_5 => "Claude Opus 4.5",
            Self::ClaudeSonnet4_5 => "Claude Sonnet 4.5",
            Self::ClaudeHaiku4_5 => "Claude Haiku 4.5",
            // Gemini 3
            Self::Gemini3ProPreview => "Gemini 3 Pro Preview",
            Self::Gemini3FlashPreview => "Gemini 3 Flash Preview",
            // Gemini 2.5
            Self::Gemini2_5Pro => "Gemini 2.5 Pro",
            Self::Gemini2_5Flash => "Gemini 2.5 Flash",
            Self::Gemini2_5FlashLite => "Gemini 2.5 Flash Lite",
            Self::Custom {
                display_name,
                name,
                ..
            } => display_name.as_deref().unwrap_or(name),
        }
    }

    pub fn publisher(&self) -> Publisher {
        match self {
            Self::ClaudeOpus4_5
            | Self::ClaudeSonnet4_5
            | Self::ClaudeHaiku4_5 => Publisher::Anthropic,
            Self::Gemini3ProPreview
            | Self::Gemini3FlashPreview
            | Self::Gemini2_5Pro
            | Self::Gemini2_5Flash
            | Self::Gemini2_5FlashLite => Publisher::Google,
            Self::Custom { publisher, name, .. } => {
                publisher.unwrap_or_else(|| {
                    Publisher::detect_from_model(name).unwrap_or(Publisher::Google)
                })
            }
        }
    }

    pub fn max_token_count(&self) -> u64 {
        match self {
            // Claude 4.5 - 200K context
            // Note: Sonnet 4.5 supports 1M in Beta, but defaulting to 200K
            Self::ClaudeOpus4_5
            | Self::ClaudeSonnet4_5
            | Self::ClaudeHaiku4_5 => 200_000,
            // Gemini 3 - 1M context
            Self::Gemini3ProPreview
            | Self::Gemini3FlashPreview => 1_000_000,
            // Gemini 2.5 - 1M context
            Self::Gemini2_5Pro
            | Self::Gemini2_5Flash
            | Self::Gemini2_5FlashLite => 1_048_576,
            Self::Custom { max_tokens, .. } => *max_tokens,
        }
    }

    pub fn max_output_tokens(&self) -> u64 {
        match self {
            // Claude 4.5 - all 64K output per Vertex docs
            Self::ClaudeOpus4_5
            | Self::ClaudeSonnet4_5
            | Self::ClaudeHaiku4_5 => 64_000,
            // Gemini 3 - 64K output
            Self::Gemini3ProPreview
            | Self::Gemini3FlashPreview => 64_000,
            // Gemini 2.5 Pro - 65,536 output
            Self::Gemini2_5Pro => 65_536,
            // Gemini 2.5 Flash/Lite - 64K output
            Self::Gemini2_5Flash
            | Self::Gemini2_5FlashLite => 64_000,
            Self::Custom {
                max_output_tokens, ..
            } => max_output_tokens.unwrap_or(8_192),
        }
    }

    pub fn default_temperature(&self) -> f32 {
        match self {
            Self::Custom {
                default_temperature,
                ..
            } => default_temperature.unwrap_or(1.0),
            _ => 1.0,
        }
    }

    pub fn supports_tools(&self) -> bool {
        true
    }

    pub fn supports_images(&self) -> bool {
        true
    }

    pub fn mode(&self) -> ModelMode {
        match self {
            Self::Custom { mode, .. } => mode.clone().unwrap_or_default(),
            _ => ModelMode::Default,
        }
    }

    pub fn cache_configuration(&self) -> Option<LanguageModelCacheConfiguration> {
        match self.publisher() {
            Publisher::Anthropic => Some(LanguageModelCacheConfiguration {
                max_cache_anchors: 4,
                should_speculate: false,
                min_total_token: 2048,
            }),
            Publisher::Google => None,
        }
    }
}

struct VertexLanguageModel {
    id: LanguageModelId,
    model: VertexModel,
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
    request_limiter: RateLimiter,
}

impl VertexLanguageModel {
    async fn get_access_token(command: String) -> Result<String> {
        // Fetch token using the configured command
        // Note: gcloud auth print-access-token already handles caching internally
        let output = Command::new("sh")
            .args(["-c", &command])
            .output()
            .await
            .context("Failed to execute credentials command")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "Credentials command failed: {}",
                stderr.trim()
            ));
        }

        let token = String::from_utf8(output.stdout)
            .context("Invalid UTF-8 in credentials output")?
            .trim()
            .to_string();

        if token.is_empty() {
            return Err(anyhow!("Credentials command returned empty token"));
        }

        Ok(token)
    }
}

impl LanguageModel for VertexLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn supports_tools(&self) -> bool {
        self.model.supports_tools()
    }

    fn supports_images(&self) -> bool {
        self.model.supports_images()
    }

    fn supports_streaming_tools(&self) -> bool {
        matches!(self.model.publisher(), Publisher::Anthropic)
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto
            | LanguageModelToolChoice::Any
            | LanguageModelToolChoice::None => true,
        }
    }

    fn tool_input_format(&self) -> LanguageModelToolSchemaFormat {
        match self.model.publisher() {
            Publisher::Google => LanguageModelToolSchemaFormat::JsonSchemaSubset,
            Publisher::Anthropic => LanguageModelToolSchemaFormat::JsonSchema,
        }
    }

    fn telemetry_id(&self) -> String {
        format!("vertex/{}", self.model.id())
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u64> {
        Some(self.model.max_output_tokens())
    }

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        cx: &App,
    ) -> BoxFuture<'static, Result<u64>> {
        // Vertex AI doesn't expose a direct token counting API, so we use tiktoken estimation
        // (same approach as Bedrock)
        count_vertex_tokens(request, cx)
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<
                'static,
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            >,
            LanguageModelCompletionError,
        >,
    > {
        let http_client = self.http_client.clone();
        let model_id = self.model.api_model_id().to_string();
        let publisher = self.model.publisher();
        let default_temperature = self.model.default_temperature();
        let max_output_tokens = self.model.max_output_tokens();
        let mode = self.model.mode();

        // Read all state synchronously before the async block
        let (project, location, credentials_command) =
            self.state.read_with(cx, |state, _| {
                (
                    state.get_project(),
                    state.get_location(),
                    state.get_credentials_command(),
                )
            });

        let future = self.request_limiter.stream(async move {
            let project = project.ok_or_else(|| {
                LanguageModelCompletionError::Other(anyhow!("Vertex AI project not configured"))
            })?;

            let access_token = Self::get_access_token(credentials_command)
                .await
                .map_err(LanguageModelCompletionError::Other)?;

            match publisher {
                Publisher::Google => {
                    let gemini_request = into_gemini(request, model_id.clone(), mode);
                    let response = gemini::stream_generate_content(
                        http_client.as_ref(),
                        &location,
                        &project,
                        &access_token,
                        gemini_request,
                    )
                    .await
                    .map_err(LanguageModelCompletionError::Other)?;

                    Ok(GeminiEventMapper::new().map_stream(response).boxed())
                }
                Publisher::Anthropic => {
                    let anthropic_request = into_anthropic(
                        request,
                        model_id.clone(),
                        default_temperature,
                        max_output_tokens,
                        mode,
                    );
                    let response = vertex_anthropic::stream_completion(
                        http_client.as_ref(),
                        &location,
                        &project,
                        &access_token,
                        anthropic_request,
                    )
                    .await
                    .map_err(LanguageModelCompletionError::Other)?;

                    Ok(AnthropicEventMapper::new().map_stream(response).boxed())
                }
            }
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }

    fn cache_configuration(&self) -> Option<LanguageModelCacheConfiguration> {
        self.model.cache_configuration()
    }
}

fn into_gemini(
    mut request: LanguageModelRequest,
    model_id: String,
    mode: ModelMode,
) -> gemini::GenerateContentRequest {
    fn map_content(content: Vec<MessageContent>) -> Vec<gemini::Part> {
        content
            .into_iter()
            .flat_map(|content| match content {
                MessageContent::Text(text) => {
                    if !text.is_empty() {
                        vec![gemini::Part::TextPart(gemini::TextPart { text })]
                    } else {
                        vec![]
                    }
                }
                MessageContent::Thinking {
                    text: _,
                    signature: Some(signature),
                } => {
                    if !signature.is_empty() {
                        vec![gemini::Part::ThoughtPart(gemini::ThoughtPart {
                            thought: true,
                            thought_signature: signature,
                        })]
                    } else {
                        vec![]
                    }
                }
                MessageContent::Thinking { .. } => vec![],
                MessageContent::RedactedThinking(_) => vec![],
                MessageContent::Image(image) => {
                    vec![gemini::Part::InlineDataPart(gemini::InlineDataPart {
                        inline_data: gemini::GenerativeContentBlob {
                            mime_type: "image/png".to_string(),
                            data: image.source.to_string(),
                        },
                    })]
                }
                MessageContent::ToolUse(tool_use) => {
                    let thought_signature = tool_use.thought_signature.filter(|s| !s.is_empty());
                    vec![gemini::Part::FunctionCallPart(gemini::FunctionCallPart {
                        function_call: gemini::FunctionCall {
                            name: tool_use.name.to_string(),
                            args: tool_use.input,
                        },
                        thought_signature,
                    })]
                }
                MessageContent::ToolResult(tool_result) => {
                    match tool_result.content {
                        LanguageModelToolResultContent::Text(text) => {
                            vec![gemini::Part::FunctionResponsePart(
                                gemini::FunctionResponsePart {
                                    function_response: gemini::FunctionResponse {
                                        name: tool_result.tool_name.to_string(),
                                        response: serde_json::json!({ "output": text }),
                                    },
                                },
                            )]
                        }
                        LanguageModelToolResultContent::Image(image) => {
                            vec![
                                gemini::Part::FunctionResponsePart(gemini::FunctionResponsePart {
                                    function_response: gemini::FunctionResponse {
                                        name: tool_result.tool_name.to_string(),
                                        response: serde_json::json!({
                                            "output": "Tool responded with an image"
                                        }),
                                    },
                                }),
                                gemini::Part::InlineDataPart(gemini::InlineDataPart {
                                    inline_data: gemini::GenerativeContentBlob {
                                        mime_type: "image/png".to_string(),
                                        data: image.source.to_string(),
                                    },
                                }),
                            ]
                        }
                    }
                }
            })
            .collect()
    }

    let system_instructions = if request
        .messages
        .first()
        .is_some_and(|msg| matches!(msg.role, Role::System))
    {
        let message = request.messages.remove(0);
        Some(gemini::SystemInstruction {
            parts: map_content(message.content),
        })
    } else {
        None
    };

    // Thinking/reasoning mode configuration for Gemini models.
    // - Gemini 3 models use `thinkingLevel` (Low/Medium/High)
    // - Gemini 2.5 Pro and Flash-Lite use `thinkingBudget` (token count)
    // - Gemini 2.5 Flash does not support thinking
    let thinking_config = match (request.thinking_allowed, mode) {
        (true, ModelMode::Thinking { budget_tokens }) => {
            if model_id.starts_with("gemini-3") {
                Some(gemini::ThinkingConfig::for_gemini_3(gemini::ThinkingLevel::High))
            } else if model_id.starts_with("gemini-2.5-pro")
                || model_id.starts_with("gemini-2.5-flash-lite")
            {
                budget_tokens
                    .map(|b| gemini::ThinkingConfig::for_gemini_2_5_with_budget(b))
                    .or_else(|| Some(gemini::ThinkingConfig::for_gemini_2_5(true)))
            } else {
                None
            }
        }
        _ => None,
    };

    gemini::GenerateContentRequest {
        model: gemini::ModelName { model_id },
        system_instruction: system_instructions,
        contents: request
            .messages
            .into_iter()
            .filter_map(|message| {
                let parts = map_content(message.content);
                if parts.is_empty() {
                    None
                } else {
                    Some(gemini::Content {
                        parts,
                        role: match message.role {
                            Role::User => gemini::Role::User,
                            Role::Assistant => gemini::Role::Model,
                            Role::System => gemini::Role::User,
                        },
                    })
                }
            })
            .collect(),
        generation_config: Some(gemini::GenerationConfig {
            candidate_count: Some(1),
            stop_sequences: Some(request.stop),
            max_output_tokens: None,
            temperature: request.temperature.map(|t| t as f64).or(Some(1.0)),
            thinking_config,
            top_p: None,
            top_k: None,
        }),
        safety_settings: None,
        tools: (!request.tools.is_empty()).then(|| {
            vec![gemini::Tool {
                function_declarations: request
                    .tools
                    .into_iter()
                    .map(|tool| gemini::FunctionDeclaration {
                        name: tool.name,
                        description: tool.description,
                        parameters: tool.input_schema,
                    })
                    .collect(),
            }]
        }),
        tool_config: request.tool_choice.map(|choice| gemini::ToolConfig {
            function_calling_config: gemini::FunctionCallingConfig {
                mode: match choice {
                    LanguageModelToolChoice::Auto => gemini::FunctionCallingMode::Auto,
                    LanguageModelToolChoice::Any => gemini::FunctionCallingMode::Any,
                    LanguageModelToolChoice::None => gemini::FunctionCallingMode::None,
                },
                allowed_function_names: None,
            },
        }),
    }
}

fn into_anthropic(
    request: LanguageModelRequest,
    model: String,
    default_temperature: f32,
    max_output_tokens: u64,
    mode: ModelMode,
) -> vertex_anthropic::Request {
    let mut new_messages: Vec<vertex_anthropic::Message> = Vec::new();
    let mut system_message = String::new();

    for message in request.messages {
        if message.contents_empty() {
            continue;
        }

        match message.role {
            Role::User | Role::Assistant => {
                let mut anthropic_message_content: Vec<vertex_anthropic::RequestContent> = message
                    .content
                    .into_iter()
                    .filter_map(|content| match content {
                        MessageContent::Text(text) => {
                            let text = if text.chars().last().is_some_and(|c| c.is_whitespace()) {
                                text.trim_end().to_string()
                            } else {
                                text
                            };
                            if !text.is_empty() {
                                Some(vertex_anthropic::RequestContent::Text {
                                    text,
                                    cache_control: None,
                                })
                            } else {
                                None
                            }
                        }
                        MessageContent::Thinking {
                            text: thinking,
                            signature,
                        } => {
                            if !thinking.is_empty() {
                                Some(vertex_anthropic::RequestContent::Thinking {
                                    thinking,
                                    signature: signature.unwrap_or_default(),
                                    cache_control: None,
                                })
                            } else {
                                None
                            }
                        }
                        MessageContent::RedactedThinking(data) => {
                            if !data.is_empty() {
                                Some(vertex_anthropic::RequestContent::RedactedThinking { data })
                            } else {
                                None
                            }
                        }
                        MessageContent::Image(image) => {
                            Some(vertex_anthropic::RequestContent::Image {
                                source: vertex_anthropic::ImageSource {
                                    source_type: "base64".to_string(),
                                    media_type: "image/png".to_string(),
                                    data: image.source.to_string(),
                                },
                                cache_control: None,
                            })
                        }
                        MessageContent::ToolUse(tool_use) => {
                            Some(vertex_anthropic::RequestContent::ToolUse {
                                id: tool_use.id.to_string(),
                                name: tool_use.name.to_string(),
                                input: tool_use.input,
                                cache_control: None,
                            })
                        }
                        MessageContent::ToolResult(tool_result) => {
                            Some(vertex_anthropic::RequestContent::ToolResult {
                                tool_use_id: tool_result.tool_use_id.to_string(),
                                is_error: tool_result.is_error,
                                content: match tool_result.content {
                                    LanguageModelToolResultContent::Text(text) => {
                                        vertex_anthropic::ToolResultContent::Plain(text.to_string())
                                    }
                                    LanguageModelToolResultContent::Image(image) => {
                                        vertex_anthropic::ToolResultContent::Multipart(vec![
                                            vertex_anthropic::ToolResultPart::Image {
                                                source: vertex_anthropic::ImageSource {
                                                    source_type: "base64".to_string(),
                                                    media_type: "image/png".to_string(),
                                                    data: image.source.to_string(),
                                                },
                                            },
                                        ])
                                    }
                                },
                                cache_control: None,
                            })
                        }
                    })
                    .collect();

                let anthropic_role = match message.role {
                    Role::User => vertex_anthropic::Role::User,
                    Role::Assistant => vertex_anthropic::Role::Assistant,
                    Role::System => unreachable!("System role should never occur here"),
                };

                if let Some(last_message) = new_messages.last_mut()
                    && last_message.role == anthropic_role
                {
                    last_message.content.extend(anthropic_message_content);
                    continue;
                }

                // Mark the last segment of the message as cached
                if message.cache {
                    let cache_control_value = Some(vertex_anthropic::CacheControl {
                        cache_type: vertex_anthropic::CacheControlType::Ephemeral,
                    });
                    for message_content in anthropic_message_content.iter_mut().rev() {
                        match message_content {
                            vertex_anthropic::RequestContent::RedactedThinking { .. } => {
                                // Caching is not possible, fallback to next message
                            }
                            vertex_anthropic::RequestContent::Text { cache_control, .. }
                            | vertex_anthropic::RequestContent::Thinking { cache_control, .. }
                            | vertex_anthropic::RequestContent::Image { cache_control, .. }
                            | vertex_anthropic::RequestContent::ToolUse { cache_control, .. }
                            | vertex_anthropic::RequestContent::ToolResult { cache_control, .. } => {
                                *cache_control = cache_control_value;
                                break;
                            }
                        }
                    }
                }

                new_messages.push(vertex_anthropic::Message {
                    role: anthropic_role,
                    content: anthropic_message_content,
                });
            }
            Role::System => {
                if !system_message.is_empty() {
                    system_message.push_str("\n\n");
                }
                system_message.push_str(&message.string_contents());
            }
        }
    }

    let thinking = match (request.thinking_allowed, mode) {
        (true, ModelMode::Thinking { budget_tokens }) => {
            Some(vertex_anthropic::Thinking::Enabled { budget_tokens })
        }
        _ => None,
    };

    vertex_anthropic::Request {
        anthropic_version: vertex_anthropic::ANTHROPIC_VERTEX_VERSION.to_string(),
        model,
        messages: new_messages,
        max_tokens: max_output_tokens,
        system: if system_message.is_empty() {
            None
        } else {
            Some(vertex_anthropic::StringOrContents::String(system_message))
        },
        thinking,
        tools: request
            .tools
            .into_iter()
            .map(|tool| vertex_anthropic::Tool {
                name: tool.name,
                description: tool.description,
                input_schema: tool.input_schema,
            })
            .collect(),
        tool_choice: request.tool_choice.map(|choice| match choice {
            LanguageModelToolChoice::Auto => vertex_anthropic::ToolChoice::Auto,
            LanguageModelToolChoice::Any => vertex_anthropic::ToolChoice::Any,
            LanguageModelToolChoice::None => vertex_anthropic::ToolChoice::None,
        }),
        stop_sequences: Vec::new(),
        temperature: request.temperature.or(Some(default_temperature)),
        top_k: None,
        top_p: None,
    }
}

pub struct GeminiEventMapper {
    usage: gemini::UsageMetadata,
    stop_reason: StopReason,
}

impl GeminiEventMapper {
    pub fn new() -> Self {
        Self {
            usage: gemini::UsageMetadata::default(),
            stop_reason: StopReason::EndTurn,
        }
    }

    pub fn map_stream(
        mut self,
        events: Pin<Box<dyn Send + Stream<Item = Result<gemini::GenerateContentResponse>>>>,
    ) -> impl Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>
    {
        events
            .map(Some)
            .chain(futures::stream::once(async { None }))
            .flat_map(move |event| {
                futures::stream::iter(match event {
                    Some(Ok(event)) => self.map_event(event),
                    Some(Err(error)) => {
                        vec![Err(LanguageModelCompletionError::from(error))]
                    }
                    None => vec![Ok(LanguageModelCompletionEvent::Stop(self.stop_reason))],
                })
            })
    }

    pub fn map_event(
        &mut self,
        event: gemini::GenerateContentResponse,
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

        let mut events: Vec<_> = Vec::new();
        let mut wants_to_use_tool = false;

        if let Some(usage_metadata) = event.usage_metadata {
            update_gemini_usage(&mut self.usage, &usage_metadata);
            events.push(Ok(LanguageModelCompletionEvent::UsageUpdate(
                convert_gemini_usage(&self.usage),
            )));
        }

        if let Some(prompt_feedback) = event.prompt_feedback
            && let Some(block_reason) = prompt_feedback.block_reason.as_deref()
        {
            self.stop_reason = match block_reason {
                "SAFETY" | "OTHER" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "IMAGE_SAFETY" => {
                    StopReason::Refusal
                }
                _ => {
                    log::error!("Unexpected Vertex/Gemini block_reason: {block_reason}");
                    StopReason::Refusal
                }
            };
            events.push(Ok(LanguageModelCompletionEvent::Stop(self.stop_reason)));
            return events;
        }

        if let Some(candidates) = event.candidates {
            for candidate in candidates {
                if let Some(finish_reason) = candidate.finish_reason.as_deref() {
                    self.stop_reason = match finish_reason {
                        "STOP" => StopReason::EndTurn,
                        "MAX_TOKENS" => StopReason::MaxTokens,
                        _ => {
                            log::error!("Unexpected Vertex/Gemini finish_reason: {finish_reason}");
                            StopReason::EndTurn
                        }
                    };
                }

                for part in candidate.content.parts {
                    match part {
                        gemini::Part::TextPart(text_part) => {
                            events.push(Ok(LanguageModelCompletionEvent::Text(text_part.text)));
                        }
                        gemini::Part::InlineDataPart(_) => {}
                        gemini::Part::FunctionCallPart(function_call_part) => {
                            wants_to_use_tool = true;
                            let name: Arc<str> = function_call_part.function_call.name.into();
                            let next_tool_id =
                                TOOL_CALL_COUNTER.fetch_add(1, atomic::Ordering::SeqCst);
                            let id: LanguageModelToolUseId =
                                format!("{}-{}", name, next_tool_id).into();

                            let thought_signature = function_call_part
                                .thought_signature
                                .filter(|s| !s.is_empty());

                            events.push(Ok(LanguageModelCompletionEvent::ToolUse(
                                LanguageModelToolUse {
                                    id,
                                    name,
                                    is_input_complete: true,
                                    raw_input: function_call_part.function_call.args.to_string(),
                                    input: function_call_part.function_call.args,
                                    thought_signature,
                                },
                            )));
                        }
                        gemini::Part::FunctionResponsePart(_) => {}
                        gemini::Part::ThoughtPart(part) => {
                            events.push(Ok(LanguageModelCompletionEvent::Thinking {
                                text: "(Encrypted thought)".to_string(),
                                signature: Some(part.thought_signature),
                            }));
                        }
                    }
                }
            }
        }

        if wants_to_use_tool {
            self.stop_reason = StopReason::ToolUse;
            events.push(Ok(LanguageModelCompletionEvent::Stop(StopReason::ToolUse)));
        }

        events
    }
}

fn update_gemini_usage(usage: &mut gemini::UsageMetadata, new: &gemini::UsageMetadata) {
    if let Some(prompt_token_count) = new.prompt_token_count {
        usage.prompt_token_count = Some(prompt_token_count);
    }
    if let Some(cached_content_token_count) = new.cached_content_token_count {
        usage.cached_content_token_count = Some(cached_content_token_count);
    }
    if let Some(candidates_token_count) = new.candidates_token_count {
        usage.candidates_token_count = Some(candidates_token_count);
    }
    if let Some(tool_use_prompt_token_count) = new.tool_use_prompt_token_count {
        usage.tool_use_prompt_token_count = Some(tool_use_prompt_token_count);
    }
    if let Some(thoughts_token_count) = new.thoughts_token_count {
        usage.thoughts_token_count = Some(thoughts_token_count);
    }
    if let Some(total_token_count) = new.total_token_count {
        usage.total_token_count = Some(total_token_count);
    }
}

fn convert_gemini_usage(usage: &gemini::UsageMetadata) -> TokenUsage {
    let prompt_tokens = usage.prompt_token_count.unwrap_or(0);
    let cached_tokens = usage.cached_content_token_count.unwrap_or(0);
    let input_tokens = prompt_tokens.saturating_sub(cached_tokens);
    let output_tokens = usage.candidates_token_count.unwrap_or(0);

    TokenUsage {
        input_tokens,
        output_tokens,
        cache_read_input_tokens: cached_tokens,
        cache_creation_input_tokens: 0,
    }
}

pub struct AnthropicEventMapper {
    tool_uses_by_index: HashMap<usize, RawToolUse>,
    usage: vertex_anthropic::Usage,
    stop_reason: StopReason,
}

struct RawToolUse {
    id: String,
    name: String,
    input_json: String,
}

impl AnthropicEventMapper {
    pub fn new() -> Self {
        Self {
            tool_uses_by_index: HashMap::default(),
            usage: vertex_anthropic::Usage::default(),
            stop_reason: StopReason::EndTurn,
        }
    }

    pub fn map_stream(
        mut self,
        events: Pin<Box<dyn Send + Stream<Item = Result<vertex_anthropic::Event>>>>,
    ) -> impl Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>
    {
        events.flat_map(move |event| {
            futures::stream::iter(match event {
                Ok(event) => self.map_event(event),
                Err(error) => vec![Err(LanguageModelCompletionError::from(error))],
            })
        })
    }

    pub fn map_event(
        &mut self,
        event: vertex_anthropic::Event,
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        match event {
            vertex_anthropic::Event::ContentBlockStart {
                index,
                content_block,
            } => match content_block {
                vertex_anthropic::ResponseContent::Text { text } => {
                    vec![Ok(LanguageModelCompletionEvent::Text(text))]
                }
                vertex_anthropic::ResponseContent::Thinking { thinking } => {
                    vec![Ok(LanguageModelCompletionEvent::Thinking {
                        text: thinking,
                        signature: None,
                    })]
                }
                vertex_anthropic::ResponseContent::RedactedThinking { data } => {
                    vec![Ok(LanguageModelCompletionEvent::RedactedThinking { data })]
                }
                vertex_anthropic::ResponseContent::ToolUse { id, name, .. } => {
                    self.tool_uses_by_index.insert(
                        index,
                        RawToolUse {
                            id,
                            name,
                            input_json: String::new(),
                        },
                    );
                    Vec::new()
                }
            },
            vertex_anthropic::Event::ContentBlockDelta { index, delta } => match delta {
                vertex_anthropic::ContentDelta::TextDelta { text } => {
                    vec![Ok(LanguageModelCompletionEvent::Text(text))]
                }
                vertex_anthropic::ContentDelta::ThinkingDelta { thinking } => {
                    vec![Ok(LanguageModelCompletionEvent::Thinking {
                        text: thinking,
                        signature: None,
                    })]
                }
                vertex_anthropic::ContentDelta::SignatureDelta { signature } => {
                    vec![Ok(LanguageModelCompletionEvent::Thinking {
                        text: "".to_string(),
                        signature: Some(signature),
                    })]
                }
                vertex_anthropic::ContentDelta::InputJsonDelta { partial_json } => {
                    if let Some(tool_use) = self.tool_uses_by_index.get_mut(&index) {
                        tool_use.input_json.push_str(&partial_json);

                        if let Ok(input) = serde_json::Value::from_str(
                            &partial_json_fixer::fix_json(&tool_use.input_json),
                        ) {
                            return vec![Ok(LanguageModelCompletionEvent::ToolUse(
                                LanguageModelToolUse {
                                    id: tool_use.id.clone().into(),
                                    name: tool_use.name.clone().into(),
                                    is_input_complete: false,
                                    raw_input: tool_use.input_json.clone(),
                                    input,
                                    thought_signature: None,
                                },
                            ))];
                        }
                    }
                    vec![]
                }
            },
            vertex_anthropic::Event::ContentBlockStop { index } => {
                if let Some(tool_use) = self.tool_uses_by_index.remove(&index) {
                    let input_json = tool_use.input_json.trim();
                    let input_value = if input_json.is_empty() {
                        Ok(serde_json::Value::Object(serde_json::Map::default()))
                    } else {
                        serde_json::Value::from_str(input_json)
                    };
                    let event_result = match input_value {
                        Ok(input) => Ok(LanguageModelCompletionEvent::ToolUse(
                            LanguageModelToolUse {
                                id: tool_use.id.into(),
                                name: tool_use.name.into(),
                                is_input_complete: true,
                                input,
                                raw_input: tool_use.input_json.clone(),
                                thought_signature: None,
                            },
                        )),
                        Err(json_parse_err) => {
                            Ok(LanguageModelCompletionEvent::ToolUseJsonParseError {
                                id: tool_use.id.into(),
                                tool_name: tool_use.name.into(),
                                raw_input: input_json.into(),
                                json_parse_error: json_parse_err.to_string(),
                            })
                        }
                    };
                    vec![event_result]
                } else {
                    Vec::new()
                }
            }
            vertex_anthropic::Event::MessageStart { message } => {
                update_anthropic_usage(&mut self.usage, &message.usage);
                vec![
                    Ok(LanguageModelCompletionEvent::UsageUpdate(
                        convert_anthropic_usage(&self.usage),
                    )),
                    Ok(LanguageModelCompletionEvent::StartMessage {
                        message_id: message.id,
                    }),
                ]
            }
            vertex_anthropic::Event::MessageDelta { delta, usage } => {
                update_anthropic_usage(&mut self.usage, &usage);
                if let Some(stop_reason) = delta.stop_reason.as_deref() {
                    self.stop_reason = match stop_reason {
                        "end_turn" => StopReason::EndTurn,
                        "max_tokens" => StopReason::MaxTokens,
                        "tool_use" => StopReason::ToolUse,
                        "refusal" => StopReason::Refusal,
                        _ => {
                            log::error!("Unexpected Vertex/Anthropic stop_reason: {stop_reason}");
                            StopReason::EndTurn
                        }
                    };
                }
                vec![Ok(LanguageModelCompletionEvent::UsageUpdate(
                    convert_anthropic_usage(&self.usage),
                ))]
            }
            vertex_anthropic::Event::MessageStop => {
                vec![Ok(LanguageModelCompletionEvent::Stop(self.stop_reason))]
            }
            vertex_anthropic::Event::Error { error } => {
                vec![Err(LanguageModelCompletionError::Other(anyhow!(
                    "{}: {}",
                    error.error_type,
                    error.message
                )))]
            }
            vertex_anthropic::Event::Ping => Vec::new(),
        }
    }
}

fn update_anthropic_usage(usage: &mut vertex_anthropic::Usage, new: &vertex_anthropic::Usage) {
    if let Some(input_tokens) = new.input_tokens {
        usage.input_tokens = Some(input_tokens);
    }
    if let Some(output_tokens) = new.output_tokens {
        usage.output_tokens = Some(output_tokens);
    }
    if let Some(cache_creation_input_tokens) = new.cache_creation_input_tokens {
        usage.cache_creation_input_tokens = Some(cache_creation_input_tokens);
    }
    if let Some(cache_read_input_tokens) = new.cache_read_input_tokens {
        usage.cache_read_input_tokens = Some(cache_read_input_tokens);
    }
}

fn convert_anthropic_usage(usage: &vertex_anthropic::Usage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.input_tokens.unwrap_or(0),
        output_tokens: usage.output_tokens.unwrap_or(0),
        cache_creation_input_tokens: usage.cache_creation_input_tokens.unwrap_or(0),
        cache_read_input_tokens: usage.cache_read_input_tokens.unwrap_or(0),
    }
}

pub fn count_vertex_tokens(
    request: LanguageModelRequest,
    cx: &App,
) -> BoxFuture<'static, Result<u64>> {
    cx.background_executor()
        .spawn(async move {
            let messages = request.messages;
            let mut tokens_from_images = 0;
            let mut string_messages = Vec::with_capacity(messages.len());

            for message in messages {
                let mut string_contents = String::new();

                for content in message.content {
                    match content {
                        MessageContent::Text(text) | MessageContent::Thinking { text, .. } => {
                            string_contents.push_str(&text);
                        }
                        MessageContent::RedactedThinking(_) => {}
                        MessageContent::Image(image) => {
                            tokens_from_images += image.estimate_tokens();
                        }
                        MessageContent::ToolUse(_tool_use) => {
                            // TODO: Estimate token usage from tool uses.
                        }
                        MessageContent::ToolResult(tool_result) => match tool_result.content {
                            LanguageModelToolResultContent::Text(text) => {
                                string_contents.push_str(&text);
                            }
                            LanguageModelToolResultContent::Image(image) => {
                                tokens_from_images += image.estimate_tokens();
                            }
                        },
                    }
                }

                if !string_contents.is_empty() {
                    string_messages.push(tiktoken_rs::ChatCompletionRequestMessage {
                        role: match message.role {
                            Role::User => "user".into(),
                            Role::Assistant => "assistant".into(),
                            Role::System => "system".into(),
                        },
                        content: Some(string_contents),
                        name: None,
                        function_call: None,
                    });
                }
            }

            tiktoken_rs::num_tokens_from_messages("gpt-4", &string_messages)
                .map(|tokens| (tokens + tokens_from_images) as u64)
        })
        .boxed()
}

struct ConfigurationView {
    project_editor: Entity<InputField>,
    location_editor: Entity<InputField>,
    state: Entity<State>,
    load_credentials_task: Option<Task<()>>,
}

impl ConfigurationView {
    fn new(state: Entity<State>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        let project_editor = cx.new(|cx| {
            InputField::new(window, cx, "my-gcp-project")
                .label("GCP Project ID")
        });

        let location_editor = cx.new(|cx| {
            InputField::new(window, cx, DEFAULT_LOCATION)
                .label("Region/Location")
        });

        let load_credentials_task = Some(cx.spawn({
            let state = state.clone();
            async move |this, cx| {
                let task = state.update(cx, |state, _cx| {
                    if state.get_project().is_some() {
                        Task::ready(Ok(()))
                    } else {
                        Task::ready(Err(AuthenticateError::CredentialsNotFound))
                    }
                });
                let _ = task.await;
                this.update(cx, |this, cx| {
                    this.load_credentials_task = None;
                    cx.notify();
                })
                .log_err();
            }
        }));

        Self {
            project_editor,
            location_editor,
            state,
            load_credentials_task,
        }
    }

    fn save_configuration(
        &mut self,
        _: &menu::Confirm,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let project = self.project_editor.read(cx).text(cx).trim().to_string();
        let location = self.location_editor.read(cx).text(cx).trim().to_string();

        if project.is_empty() {
            return;
        }

        self.state.update(cx, |state, cx| {
            state.set_project(Some(project), cx);
            if !location.is_empty() {
                state.set_location(Some(location), cx);
            }
        });
    }

    fn reset_configuration(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.project_editor
            .update(cx, |editor, cx| editor.set_text("", window, cx));
        self.location_editor
            .update(cx, |editor, cx| editor.set_text("", window, cx));

        self.state.update(cx, |state, cx| {
            state.reset_credentials(cx);
        });
    }

    fn should_render_editor(&self, cx: &Context<Self>) -> bool {
        !self.state.read(cx).is_authenticated()
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);
        let project = state.get_project();

        if self.load_credentials_task.is_some() {
            return div()
                .child(Label::new("Loading configuration..."))
                .into_any_element();
        }

        if self.should_render_editor(cx) {
            v_flex()
                .size_full()
                .on_action(cx.listener(Self::save_configuration))
                .child(Label::new(
                    "To use Google Vertex AI, you need to configure your GCP project and ensure gcloud CLI is authenticated.",
                ))
                .child(
                    List::new()
                        .child(
                            ListBulletItem::new("")
                                .child(Label::new("Install and authenticate the"))
                                .child(ButtonLink::new(
                                    "gcloud CLI",
                                    "https://cloud.google.com/sdk/docs/install",
                                )),
                        )
                        .child(ListBulletItem::new(
                            "Run: gcloud auth login && gcloud auth application-default login",
                        ))
                        .child(
                            ListBulletItem::new("")
                                .child(Label::new("Enable the"))
                                .child(ButtonLink::new(
                                    "Vertex AI API",
                                    "https://console.cloud.google.com/apis/library/aiplatform.googleapis.com",
                                )),
                        )
                        .child(ListBulletItem::new(
                            "Enter your GCP project ID below and press Enter",
                        )),
                )
                .child(self.project_editor.clone())
                .child(self.location_editor.clone())
                .child(
                    Label::new(
                        "You can also configure these in settings.json under language_models.vertex",
                    )
                    .size(LabelSize::Small)
                    .color(Color::Muted)
                    .mt_0p5(),
                )
                .into_any_element()
        } else {
            let configured_label = format!(
                "Configured for project: {}",
                project.as_deref().unwrap_or("unknown")
            );

            ConfiguredApiCard::new(configured_label)
                .on_click(cx.listener(|this, _, window, cx| {
                    this.reset_configuration(window, cx)
                }))
                .into_any_element()
        }
    }
}
