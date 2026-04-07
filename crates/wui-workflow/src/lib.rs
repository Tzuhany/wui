// ============================================================================
// wui-workflow — deterministic orchestration overlay for Wui agents.
//
// This crate is a companion, not the product. Wui's core loop is LLM-driven:
// the model decides what to do next. That is the right default for open-ended
// tasks. This crate adds an explicit-control-flow overlay for when you know
// the structure of the work in advance. Use it when you need it; ignore it
// when the LLM-driven loop is sufficient.
//
// What this crate is NOT:
//   - A claim that deterministic orchestration is better than LLM-driven flow.
//   - A replacement for wui's core loop.
//   - A general graph execution engine.
//
// ── wui-workflow — deterministic orchestration on top of the wui runtime.
//
// wui's core loop is LLM-driven: the model decides what to do next. That is
// the right default for open-ended reasoning tasks. But some workflows are
// better expressed as explicit control flow: sequential stages, conditional
// branches, parallel fan-out. This crate provides those primitives.
//
// The unit of composition is `TextStep` — a named async transformation from
// a `String` input to a `String` output. Every primitive (AgentStep, MapStep,
// Branch, Parallel) implements `TextStep`.
//
// String is the lingua franca: LLM outputs are strings, tool results are
// strings, prompts are strings. If you need structured exchange between steps,
// use JSON: serialize to `String` going out, deserialize at the next step.
//
// Example — a two-stage research pipeline:
//
//   let pipeline = Pipeline::new("research")
//       .step(AgentStep::new("fetch", searcher_agent))
//       .step(AgentStep::new("summarise", summariser_agent));
//
//   let result = pipeline.run("quantum computing").await?;
//   println!("{}", result.output);
//   println!("total: {}ms", result.elapsed.as_millis());
//
// Example — conditional branch:
//
//   let pipeline = Pipeline::new("triage")
//       .step(Branch::new(
//           "route",
//           |input: &str| input.to_lowercase().contains("urgent"),
//           AgentStep::new("urgent_path", urgent_agent),
//           AgentStep::new("normal_path", normal_agent),
//       ));
//
// Example — parallel fan-out:
//
//   let pipeline = Pipeline::new("multi-perspective")
//       .step(Parallel::new("analyse", vec![
//           Arc::new(AgentStep::new("technical", tech_agent)),
//           Arc::new(AgentStep::new("business",  biz_agent)),
//       ], |results| results.join("\n\n")));
//
// ============================================================================

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::task::JoinSet;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

// ── Error ─────────────────────────────────────────────────────────────────────

/// An error produced by a workflow step.
#[derive(Debug, thiserror::Error)]
pub enum WorkflowError {
    #[error("step '{step}' failed: {reason}")]
    StepFailed { step: String, reason: String },

    #[error("workflow cancelled")]
    Cancelled,

    #[error("{0}")]
    Other(String),
}

impl WorkflowError {
    pub fn step(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::StepFailed {
            step: name.into(),
            reason: reason.into(),
        }
    }
}

// ── StepCtx ───────────────────────────────────────────────────────────────────

/// Runtime context passed to every step.
#[derive(Clone)]
pub struct StepCtx {
    /// Cancellation signal. Check periodically in long-running steps.
    pub cancel: CancellationToken,
    /// Name of the enclosing pipeline.
    pub pipeline_name: Arc<str>,
    /// Name of this step.
    pub step_name: Arc<str>,
    /// Zero-based position of this step in the pipeline.
    pub step_index: usize,
}

// ── TextStep ──────────────────────────────────────────────────────────────────

/// A named, composable unit of work: `String → String`.
///
/// All workflow primitives implement this trait. Compose them freely:
/// pipelines are just `Vec<Arc<dyn TextStep>>`.
#[async_trait]
pub trait TextStep: Send + Sync + 'static {
    /// Display name for this step (used in tracing and run records).
    fn name(&self) -> &str;

    /// Execute the step with the given input and return the output.
    async fn run(&self, input: String, ctx: &StepCtx) -> Result<String, WorkflowError>;
}

// ── StepRun ───────────────────────────────────────────────────────────────────

/// The result of a single step execution.
#[derive(Debug, Clone)]
pub struct StepRun {
    /// The step's name.
    pub name: String,
    /// The input the step received.
    pub input: String,
    /// The step's output.
    pub output: String,
    /// How long the step took.
    pub elapsed: Duration,
}

// ── PipelineRun ───────────────────────────────────────────────────────────────

/// The complete result of running a `Pipeline`.
#[derive(Debug, Clone)]
pub struct PipelineRun {
    /// The final output of the last step.
    pub output: String,
    /// Per-step execution records, in order.
    pub steps: Vec<StepRun>,
    /// Total elapsed time across all steps.
    pub elapsed: Duration,
}

impl PipelineRun {
    /// Find a step's record by name.
    pub fn step(&self, name: &str) -> Option<&StepRun> {
        self.steps.iter().find(|s| s.name == name)
    }
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// A sequential pipeline of `TextStep`s.
///
/// Each step's output becomes the next step's input.
/// The pipeline short-circuits on the first error.
///
/// ```rust,ignore
/// let result = Pipeline::new("summarise")
///     .step(fetch_step)
///     .step(summarise_step)
///     .run("https://example.com")
///     .await?;
/// ```
pub struct Pipeline {
    name: String,
    steps: Vec<Arc<dyn TextStep>>,
    cancel: CancellationToken,
}

impl Pipeline {
    /// Create a new empty pipeline.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
            cancel: CancellationToken::new(),
        }
    }

    /// Append a step to the pipeline.
    pub fn step(mut self, step: impl TextStep) -> Self {
        self.steps.push(Arc::new(step));
        self
    }

    /// Append a pre-boxed step (useful for sharing steps across pipelines).
    pub fn step_arc(mut self, step: Arc<dyn TextStep>) -> Self {
        self.steps.push(step);
        self
    }

    /// Share a cancellation token with the pipeline.
    ///
    /// When the token is cancelled, the current step receives cancellation
    /// via `StepCtx::cancel` and the pipeline returns `WorkflowError::Cancelled`.
    pub fn with_cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Run the pipeline with the given initial input.
    pub async fn run(&self, input: impl Into<String>) -> Result<PipelineRun, WorkflowError> {
        let pipeline_start = Instant::now();
        let pipeline_name: Arc<str> = Arc::from(self.name.as_str());
        let mut value = input.into();
        let mut step_runs = Vec::new();

        for (index, step) in self.steps.iter().enumerate() {
            if self.cancel.is_cancelled() {
                return Err(WorkflowError::Cancelled);
            }

            let step_name: Arc<str> = Arc::from(step.name());
            let ctx = StepCtx {
                cancel: self.cancel.clone(),
                pipeline_name: pipeline_name.clone(),
                step_name: step_name.clone(),
                step_index: index,
            };

            let step_start = Instant::now();
            tracing::debug!(pipeline = %self.name, step = %step.name(), index, "step starting");

            let output = step.run(value.clone(), &ctx).await
                .map_err(|e| {
                    tracing::error!(pipeline = %self.name, step = %step.name(), error = %e, "step failed");
                    e
                })?;

            let elapsed = step_start.elapsed();
            tracing::debug!(pipeline = %self.name, step = %step.name(), elapsed_ms = elapsed.as_millis(), "step done");

            step_runs.push(StepRun {
                name: step.name().to_string(),
                input: value,
                output: output.clone(),
                elapsed,
            });
            value = output;
        }

        Ok(PipelineRun {
            output: value,
            steps: step_runs,
            elapsed: pipeline_start.elapsed(),
        })
    }
}

// ── AgentStep ─────────────────────────────────────────────────────────────────

/// A step that runs a [`wui::Agent`] with the input as its prompt.
///
/// The agent's final text response becomes the step's output.
pub struct AgentStep {
    name: String,
    agent: wui::Agent,
}

impl AgentStep {
    pub fn new(name: impl Into<String>, agent: wui::Agent) -> Self {
        Self {
            name: name.into(),
            agent,
        }
    }
}

#[async_trait]
impl TextStep for AgentStep {
    fn name(&self) -> &str {
        &self.name
    }

    async fn run(&self, input: String, ctx: &StepCtx) -> Result<String, WorkflowError> {
        let result = tokio::select! {
            r = self.agent.run(input) => r,
            _ = ctx.cancel.cancelled() => return Err(WorkflowError::Cancelled),
        };
        result.map_err(|e| WorkflowError::step(&self.name, e.to_string()))
    }
}

// ── MapStep ───────────────────────────────────────────────────────────────────

/// A step that applies a synchronous transformation.
///
/// ```rust,ignore
/// let normalise = MapStep::new("normalise", |s: String| s.trim().to_lowercase());
/// ```
pub struct MapStep<F: Fn(String) -> String + Send + Sync + 'static> {
    name: String,
    func: F,
}

impl<F: Fn(String) -> String + Send + Sync + 'static> MapStep<F> {
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
        }
    }
}

#[async_trait]
impl<F: Fn(String) -> String + Send + Sync + 'static> TextStep for MapStep<F> {
    fn name(&self) -> &str {
        &self.name
    }

    async fn run(&self, input: String, _ctx: &StepCtx) -> Result<String, WorkflowError> {
        Ok((self.func)(input))
    }
}

// ── Branch ────────────────────────────────────────────────────────────────────

/// A conditional step that routes to one of two branches.
///
/// The `condition` function receives the current input and returns `true` to
/// take the `yes` branch, `false` to take the `no` branch.
///
/// ```rust,ignore
/// Branch::new(
///     "route",
///     |s: &str| s.starts_with("ERROR"),
///     error_handler_step,
///     normal_step,
/// )
/// ```
pub struct Branch {
    name: String,
    condition: Box<dyn Fn(&str) -> bool + Send + Sync + 'static>,
    yes: Arc<dyn TextStep>,
    no: Arc<dyn TextStep>,
}

impl Branch {
    pub fn new(
        name: impl Into<String>,
        condition: impl Fn(&str) -> bool + Send + Sync + 'static,
        yes: impl TextStep,
        no: impl TextStep,
    ) -> Self {
        Self {
            name: name.into(),
            condition: Box::new(condition),
            yes: Arc::new(yes),
            no: Arc::new(no),
        }
    }
}

#[async_trait]
impl TextStep for Branch {
    fn name(&self) -> &str {
        &self.name
    }

    async fn run(&self, input: String, ctx: &StepCtx) -> Result<String, WorkflowError> {
        let branch_ctx = StepCtx {
            step_name: Arc::from(if (self.condition)(&input) {
                self.yes.name()
            } else {
                self.no.name()
            }),
            ..ctx.clone()
        };
        if (self.condition)(&input) {
            self.yes.run(input, &branch_ctx).await
        } else {
            self.no.run(input, &branch_ctx).await
        }
    }
}

// ── Parallel ──────────────────────────────────────────────────────────────────

/// Runs multiple steps concurrently on the same input, then joins the results.
///
/// All branches receive the same input string. The `join` function combines
/// their outputs (in order) into the step's final output.
///
/// If any branch fails, all remaining branches are cancelled immediately —
/// first via `StepCtx::cancel` (cooperative), then hard-aborted — so no
/// orphaned agent loops, tokens, or side-effects accumulate after a failure.
///
/// ```rust,ignore
/// Parallel::new(
///     "multi-perspective",
///     vec![Arc::new(technical_step), Arc::new(business_step)],
///     |outputs| outputs.join("\n\n---\n\n"),
/// )
/// ```
pub struct Parallel<F: Fn(Vec<String>) -> String + Send + Sync + 'static> {
    name: String,
    branches: Vec<Arc<dyn TextStep>>,
    join: F,
}

impl<F: Fn(Vec<String>) -> String + Send + Sync + 'static> Parallel<F> {
    pub fn new(name: impl Into<String>, branches: Vec<Arc<dyn TextStep>>, join: F) -> Self {
        Self {
            name: name.into(),
            branches,
            join,
        }
    }
}

#[async_trait]
impl<F: Fn(Vec<String>) -> String + Send + Sync + 'static> TextStep for Parallel<F> {
    fn name(&self) -> &str {
        &self.name
    }

    async fn run(&self, input: String, ctx: &StepCtx) -> Result<String, WorkflowError> {
        // Use a child token so cancelling one Parallel step does not propagate
        // upward to the enclosing pipeline — cancellation is scoped to this step.
        let branch_cancel = ctx.cancel.child_token();

        let mut set: JoinSet<(usize, Result<String, WorkflowError>)> = JoinSet::new();

        for (i, branch) in self.branches.iter().enumerate() {
            let branch = branch.clone();
            let input = input.clone();
            let branch_ctx = StepCtx {
                step_name: Arc::from(branch.name()),
                cancel: branch_cancel.clone(),
                ..ctx.clone()
            };
            set.spawn(async move { (i, branch.run(input, &branch_ctx).await) });
        }

        // Collect results keyed by branch index so we can re-order after join.
        let mut outputs: Vec<Option<String>> = vec![None; self.branches.len()];

        while let Some(join_result) = set.join_next().await {
            match join_result {
                Err(e) => {
                    // A branch task panicked.
                    branch_cancel.cancel();
                    set.abort_all();
                    return Err(WorkflowError::step(
                        &self.name,
                        format!("branch panicked: {e}"),
                    ));
                }
                Ok((_, Err(e))) => {
                    // A branch returned an error — cancel all sibling branches.
                    branch_cancel.cancel();
                    set.abort_all();
                    return Err(WorkflowError::step(&self.name, e.to_string()));
                }
                Ok((i, Ok(output))) => {
                    outputs[i] = Some(output);
                }
            }
        }

        // All branches succeeded — unwrap is safe (every slot was filled).
        let ordered: Vec<String> = outputs.into_iter().map(|o| o.unwrap()).collect();
        Ok((self.join)(ordered))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // A trivial echo step for testing.
    struct Echo {
        name: String,
    }

    #[async_trait]
    impl TextStep for Echo {
        fn name(&self) -> &str {
            &self.name
        }
        async fn run(&self, input: String, _ctx: &StepCtx) -> Result<String, WorkflowError> {
            Ok(input)
        }
    }

    // A step that appends a suffix.
    struct Append {
        name: String,
        suffix: String,
    }

    #[async_trait]
    impl TextStep for Append {
        fn name(&self) -> &str {
            &self.name
        }
        async fn run(&self, input: String, _ctx: &StepCtx) -> Result<String, WorkflowError> {
            Ok(format!("{input}{}", self.suffix))
        }
    }

    #[tokio::test]
    async fn pipeline_chains_steps() {
        let result = Pipeline::new("test")
            .step(Append {
                name: "a".into(),
                suffix: "-A".into(),
            })
            .step(Append {
                name: "b".into(),
                suffix: "-B".into(),
            })
            .run("hello")
            .await
            .unwrap();

        assert_eq!(result.output, "hello-A-B");
        assert_eq!(result.steps.len(), 2);
        assert_eq!(result.steps[0].input, "hello");
        assert_eq!(result.steps[0].output, "hello-A");
        assert_eq!(result.steps[1].input, "hello-A");
        assert_eq!(result.steps[1].output, "hello-A-B");
    }

    #[tokio::test]
    async fn map_step_transforms() {
        let result = Pipeline::new("upper")
            .step(MapStep::new("upper", |s: String| s.to_uppercase()))
            .run("hello")
            .await
            .unwrap();
        assert_eq!(result.output, "HELLO");
    }

    #[tokio::test]
    async fn map_step_transforms_input() {
        // Explicit smoke test: MapStep applies a pure String → String function.
        let result = Pipeline::new("reverse")
            .step(MapStep::new("reverse", |s: String| {
                s.chars().rev().collect::<String>()
            }))
            .run("abcde")
            .await
            .unwrap();
        assert_eq!(result.output, "edcba");
        assert_eq!(result.steps[0].name, "reverse");
        assert_eq!(result.steps[0].input, "abcde");
    }

    #[tokio::test]
    async fn branch_takes_yes_path() {
        let result = Pipeline::new("branch_test")
            .step(Branch::new(
                "route",
                |s: &str| s.starts_with("yes"),
                Append {
                    name: "yes_path".into(),
                    suffix: "→yes".into(),
                },
                Append {
                    name: "no_path".into(),
                    suffix: "→no".into(),
                },
            ))
            .run("yes_input")
            .await
            .unwrap();
        assert_eq!(result.output, "yes_input→yes");
    }

    #[tokio::test]
    async fn branch_takes_no_path() {
        let result = Pipeline::new("branch_test")
            .step(Branch::new(
                "route",
                |s: &str| s.starts_with("yes"),
                Append {
                    name: "yes_path".into(),
                    suffix: "→yes".into(),
                },
                Append {
                    name: "no_path".into(),
                    suffix: "→no".into(),
                },
            ))
            .run("no_input")
            .await
            .unwrap();
        assert_eq!(result.output, "no_input→no");
    }

    #[tokio::test]
    async fn parallel_runs_all_branches() {
        let result = Pipeline::new("parallel_test")
            .step(Parallel::new(
                "fanout",
                vec![
                    Arc::new(Append {
                        name: "a".into(),
                        suffix: "-A".into(),
                    }),
                    Arc::new(Append {
                        name: "b".into(),
                        suffix: "-B".into(),
                    }),
                    Arc::new(Append {
                        name: "c".into(),
                        suffix: "-C".into(),
                    }),
                ],
                |outputs| outputs.join("|"),
            ))
            .run("x")
            .await
            .unwrap();
        assert_eq!(result.output, "x-A|x-B|x-C");
    }

    #[tokio::test]
    async fn cancellation_stops_pipeline() {
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = Pipeline::new("cancel_test")
            .with_cancel(cancel)
            .step(Echo {
                name: "step".into(),
            })
            .run("input")
            .await;

        assert!(matches!(result, Err(WorkflowError::Cancelled)));
    }
}
