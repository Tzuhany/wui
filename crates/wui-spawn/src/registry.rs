use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use wui_core::event::AgentEvent;

/// Tracks in-flight background sub-agent runs.
#[derive(Default, Clone)]
pub struct AgentRegistry {
    jobs: Arc<Mutex<HashMap<Uuid, Job>>>,
}

struct Job {
    cancel: CancellationToken,
    handle: JoinHandle<Result<String, String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    Running,
    Done(String),
    Failed(String),
    NotFound,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Spawn a sub-agent run in the background. Returns the job ID.
    pub async fn spawn(&self, agent: &wui::Agent, prompt: String) -> Uuid {
        let id = Uuid::new_v4();
        let cancel = CancellationToken::new();
        let agent = agent.clone();
        let ct = cancel.clone();

        let handle = tokio::spawn(async move {
            let mut stream = agent.stream(prompt);
            let mut text = String::new();

            loop {
                let event = tokio::select! {
                    e = stream.next() => match e {
                        Some(e) => e,
                        None    => break,
                    },
                    _ = ct.cancelled() => {
                        return Err("cancelled".to_string());
                    }
                };

                match event {
                    AgentEvent::TextDelta(t) => text.push_str(&t),
                    AgentEvent::Done(_) => break,
                    AgentEvent::Error(e) => return Err(e.to_string()),
                    _ => {}
                }
            }
            Ok(text)
        });

        self.jobs.lock().await.insert(id, Job { cancel, handle });
        id
    }

    pub async fn status(&self, id: Uuid) -> JobStatus {
        let mut jobs = self.jobs.lock().await;
        match jobs.get(&id) {
            None => JobStatus::NotFound,
            Some(job) => {
                if job.handle.is_finished() {
                    let job = jobs.remove(&id).unwrap();
                    match job.handle.await {
                        Ok(Ok(text)) => JobStatus::Done(text),
                        Ok(Err(e)) => JobStatus::Failed(e),
                        Err(e) => JobStatus::Failed(e.to_string()),
                    }
                } else {
                    JobStatus::Running
                }
            }
        }
    }

    pub async fn cancel(&self, id: Uuid) -> bool {
        if let Some(job) = self.jobs.lock().await.get(&id) {
            job.cancel.cancel();
            true
        } else {
            false
        }
    }

    pub async fn wait(&self, id: Uuid) -> JobStatus {
        loop {
            let s = self.status(id).await;
            if !matches!(s, JobStatus::Running) {
                return s;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use wui::Agent;
    use wui_eval::MockProvider;

    fn instant_agent(reply: &'static str) -> Agent {
        let provider = MockProvider::new(vec![MockProvider::text(reply)]);
        Agent::builder(provider)
            .permission(wui::PermissionMode::Auto)
            .build()
    }

    #[tokio::test]
    async fn registry_cancel_stops_job() {
        // Spawn an agent that never finishes (mock queue is empty → panic would
        // indicate test logic error; we cancel before it runs).
        // Use an instant-reply agent and cancel immediately to verify cancel path.
        let registry = AgentRegistry::new();
        let agent = instant_agent("finished");

        let id = registry.spawn(&agent, "Do something.".into()).await;

        // Status should be Running or Done (instant agent may finish before cancel).
        let cancelled = registry.cancel(id).await;
        assert!(cancelled, "cancel should return true for a known job id");

        // After cancel, the job should not be Running.
        // Give a tiny window for the spawned task to notice cancellation.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let status = registry.status(id).await;
        // Accept Done (completed before cancel landed) or NotFound (job finished and
        // was removed) — just not Running.
        assert!(
            !matches!(status, JobStatus::Running),
            "job should not be Running after cancel; got {status:?}"
        );
    }

    #[tokio::test]
    async fn registry_wait_returns_done_for_fast_job() {
        let registry = AgentRegistry::new();
        let agent = instant_agent("the answer is 42");

        let id = registry.spawn(&agent, "What is the answer?".into()).await;
        let status = registry.wait(id).await;

        assert!(matches!(status, JobStatus::Done(text) if text.contains("42")));
    }

    #[tokio::test]
    async fn registry_status_not_found_for_unknown_id() {
        let registry = AgentRegistry::new();
        let unknown = Uuid::new_v4();
        assert_eq!(registry.status(unknown).await, JobStatus::NotFound);
    }
}
