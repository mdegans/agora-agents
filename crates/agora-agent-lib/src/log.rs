use misanthropic::response::TokenCounts;
use serde::Serialize;

#[macro_export]
macro_rules! payload {
      ($level:ident, $payload:expr) => {{
          let p = $payload;
          ::tracing::$level!(
              event_type = ::std::any::type_name_of_val(&p),
              payload = %::serde_json::to_string(&p).unwrap_or_default(),
          );
      }};
  }

#[macro_export]
macro_rules! info_payload {
    ($p:expr) => {
        $crate::payload!(info, $p)
    };
}
#[macro_export]
macro_rules! warn_payload {
    ($p:expr) => {
        $crate::payload!(warn, $p)
    };
}
#[macro_export]
macro_rules! debug_payload {
    ($p:expr) => {
        $crate::payload!(debug, $p)
    };
}
#[macro_export]
macro_rules! error_payload {
    ($p:expr) => {
        $crate::payload!(error, $p)
    };
}

#[derive(Serialize)]
pub struct UsageLog<'a> {
    pub elapsed: std::time::Duration,
    pub usage: TokenCounts,
    pub model: &'a str,
    pub i_tok_s: f64,
    pub o_tok_s: f64,
}

/// Log usage stats at the debug level
pub fn log_usage(elapsed: std::time::Duration, usage: TokenCounts, model: &str) {
    if usage == TokenCounts::default() {
        // Likely backend doen't support Usage
        return;
    }

    let elapsed_secs = elapsed.as_secs_f64();
    if elapsed_secs == 0.0 {
        // Something very wrong
        return;
    }
    let i_tok_s = usage.input_tokens as f64 / elapsed_secs;
    let o_tok_s = usage.output_tokens as f64 / elapsed_secs;

    info_payload!(UsageLog {
        elapsed,
        usage,
        model,
        i_tok_s,
        o_tok_s
    })
}
