//! Browser runtime using xybrid's existing safe llama.cpp wrapper.
//!
//! Bazel links these exports and `xybrid-llama` into one Emscripten module.
//! The browser worker owns each handle and serializes all calls. Cancellation
//! stays in JavaScript while generation is suspended, so no reentrant Rust
//! access occurs while an exclusive engine borrow is live.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "runtime")]
mod runtime {
    use std::error::Error;
    use std::ffi::{c_char, CStr, CString};
    use std::fmt;

    use xybrid_llama::{
        format_chat, generate_streaming, LlamaContext, LlamaError, LlamaModel, LlamaResult,
    };

    /// The worker owns this handle; context must be dropped before its model.
    pub struct Engine {
        context: Option<LlamaContext>,
        model: Option<LlamaModel>,
        error: CString,
        prompt_tokens: usize,
        generated_tokens: usize,
        context_length: usize,
    }

    #[derive(Debug)]
    struct Cancelled;

    impl fmt::Display for Cancelled {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("generation cancelled")
        }
    }

    impl Error for Cancelled {}

    extern "C" {
        fn xybrid_web_on_token(token_id: i32, text: *const c_char) -> i32;
    }

    impl Engine {
        fn record_error(&mut self, error: impl fmt::Display) -> i32 {
            self.error = CString::new(error.to_string().replace('\0', "�"))
                .expect("interior nulls have been removed");
            -1
        }

        fn load(&mut self, path: &str, gpu_layers: i32, context_length: usize) -> LlamaResult<()> {
            if !(1..=32_768).contains(&context_length) {
                return Err(LlamaError::InvalidInput(
                    "context_length must be 1..=32768".into(),
                ));
            }
            self.context = None;
            self.model = None;
            xybrid_llama::backend_init();
            let model = LlamaModel::load(path, gpu_layers)?;
            // A single worker owns inference; one thread needs no shared memory.
            let context = LlamaContext::new(&model, context_length, 1, 128, false)?;
            // llama.cpp pads small contexts for its cache layout. Preserve
            // the caller's token budget even when allocation capacity is larger.
            self.context_length = context_length.min(context.n_ctx());
            self.model = Some(model);
            self.context = Some(context);
            Ok(())
        }

        fn generate(&mut self, prompt: &str, max_tokens: usize) -> LlamaResult<bool> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| LlamaError::InvalidInput("model is not loaded".into()))?;
            let context = self
                .context
                .as_ref()
                .ok_or_else(|| LlamaError::InvalidInput("context is not loaded".into()))?;
            if max_tokens == 0 || max_tokens >= self.context_length {
                return Err(LlamaError::InvalidInput(
                    "max_tokens must be positive and less than the context length".into(),
                ));
            }
            let formatted = format_chat(model, &["user"], &[prompt])?.ok_or_else(|| {
                LlamaError::InvalidInput("model needs a supported embedded chat template".into())
            })?;
            let tokens = model.tokenize_special(&formatted, true)?;
            if tokens.len() + max_tokens >= self.context_length {
                return Err(LlamaError::InvalidInput(
                    "prompt exceeds context budget".into(),
                ));
            }
            context.kv_cache_clear();
            self.prompt_tokens = tokens.len();
            self.generated_tokens = 0;
            let generated_tokens = &mut self.generated_tokens;
            let result = generate_streaming(
                context,
                model,
                &tokens,
                max_tokens,
                0.0,
                1.0,
                0.0,
                1,
                1.0,
                &[],
                None,
                |token_id, text| {
                    *generated_tokens += 1;
                    let text = CString::new(text.replace('\0', "�"))
                        .expect("interior nulls have been removed");
                    // SAFETY: text remains alive while Emscripten suspends this
                    // call. The bridge copies it before yielding to the worker.
                    let cancelled = unsafe { xybrid_web_on_token(token_id, text.as_ptr()) } != 0;
                    if cancelled {
                        Err(Box::new(Cancelled))
                    } else {
                        Ok(())
                    }
                },
                0,
            );
            match result {
                Ok((_, stopped)) => Ok(stopped),
                Err(LlamaError::StreamingCallbackAborted(error)) if error.is::<Cancelled>() => {
                    Ok(true)
                }
                Err(error) => Err(error),
            }
        }
    }

    /// Create a handle owned exclusively by the browser worker.
    #[no_mangle]
    pub extern "C" fn xybrid_web_create() -> *mut Engine {
        Box::into_raw(Box::new(Engine {
            context: None,
            model: None,
            error: CString::default(),
            prompt_tokens: 0,
            generated_tokens: 0,
            context_length: 0,
        }))
    }

    /// Return the actual workspace version compiled into the Rust artifact.
    #[no_mangle]
    pub extern "C" fn xybrid_web_version() -> *const c_char {
        concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
    }

    /// Load a GGUF through the existing Rust RAII model and context wrappers.
    ///
    /// # Safety
    /// `engine` is a live, exclusively owned handle; `path` is a live C string.
    #[no_mangle]
    pub unsafe extern "C" fn xybrid_web_load(
        engine: *mut Engine,
        path: *const c_char,
        gpu_layers: i32,
        context_length: usize,
    ) -> i32 {
        // SAFETY: the worker upholds the handle and string ownership contract.
        let engine = unsafe { &mut *engine };
        let path = unsafe { CStr::from_ptr(path) }.to_string_lossy();
        match engine.load(&path, gpu_layers, context_length) {
            Ok(()) => 0,
            Err(error) => engine.record_error(error),
        }
    }

    /// Generate tokens through xybrid's Rust streaming trampoline.
    ///
    /// # Safety
    /// `engine` is exclusively owned for the entire asynchronous call;
    /// `prompt` remains a live C string until the promise settles.
    #[no_mangle]
    pub unsafe extern "C" fn xybrid_web_generate(
        engine: *mut Engine,
        prompt: *const c_char,
        max_tokens: usize,
    ) -> i32 {
        // SAFETY: the worker serializes operations and retains prompt storage.
        let engine = unsafe { &mut *engine };
        let prompt = unsafe { CStr::from_ptr(prompt) }.to_string_lossy();
        match engine.generate(&prompt, max_tokens) {
            Ok(cancelled) => i32::from(cancelled),
            Err(error) => engine.record_error(error),
        }
    }

    /// Return the last error, valid until the next failed operation or destroy.
    ///
    /// # Safety
    /// `engine` is live and no mutating operation is in flight.
    #[no_mangle]
    pub unsafe extern "C" fn xybrid_web_error(engine: *const Engine) -> *const c_char {
        // SAFETY: handle lifetime and exclusive scheduling are worker-owned.
        unsafe { &*engine }.error.as_ptr()
    }

    /// Return the prompt token count after generation has settled.
    ///
    /// # Safety
    /// `engine` is live and no mutating operation is in flight.
    #[no_mangle]
    pub unsafe extern "C" fn xybrid_web_prompt_tokens(engine: *const Engine) -> usize {
        // SAFETY: worker only reads metrics once the Rust borrow has ended.
        unsafe { &*engine }.prompt_tokens
    }

    /// Return the emitted token count after generation has settled.
    ///
    /// # Safety
    /// `engine` is live and no mutating operation is in flight.
    #[no_mangle]
    pub unsafe extern "C" fn xybrid_web_generated_tokens(engine: *const Engine) -> usize {
        // SAFETY: worker only reads metrics once the Rust borrow has ended.
        unsafe { &*engine }.generated_tokens
    }

    /// Drop context and model, releasing their native allocations.
    ///
    /// # Safety
    /// `engine` came from create, is not in use, and has not been destroyed.
    #[no_mangle]
    pub unsafe extern "C" fn xybrid_web_destroy(engine: *mut Engine) {
        // SAFETY: the worker consumes the handle exactly once after settling.
        drop(unsafe { Box::from_raw(engine) });
    }
}
