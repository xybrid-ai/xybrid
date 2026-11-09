#include <emscripten.h>
#include <malloc.h>
#include <stddef.h>

// Yield after every Rust token callback. Worker messages can then set the
// cancellation flag even on the synchronous CPU path. Asyncify suspends
// the complete Rust -> C++ -> Rust -> JS stack until this promise resolves.
EM_ASYNC_JS(int, xybrid_web_on_token, (int token_id, const char * text), {
    Module.xybridToken(token_id, UTF8ToString(text));
    await new Promise(resolve => setTimeout(resolve, 0));
    return Module.xybridCancelled ? 1 : 0;
});

extern "C" size_t xybrid_web_allocated_bytes() {
    return mallinfo().uordblks;
}
