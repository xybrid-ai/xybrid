# Test fixtures

## `qwen_tokenizer.json`

The `tokenizer.json` of [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
at revision `7ae557604adf67be50417f59c2c2f167def9a775`, unmodified (git blob
`443909a61d429dff23010e5bddd28ff530edda00`).

Copyright 2024 Alibaba Cloud. Licensed under the Apache License, Version 2.0;
the license text is the repository's [`LICENSE`](../../../../LICENSE).

The MLX tests write small bundles with random weights and need a real
tokenizer to turn prompts into token ids for them. It is a test input only: it
is excluded from the published `xybrid-core` crate and nothing links it into a
binary.
