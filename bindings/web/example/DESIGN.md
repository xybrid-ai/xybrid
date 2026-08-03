# Xybrid LiteRT Browser Demo Design System

## 0. Research Log

- Design read: developer diagnostics for SDK integrators, terminal-native and restrained, with an OpenCode-inspired operational tool surface.
- Embedded refs: shortlisted opencode.ai, vercel, and warp; picked taste + opencode for a deterministic developer-tool surface.
- Skipped lanes: lazyweb and image-concept lanes were ruled out because this is a functional demo harness with no imagery or product-marketing brief.
- Revision 2: reworked from a bare QA jig into a teaching demo. Added the step tracker, run facts, output grid, and live code sample; replaced the status side rail with a full border; added pass and error state hues on top of the explicit text labels.
- Revision 3: the landing page became a language-model demo (SmolLM2-135M via LiteRT-LM) and the deterministic addition jig moved to `/tensor.html`. Same system, two new components: a prompt textarea and a streamed-output panel; the load step note doubles as the model download progress readout, and DONE joins PASS in the status lexicon for generative runs that have no exact expected output.

## 1. Atmosphere & Identity

A quiet, local diagnostic console for SDK integrators. The signature is a warm terminal field with a single blue interaction rail: dense enough for reproducible inference evidence, never decorative. The page must teach as it runs: every phase of the inference is visible, timed, and mapped to the API call that produced it. Dials: variance 3, motion 2, density 6.

## 2. Color

| Role | Token | Value | Usage |
|---|---|---:|---|
| Page | `--surface-page` | `#201d1d` | Full page background |
| Panel | `--surface-panel` | `#302c2c` | Status output and code sample surfaces |
| Text | `--text-primary` | `#fdfcfc` | Primary content |
| Muted text | `--text-muted` | `#9a9898` | Supporting metadata, hints, idle steps |
| Border | `--border-default` | `#777575` | Section dividers, controls, idle marks |
| Subtle border | `--border-subtle` | `#4a4747` | Dense internal boundaries: output grid cells, code block |
| Accent | `--accent-blue` | `#2997ff` | Controls, focus, loading state, active step |
| Accent active | `--accent-blue-active` | `#64b5ff` | Pressed controls |
| Pass | `--state-pass` | `#7fce8f` | PASS label, completed step marks |
| Error | `--state-error` | `#f2705f` | ERROR label, failed step marks |

Blue is the interaction accent. Pass green and error red appear only on state labels and step marks, always paired with explicit PASS, ERROR, check, or cross text; no state is communicated by color alone.

## 3. Typography

| Level | Token | Size | Weight | Line height | Usage |
|---|---|---:|---:|---:|---|
| Title | `--font-title` | 24px | 700 | 1.5 | Page title |
| Body | `--font-body` | 16px | 400 | 1.5 | Default content |
| Small | `--font-small` | 14px | 400 | 1.5 | Steps, facts, code, status labels |
| Caption | `--font-caption` | 12px | 500 | 1.5 | Field labels, hints, step notes, grid cells |

Use `ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace` everywhere. No second font family. Prose blocks cap at 65ch.

## 4. Spacing & Layout

Base unit: 4px. Use only `--space-1` 4px, `--space-2` 8px, `--space-3` 12px, `--space-4` 16px, `--space-5` 20px, `--space-6` 24px, `--space-8` 32px, and `--space-10` 40px.

The content column is `--content-width` 960px with `--page-padding` 16px at 375px, 24px at 768px, and 32px at 1280px. At 375px, everything stacks with full-width controls and a horizontally scrollable output grid. At 768px, the two form fields sit side by side with the run button on its own row. At 1280px, each section splits into a 1:2 grid: heading left, content right.

Page order tells the story: what this is (masthead), run it (form), what happened (steps, status, facts, output grid), how to write it yourself (code sample).

## 5. Components

### Control field
- **Structure**: `label`, then `select` or `input`, then a muted `.hint` line explaining the parameter.
- **States**: default, hover, focus-visible, disabled.
- **Spacing**: `--space-2` label gap and `--space-3` inner padding.
- **Accessibility**: visible label, native controls, blue 2px focus outline, keyboard reachable.
- **Motion**: none.

### Run button
- **Structure**: native `button` with text label. Label reflects state: "Load model and run" before a model exists for the chosen accelerator, "Run again" once one is cached.
- **States**: default, hover, active, focus-visible, disabled, loading.
- **Spacing**: `--space-3` vertical and `--space-4` horizontal padding.
- **Accessibility**: 44px minimum target, explicit busy state, keyboard activation.
- **Motion**: immediate color feedback only, maximum 100ms.

### Step tracker
- **Structure**: ordered list of the three observable phases (load, run, verify). Each row: state mark, bold phase name plus plain-language description, right-aligned note for timings or failure.
- **Variants**: idle (numbered, muted), active (blue number), done (green check plus timing), error (red cross plus "failed here").
- **Accessibility**: state changes are text (check, cross, note), never color alone. The list is not a live region; the status output announces outcomes.
- **Motion**: immediate state swaps, no animation.

### Status output
- **Structure**: `output` with a text state label and detail line, full 1px border, panel background.
- **Variants**: ready (default border), loading (blue), pass (green), error (red). The label text always names the state.
- **Accessibility**: `aria-live="polite"`, text is never color-only.
- **Motion**: immediate state update; reduced motion uses the same behavior.

### Run facts
- **Structure**: definition list of compile path, delegation status, load time, and inference time, followed by a hint explaining delegation. Hidden until a run passes.
- **Spacing**: `--space-2` row gap, 120 to 160px label column.

### Output grid
- **Structure**: `table` with a caption naming the tensor, shape, and formula; 10 by 10 cells with tabular numerals and `--border-subtle` cell borders inside a horizontal scroll container.
- **Empty state**: a muted sentence saying where the values will appear, so the region teaches before the first run.
- **Accessibility**: real table semantics, caption describes the data, cell titles expose the per-element formula.

### Output stream (language-model page)
- **Structure**: a single `p.output-stream` on the panel surface with a subtle border, `pre-wrap` whitespace, and a 65ch measure; text appends as deltas arrive.
- **Empty state**: the same muted teaching sentence pattern as the grid.
- **Progress**: while generating, the generate step note counts streamed characters; while downloading, the load step note counts megabytes. No spinners or bars — the step tracker is the progress surface.

### Prompt field
- **Structure**: `label`, `textarea` (vertical resize only), muted `.hint` naming the turn semantics and token cap.
- **Layout**: spans both form columns at 768px and up.
- **States**: matches the other controls, including a muted disabled state while a run is active.

### Code sample
- **Structure**: `pre` block on the panel surface with a subtle border, showing the exact `@xybrid/web` calls the page makes. Rewrites live as the accelerator or b value changes.
- **Spacing**: `--space-4` padding, horizontal scroll when narrow.

## 6. Motion & Interaction

No decorative motion. Hover and active states change only border or foreground color with `--motion-feedback` 100ms ease-out. Step and status updates are immediate swaps. Respect `prefers-reduced-motion` by removing the color transition.

## 7. Depth & Surface

Use borders and tonal shifts only. `--radius-control` is 4px for every control, mark, and panel. No shadows, gradients, blur, transparency effects, pills, or elevated card treatment. No accent side-stripes; state surfaces use full borders.

## 8. Accessibility Constraints & Accepted Debt

WCAG 2.2 AA is the target: 4.5:1 body contrast, visible keyboard focus, semantic landmarks, explicit labels, full keyboard operation, state text independent of color, and reduced-motion support. No accepted accessibility debt.
