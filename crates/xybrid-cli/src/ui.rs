//! Terminal UI primitives for polished CLI output.
//!
//! Provides themed helpers for consistent, visually appealing terminal output
//! across all xybrid CLI commands: panels, headers, tables, key-value displays,
//! status indicators, and branded spinners.

use colored::*;
use indicatif::{ProgressBar, ProgressStyle};

// ── Brand colors ─────────────────────────────────────────────

/// Accent color for primary highlights (model names, commands).
pub fn accent(s: &str) -> ColoredString {
    s.truecolor(120, 180, 255).bold()
}

/// Secondary accent for labels and categories.
pub fn secondary(s: &str) -> ColoredString {
    s.truecolor(180, 140, 255)
}

/// Muted text for paths, hashes, timestamps, secondary info.
pub fn dim(s: &str) -> ColoredString {
    s.truecolor(120, 120, 130)
}

/// Success indicator.
pub fn success(s: &str) -> ColoredString {
    s.truecolor(80, 220, 140).bold()
}

/// Warning indicator.
pub fn warn(s: &str) -> ColoredString {
    s.truecolor(240, 190, 60)
}

/// Error indicator.
pub fn error(s: &str) -> ColoredString {
    s.truecolor(240, 80, 80).bold()
}

/// Value highlight (counts, sizes, metrics).
pub fn value(s: &str) -> ColoredString {
    s.truecolor(100, 220, 220)
}

// ── Brand wordmark ───────────────────────────────────────────

/// The X logo mask — downsampled from the actual xybrid logo.
/// @ = filled (X shape), space = empty. 24 cols x 23 rows.
/// Rows are paired 2:1 into half-block characters for rendering.
const LOGO_MASK: &[&str] = &[
    "                  @@@@@@",
    "                 @@@@@@@",
    "                 @@@@@@ ",
    "    @@@@@       @@@@@@  ",
    "    @@@@@@     @@@@@@@  ",
    "     @@@@@    @@@@@@@   ",
    "      @@@@@   @@@@@@    ",
    "      @@@@@@ @@@@@@     ",
    "       @@@@@@@@@@@      ",
    "        @@@@@@@@@@      ",
    "         @@@@@@@@       ",
    "          @@@@@@        ",
    "         @@@@@@@        ",
    "        @@@@@@@@@       ",
    "       @@@@@@@@@@@      ",
    "      @@@@@@ @@@@@@     ",
    "     @@@@@@@ @@@@@@@    ",
    "    @@@@@@@   @@@@@@@   ",
    "    @@@@@@     @@@@@@@  ",
    "   @@@@@@@     @@@@@@@@ ",
    "  @@@@@@@       @@@@@@@@",
    " @@@@@@@         @@@@@@@",
    "@@@@@@@                 ",
];

/// Print the branded xybrid logo with gradient and engraving lines.
///
/// Renders the X logo from the mask using Unicode half-block characters
/// (▀ ▄ █) for double vertical resolution, with:
/// - Blue→purple horizontal gradient
/// - Alternating bright/dark rows for an engraved depth effect
pub fn brand() {
    let cols = 24;
    let mask: Vec<Vec<bool>> = LOGO_MASK
        .iter()
        .map(|line| {
            let mut row: Vec<bool> = line.chars().map(|c| c == '@').collect();
            row.resize(cols, false);
            row
        })
        .collect();

    let rows = mask.len();
    let paired_h = rows.div_ceil(2);

    println!();

    for py in 0..paired_h {
        let top_y = py * 2;
        let bot_y = py * 2 + 1;

        let top_row = &mask[top_y];
        let bot_row = if bot_y < rows {
            Some(&mask[bot_y])
        } else {
            None
        };

        // Alternating scanlines: even source rows are bright, odd are dark
        let top_bright = top_y % 2 == 0;
        let bot_bright = bot_y % 2 == 0;

        let mut out = String::from("      ");
        for (px, &top) in top_row.iter().enumerate().take(cols) {
            let bot = bot_row.is_some_and(|r| r[px]);

            if !top && !bot {
                out.push(' ');
                continue;
            }

            let t = px as f32 / cols as f32;
            // Base gradient: blue (70,140,255) → purple (190,70,240)
            let base_r = lerp(70, 190, t);
            let base_g = lerp(140, 70, t);
            let base_b = lerp(255, 240, t);

            match (top, bot) {
                (true, true) => {
                    // Both filled — use ▀ with top color as fg, bot color as bg
                    // This lets each half-row have its own brightness
                    let (tr, tg, tb) = scanline(base_r, base_g, base_b, top_bright);
                    let (br, bg, bb) = scanline(base_r, base_g, base_b, bot_bright);
                    out.push_str(&format!(
                        "{}",
                        "▀".truecolor(tr, tg, tb).on_truecolor(br, bg, bb)
                    ));
                }
                (true, false) => {
                    let (r, g, b) = scanline(base_r, base_g, base_b, top_bright);
                    out.push_str(&format!("{}", "▀".truecolor(r, g, b)));
                }
                (false, true) => {
                    let (r, g, b) = scanline(base_r, base_g, base_b, bot_bright);
                    out.push_str(&format!("{}", "▄".truecolor(r, g, b)));
                }
                _ => unreachable!(),
            }
        }
        println!("{}", out);
    }
}

/// Apply scanline brightness: bright rows get full color, dark rows are dimmed.
fn scanline(r: u8, g: u8, b: u8, bright: bool) -> (u8, u8, u8) {
    if bright {
        (r, g, b)
    } else {
        // Darken to ~55% for the engraving line effect
        (
            (r as u16 * 55 / 100) as u8,
            (g as u16 * 55 / 100) as u8,
            (b as u16 * 55 / 100) as u8,
        )
    }
}

/// Print the branded logo with a version subtitle.
pub fn brand_with_version(version: &str) {
    brand();
    let label = format!("      xybrid v{}", version);
    println!("{}", label.truecolor(80, 80, 100));
    println!();
}

fn lerp(a: u8, b: u8, t: f32) -> u8 {
    let t = t.clamp(0.0, 1.0);
    (a as f32 + (b as f32 - a as f32) * t) as u8
}

// ── Box drawing ──────────────────────────────────────────────

/// Print a branded header with the xybrid name and a section title.
///
/// ```text
///   xybrid · Model Registry
///   ────────────────────────
/// ```
pub fn header(title: &str) {
    let brand = "xybrid".truecolor(120, 180, 255).bold();
    let dot = "·".truecolor(80, 80, 90);
    let title_colored = title.truecolor(200, 200, 210);
    println!();
    println!("  {} {} {}", brand, dot, title_colored);
    println!(
        "  {}",
        "─"
            .repeat(terminal_width().saturating_sub(4))
            .truecolor(60, 60, 70)
    );
}

/// Print a section divider with an optional label.
///
/// ```text
///   ── Text-to-Speech ──────
/// ```
pub fn section(label: &str) {
    let w = terminal_width().saturating_sub(4);
    let label_len = label.len() + 4; // "── " + label + " "
    let trail = if w > label_len { w - label_len } else { 4 };
    println!();
    println!(
        "  {} {} {}",
        "──".truecolor(60, 60, 70),
        label.truecolor(180, 140, 255).bold(),
        "─".repeat(trail).truecolor(60, 60, 70)
    );
}

/// Print a framed panel with content lines.
///
/// ```text
///   ╭──────────────────────────╮
///   │  kokoro-82m              │
///   │  Task: text-to-speech    │
///   │  Size: 328 MB · Cached   │
///   ╰──────────────────────────╯
/// ```
pub fn panel(lines: &[String]) {
    let max_len = lines.iter().map(|l| strip_ansi(l).len()).max().unwrap_or(0);
    let inner_w = max_len + 2; // 1 space padding each side

    let border = |l: &str, r: &str, fill: &str| {
        format!("  {}{}{}", l, fill.repeat(inner_w), r).truecolor(60, 60, 70)
    };

    println!("{}", border("╭", "╮", "─"));
    for line in lines {
        let visible_len = strip_ansi(line).len();
        let pad = inner_w.saturating_sub(visible_len + 1);
        println!(
            "  {} {}{}{}",
            "│".truecolor(60, 60, 70),
            line,
            " ".repeat(pad),
            "│".truecolor(60, 60, 70)
        );
    }
    println!("{}", border("╰", "╯", "─"));
}

/// Print a key-value pair with consistent alignment.
pub fn kv(key: &str, val: &str) {
    println!("  {:<16} {}", dim(key), val);
}

/// Print a key-value pair where the value is highlighted.
pub fn kv_accent(key: &str, val: &str) {
    println!("  {:<16} {}", dim(key), accent(val));
}

// ── Status indicators ────────────────────────────────────────

/// Print a success message.
pub fn ok(msg: &str) {
    println!("  {} {}", success("✔"), msg);
}

/// Print a warning message.
pub fn warning(msg: &str) {
    println!("  {} {}", warn("!"), msg);
}

/// Print an error message.
pub fn err(msg: &str) {
    eprintln!("  {} {}", error("✖"), msg);
}

/// Print an info/hint message.
pub fn hint(msg: &str) {
    println!("  {} {}", dim("›"), dim(msg));
}

// ── Table ────────────────────────────────────────────────────

/// A simple column-aligned table.
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    col_widths: Vec<usize>,
}

impl Table {
    pub fn new(headers: Vec<&str>) -> Self {
        let col_widths = headers.iter().map(|h| h.len()).collect();
        Self {
            headers: headers.into_iter().map(String::from).collect(),
            rows: Vec::new(),
            col_widths,
        }
    }

    pub fn row(&mut self, cells: Vec<&str>) {
        for (i, cell) in cells.iter().enumerate() {
            if i < self.col_widths.len() {
                self.col_widths[i] = self.col_widths[i].max(cell.len());
            }
        }
        self.rows
            .push(cells.into_iter().map(String::from).collect());
    }

    pub fn print(&self) {
        // Header
        let header_line: String = self
            .headers
            .iter()
            .enumerate()
            .map(|(i, h)| format!("{:<width$}", h, width = self.col_widths[i] + 2))
            .collect();
        println!("  {}", dim(&header_line));

        // Separator
        let sep: String = self
            .col_widths
            .iter()
            .map(|w| "─".repeat(w + 2))
            .collect::<Vec<_>>()
            .join("");
        println!("  {}", dim(&sep));

        // Rows
        for row in &self.rows {
            let line: Vec<String> = row
                .iter()
                .enumerate()
                .map(|(i, cell)| {
                    let w = self.col_widths.get(i).copied().unwrap_or(cell.len()) + 2;
                    format!("{:<width$}", cell, width = w)
                })
                .collect();
            println!("  {}", line.join(""));
        }
    }
}

// ── Progress / Spinners ──────────────────────────────────────

/// Create a branded download progress bar.
pub fn download_bar(total: u64, label: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  {spinner:.cyan} {msg} {bar:30.blue/black} {bytes}/{total_bytes} {bytes_per_sec} {eta}")
            .unwrap()
            .progress_chars("━╸ "),
    );
    pb.set_message(label.to_string());
    pb
}

/// Drive a [`download_bar`] from an SDK [`DownloadStatus`].
///
/// Takes the byte counts straight from the download rather than
/// back-computing them from the fraction, and re-sizes the bar when the SDK
/// reports a total — a multi-file model's real total is the sum across every
/// artifact, which the caller's `resolved.size_bytes` does not cover.
pub fn apply_download_status(pb: &ProgressBar, status: &xybrid_sdk::DownloadStatus) {
    if let Some(total) = status.total_bytes {
        if pb.length() != Some(total) {
            pb.set_length(total);
        }
    }
    pb.set_position(status.downloaded_bytes);
}

/// Create a branded indeterminate spinner.
pub fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("  {spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✔"]),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb
}

// ── List items ───────────────────────────────────────────────

/// Print a bullet-point list item.
pub fn bullet(primary: &str, secondary_text: &str) {
    println!(
        "  {} {}  {}",
        "▸".truecolor(120, 180, 255),
        accent(primary),
        dim(secondary_text)
    );
}

/// Print a sub-item (indented under a bullet).
pub fn sub(text: &str) {
    println!("      {}", dim(text));
}

// ── Footer ───────────────────────────────────────────────────

/// Print a footer summary line.
pub fn footer(text: &str) {
    println!();
    println!("  {}", dim(text));
    println!();
}

// ── Utilities ────────────────────────────────────────────────

fn terminal_width() -> usize {
    console::Term::stdout().size().1 as usize
}

/// Strip ANSI escape codes for length calculation.
fn strip_ansi(s: &str) -> String {
    console::strip_ansi_codes(s).to_string()
}
