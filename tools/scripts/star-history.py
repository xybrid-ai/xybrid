#!/usr/bin/env python3
"""Self-hosted star history: snapshot, backfill and render, with no third party.

GitHub restricted the stargazers *list* endpoint to admins and collaborators on
2026-06-30, which broke star-history.com and every service built on it. The star
*count* (``stargazers_count`` on the repo object) is still public, and the list
still works for people who administer the repo -- so we collect it ourselves.

Subcommands:

``backfill``
    Walk the full stargazer list (needs admin/collaborator rights) and rebuild
    the true historical curve from each ``starred_at`` timestamp. Run once.

``snapshot``
    Append today's public star count. Runs daily in CI, needs no special rights.

``render``
    Turn the collected series into committed SVG charts (light + dark).

Data lives in a JSON Lines file, one ``{"date": ..., "stars": ...}`` per day,
sorted, with one entry per date. Rendering never touches the network.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / ".metrics" / "stars.jsonl"
LIGHT_SVG = REPO_ROOT / "docs" / "assets" / "star-history.svg"
DARK_SVG = REPO_ROOT / "docs" / "assets" / "star-history-dark.svg"
DEFAULT_REPO = "xybrid-ai/xybrid"

# Chart geometry, in SVG user units.
WIDTH, HEIGHT = 800, 360
MARGIN_LEFT, MARGIN_RIGHT = 62, 24
MARGIN_TOP, MARGIN_BOTTOM = 46, 44
PLOT_WIDTH = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_HEIGHT = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

Y_TICKS = 5
X_TICKS = 6

THEMES = {
    "light": {
        "bg": "#ffffff",
        "grid": "#e5e7eb",
        "axis": "#d0d7de",
        "text": "#57606a",
        "title": "#1f2328",
        "line": "#dfb317",
        "fill": "#dfb317",
        "fill_opacity": "0.14",
    },
    "dark": {
        "bg": "#0d1117",
        "grid": "#21262d",
        "axis": "#30363d",
        "text": "#8b949e",
        "title": "#e6edf3",
        "line": "#e3b341",
        "fill": "#e3b341",
        "fill_opacity": "0.16",
    },
}


# --------------------------------------------------------------------------- #
# data access
# --------------------------------------------------------------------------- #


def gh(*args: str) -> str:
    """Run `gh` and return stdout, failing loudly with gh's own stderr."""
    result = subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        sys.exit(f"gh {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


def load_series(path: Path) -> list[tuple[date, int]]:
    if not path.exists():
        return []
    by_date: dict[date, int] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        # Later entries for the same day win; CI may snapshot more than once.
        by_date[date.fromisoformat(record["date"])] = int(record["stars"])
    return sorted(by_date.items())


def save_series(path: Path, series: list[tuple[date, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"date": day.isoformat(), "stars": stars})
        for day, stars in series
    ]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #


def cmd_backfill(args: argparse.Namespace) -> None:
    """Reconstruct the real curve from starred_at timestamps.

    Requires admin or collaborator rights on the repo: since 2026-06-30 the
    stargazers list 401s unauthenticated and 404s for everyone else.
    """
    raw = gh(
        "api",
        "-H",
        "Accept: application/vnd.github.star+json",
        "--paginate",
        f"repos/{args.repo}/stargazers",
        "--jq",
        ".[].starred_at",
    )
    stamps = [line.strip() for line in raw.splitlines() if line.strip()]
    if not stamps:
        sys.exit(
            "No stargazers returned. Either the repo has none, or your token "
            "lacks admin/collaborator rights on it."
        )

    # GitHub caps the list at 40k stargazers. Past that, the early history is
    # unrecoverable and the curve would start mid-air -- say so rather than
    # silently drawing a wrong shape.
    if len(stamps) >= 40_000:
        print(
            f"warning: hit GitHub's 40k stargazer cap; history before "
            f"{stamps[0][:10]} is not retrievable",
            file=sys.stderr,
        )

    days = sorted(date.fromisoformat(stamp[:10]) for stamp in stamps)
    running, series = 0, []
    current = days[0]
    index = 0
    # Emit one cumulative point per calendar day, including flat days, so the
    # x-axis is real time rather than "days something happened".
    while current <= days[-1]:
        while index < len(days) and days[index] == current:
            running += 1
            index += 1
        series.append((current, running))
        current = date.fromordinal(current.toordinal() + 1)

    save_series(args.data, series)
    print(
        f"backfilled {len(series)} days, {series[0][0]} to {series[-1][0]}, "
        f"{series[-1][1]} stars -> {args.data}"
    )


def cmd_snapshot(args: argparse.Namespace) -> None:
    """Append today's star count from the still-public repo endpoint."""
    stars = int(gh("api", f"repos/{args.repo}", "--jq", ".stargazers_count").strip())
    today = datetime.now(timezone.utc).date()

    series = load_series(args.data)
    merged = dict(series)
    if merged.get(today) == stars:
        print(f"no change: {stars} stars on {today}")
        return
    merged[today] = stars
    save_series(args.data, sorted(merged.items()))
    print(f"recorded {stars} stars on {today}")


def cmd_render(args: argparse.Namespace) -> None:
    series = load_series(args.data)
    if len(series) < 2:
        sys.exit(f"need at least 2 data points to draw a chart, found {len(series)}")

    args.light.parent.mkdir(parents=True, exist_ok=True)
    args.light.write_text(render_svg(series, args.repo, THEMES["light"]))
    args.dark.parent.mkdir(parents=True, exist_ok=True)
    args.dark.write_text(render_svg(series, args.repo, THEMES["dark"]))
    print(f"rendered {len(series)} points -> {args.light} and {args.dark}")


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def nice_ceiling(value: int) -> int:
    """Round up to a readable axis maximum (100, 250, 500, 1000, ...)."""
    if value <= 10:
        return 10
    magnitude = 10 ** (len(str(value)) - 1)
    for step in (1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10):
        candidate = int(magnitude * step)
        if candidate >= value:
            return candidate
    return magnitude * 10


def human(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M".replace(".0M", "M")
    if value >= 1_000:
        return f"{value / 1_000:.1f}k".replace(".0k", "k")
    return str(value)


def render_svg(
    series: list[tuple[date, int]], repo: str, theme: dict[str, str]
) -> str:
    first_day, last_day = series[0][0], series[-1][0]
    span_days = max((last_day - first_day).days, 1)
    y_max = nice_ceiling(series[-1][1])

    def x_of(day: date) -> float:
        return MARGIN_LEFT + PLOT_WIDTH * ((day - first_day).days / span_days)

    def y_of(stars: int) -> float:
        return MARGIN_TOP + PLOT_HEIGHT * (1 - stars / y_max)

    points = [(x_of(day), y_of(stars)) for day, stars in series]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    baseline = MARGIN_TOP + PLOT_HEIGHT
    area = (
        f"{points[0][0]:.1f},{baseline:.1f} "
        + line
        + f" {points[-1][0]:.1f},{baseline:.1f}"
    )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{theme["bg"]}"/>',
        f'<text x="{MARGIN_LEFT}" y="26" font-size="15" font-weight="600" '
        f'fill="{theme["title"]}">{repo}</text>',
        f'<text x="{WIDTH - MARGIN_RIGHT}" y="26" font-size="13" '
        f'text-anchor="end" fill="{theme["text"]}">'
        f'{series[-1][1]} stars &#183; {first_day} to {last_day}</text>',
    ]

    # Horizontal gridlines and y labels.
    for tick in range(Y_TICKS + 1):
        value = round(y_max * tick / Y_TICKS)
        y = y_of(value)
        parts.append(
            f'<line x1="{MARGIN_LEFT}" y1="{y:.1f}" x2="{MARGIN_LEFT + PLOT_WIDTH}" '
            f'y2="{y:.1f}" stroke="{theme["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{MARGIN_LEFT - 10}" y="{y + 4:.1f}" font-size="11" '
            f'text-anchor="end" fill="{theme["text"]}">{human(value)}</text>'
        )

    # Date labels along the bottom.
    for tick in range(X_TICKS):
        offset = round(span_days * tick / (X_TICKS - 1))
        day = date.fromordinal(first_day.toordinal() + offset)
        x = x_of(day)
        anchor = "start" if tick == 0 else "end" if tick == X_TICKS - 1 else "middle"
        parts.append(
            f'<text x="{x:.1f}" y="{baseline + 20:.1f}" font-size="11" '
            f'text-anchor="{anchor}" fill="{theme["text"]}">'
            f'{day.strftime("%b %d")}</text>'
        )

    parts.append(
        f'<polygon points="{area}" fill="{theme["fill"]}" '
        f'fill-opacity="{theme["fill_opacity"]}"/>'
    )
    parts.append(
        f'<polyline points="{line}" fill="none" stroke="{theme["line"]}" '
        f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    parts.append(
        f'<circle cx="{points[-1][0]:.1f}" cy="{points[-1][1]:.1f}" r="4" '
        f'fill="{theme["line"]}"/>'
    )
    parts.append(
        f'<line x1="{MARGIN_LEFT}" y1="{baseline:.1f}" '
        f'x2="{MARGIN_LEFT + PLOT_WIDTH}" y2="{baseline:.1f}" '
        f'stroke="{theme["axis"]}" stroke-width="1"/>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--light", type=Path, default=LIGHT_SVG)
    parser.add_argument("--dark", type=Path, default=DARK_SVG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("backfill", help="rebuild history from starred_at (admin only)")
    sub.add_parser("snapshot", help="append today's public star count")
    sub.add_parser("render", help="draw the committed SVG charts")

    args = parser.parse_args()
    {"backfill": cmd_backfill, "snapshot": cmd_snapshot, "render": cmd_render}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
