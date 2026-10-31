#!/usr/bin/env python3
"""Builds web/course.json from the course modules and every lab README.

Concept modules (course/**/*.md) carry YAML front matter (title, track, order)
and a "## Flashcards" section of "Q:" / "A:" pairs. Lab READMEs get two cards
each, drawn from their concept and "what to look for" sections.

Usage: python3 tools/build_course.py [--out web/course.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
from pathlib import Path

import yaml
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/venkatacrc/vllm_labs"

TRACKS = [
    ("orientation", "Start here", "How the course is organised."),
    ("foundations", "Foundations", "Systems concepts every lab leans on."),
    ("vllm", "vLLM labs", "Make one vLLM server fast, from FP32 to fully optimized."),
    ("sglang", "SGLang labs", "The same journey in SGLang, plus RadixAttention, HiCache and more."),
    ("llmd", "llm-d labs", "Fleet-level serving: inference-aware routing, P/D, KV tiers, wide-EP, Kubernetes."),
    ("cloud", "Cloud & global", "Deploying an LLM service to a global audience."),
    ("systems", "How it fits together", "One request end to end, a reference architecture, and the playbook."),
]
MODULE_TRACKS = {
    "Orientation": "orientation",
    "Foundations": "foundations",
    "Cloud & Global": "cloud",
    "How it fits together": "systems",
}
LAB_TRACKS = [("vllm", "labs"), ("sglang", "sglang"), ("llmd", "llmd")]
TRACK_NAMES = {"vllm": "vLLM", "sglang": "SGLang", "llmd": "llm-d"}

IDEA_HEADINGS = ("concept", "background")
TAKEAWAY_HEADINGS = ("what to look for", "expected takeaways", "what you should understand",
                     "things to look at")

md = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])


def front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.index("\n---\n", 4)
    return yaml.safe_load(text[4:end]) or {}, text[end + 5:]


def sections(body: str) -> list[tuple[str, str]]:
    """Splits markdown on level-2 headings (outside code fences): [(heading, text)]."""
    out, head, buf, fence = [], "", [], False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        if not fence and line.startswith("## "):
            out.append((head, "\n".join(buf)))
            head, buf = line[3:].strip(), []
        else:
            buf.append(line)
    out.append((head, "\n".join(buf)))
    return out


def title_of(body: str, fallback: str) -> str:
    m = re.search(r"^# (.+)$", body, re.M)
    return m.group(1).strip() if m else fallback


def strip_title(body: str) -> str:
    return re.sub(r"\A\s*# .+\n", "", body, count=1)


def parse_cards(text: str) -> list[tuple[str, str]]:
    cards, q, a, mode = [], None, None, None
    for line in text.splitlines() + [""]:
        s = line.strip()
        if s.startswith("Q:"):
            if q and a:
                cards.append((q, a))
            q, a, mode = s[2:].strip(), None, "q"
        elif s.startswith("A:"):
            a, mode = s[2:].strip(), "a"
        elif not s:
            if q and a:
                cards.append((q, a))
                q = a = None
            mode = None
        elif mode == "q":
            q += " " + s
        elif mode == "a":
            a += " " + s
    return cards


def blocks(text: str) -> list[str]:
    """Raw markdown blocks separated by blank lines, without code fences or headings."""
    out, cur, fence = [], [], False
    for line in text.splitlines() + [""]:
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        if line.strip():
            cur.append(line.rstrip())
        elif cur:
            if not cur[0].lstrip().startswith("#"):
                out.append("\n".join(cur))
            cur = []
    return out


def lab_cards(title: str, track_name: str, secs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    name = title if title.lower().startswith(track_name.lower()) else f"{track_name} {title}"
    cards = []
    idea = next((t for h, t in secs if h.lower().startswith(IDEA_HEADINGS)), secs[0][1])
    parts = blocks(idea)
    if parts:
        answer = [parts[0]]
        while len(answer) < 3 and len(answer) < len(parts) and (
                sum(map(len, answer)) < 200 or answer[-1].endswith(":")):
            answer.append(parts[len(answer)])
        cards.append((f"What is the core idea of {name}?", "\n\n".join(answer)))
    look = next((t for h, t in secs if h.lower().startswith(TAKEAWAY_HEADINGS)), None)
    if look:
        items = list_items(look)[:4]
        answer = "\n".join(f"- {i}" for i in items) if items else (blocks(look) or [""])[0]
        if answer:
            cards.append((f"What should you observe in the results of {name}?", answer))
    return cards


def list_items(text: str) -> list[str]:
    items, cur, fence = [], None, False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        m = re.match(r"^(?:[-*]|\d+\.)\s+(.*)", line)
        if m:
            if cur:
                items.append(cur)
            cur = m.group(1).strip()
        elif cur is not None and line.startswith((" ", "\t")) and line.strip():
            cur += " " + line.strip()
        elif cur is not None and not line.strip():
            items.append(cur)
            cur = None
    if cur:
        items.append(cur)
    return items


class LinkResolver:
    """Points relative links at other lessons when possible, else at GitHub."""

    def __init__(self) -> None:
        self.by_path: dict[str, str] = {}

    def add(self, rel_path: str, lesson_id: str) -> None:
        self.by_path[rel_path] = lesson_id
        if rel_path.endswith("/README.md"):
            self.by_path[rel_path[: -len("/README.md")]] = lesson_id

    def resolve(self, href: str, source: str) -> str:
        if re.match(r"^[a-z]+:", href) or href.startswith(("#", "//")):
            return href
        path, _, frag = href.partition("#")
        target = posixpath.normpath(posixpath.join(posixpath.dirname(source), path)).rstrip("/")
        if target in self.by_path:
            return f"#/lesson/{self.by_path[target]}"
        kind = "tree" if (ROOT / target).is_dir() else "blob"
        return f"{REPO_URL}/{kind}/main/{target}" + (f"#{frag}" if frag else "")


def render(markdown: str, source: str, links: LinkResolver) -> str:
    tokens = md.parse(markdown)
    for tok in tokens:
        for child in tok.children or []:
            if child.type == "link_open":
                child.attrSet("href", links.resolve(child.attrGet("href") or "", source))
    html = md.renderer.render(tokens, md.options, {})
    return html.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")


def card_id(lesson_id: str, q: str) -> str:
    return f"{lesson_id}:{hashlib.sha1(q.encode()).hexdigest()[:8]}"


def collect() -> list[dict]:
    lessons = []
    for path in sorted((ROOT / "course").rglob("*.md")):
        meta, body = front_matter(path.read_text(encoding="utf-8"))
        rel = path.relative_to(ROOT).as_posix()
        track = MODULE_TRACKS[meta["track"]]
        lesson_id = f"{track}-{path.stem.replace('_', '-')}"
        lessons.append(dict(id=lesson_id, track=track, order=float(meta.get("order", 99)),
                            kind="module", source=rel, title=meta.get("title") or title_of(body, path.stem),
                            body=body))
    for track, folder in LAB_TRACKS:
        base = ROOT / folder
        overview = base / "README.md"
        if overview.exists():
            body = overview.read_text(encoding="utf-8")
            lessons.append(dict(id=f"{track}-overview", track=track, order=-1, kind="overview",
                                source=overview.relative_to(ROOT).as_posix(),
                                title=f"{TRACK_NAMES[track]} track overview", body=body))
        for readme in sorted(base.glob("lab*/README.md")):
            body = readme.read_text(encoding="utf-8")
            num = re.match(r"lab(\d+)", readme.parent.name).group(1)
            lessons.append(dict(id=f"{track}-lab{num}", track=track, order=int(num), kind="lab",
                                source=readme.relative_to(ROOT).as_posix(),
                                title=title_of(body, readme.parent.name), body=body))
    return lessons


def build() -> dict:
    lessons = collect()
    links = LinkResolver()
    for l in lessons:
        links.add(l["source"], l["id"])

    tracks = {tid: dict(id=tid, title=title, blurb=blurb, lessons=[]) for tid, title, blurb in TRACKS}
    for l in sorted(lessons, key=lambda l: (l["track"], l["order"], l["source"])):
        body = strip_title(l["body"])
        secs = sections(body)
        if l["kind"] == "module":
            raw_cards = [c for h, t in secs if h.lower() == "flashcards" for c in parse_cards(t)]
            body = re.split(r"^## Flashcards\s*$", body, maxsplit=1, flags=re.M)[0]
        elif l["kind"] == "lab":
            raw_cards = lab_cards(l["title"], TRACK_NAMES[l["track"]], secs)
        else:
            raw_cards = []
        cards = [dict(id=card_id(l["id"], q), q=q, a=a,
                      qHtml=md.renderInline(q), aHtml=render(a, l["source"], links))
                 for q, a in raw_cards]
        tracks[l["track"]]["lessons"].append(dict(
            id=l["id"], title=l["title"], kind=l["kind"], source=l["source"],
            url=f"{REPO_URL}/blob/main/{l['source']}",
            html=render(body, l["source"], links), cards=cards))

    course = dict(title="LLM Serving Labs", repo=REPO_URL,
                  tracks=[t for t in tracks.values() if t["lessons"]])
    digest = hashlib.sha1(json.dumps(course, sort_keys=True).encode()).hexdigest()[:10]
    course["version"] = digest
    return course


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "web" / "course.json"))
    args = ap.parse_args()
    course = build()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(course, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    n_lessons = sum(len(t["lessons"]) for t in course["tracks"])
    n_cards = sum(len(l["cards"]) for t in course["tracks"] for l in t["lessons"])
    print(f"Wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: "
          f"{len(course['tracks'])} tracks, {n_lessons} lessons, {n_cards} flashcards "
          f"(version {course['version']})")


if __name__ == "__main__":
    main()
