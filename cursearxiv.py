#!/usr/bin/env python3
"""
CurseArXiv - A Textual-based ArXiv browser for AI/ML/Wireless research papers.

Focuses on:
- Wireless Physical Layer Optimization (5G, 6G, WiFi)
- Machine Learning / AI techniques for wireless and sensing
- Key CS, EE, and Signal Processing categories
"""

import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import textwrap
import re
import html
import webbrowser

from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Static, ListView, ListItem,
    Label, Input, Button, LoadingIndicator, Markdown
)
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from textual.message import Message
from textual import work
from textual.worker import Worker, get_current_worker


@dataclass
class Paper:
    """Represents an ArXiv paper."""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: datetime
    updated: datetime
    pdf_url: str
    abs_url: str
    relevance_score: float = 0.0
    relevance_tags: list[str] = field(default_factory=list)


class ArXivFetcher:
    """Fetches and filters papers from ArXiv API."""

    BASE_URL = "http://export.arxiv.org/api/query"

    CATEGORIES = [
        "cs.AI", "cs.LG", "cs.IT", "cs.NI",
        "eess.SP", "eess.SY", "stat.ML",
    ]

    WIRELESS_KEYWORDS = {
        "5g": 3, "6g": 3, "lte": 3, "wifi": 3, "wi-fi": 3,
        "ofdm": 3, "mimo": 3, "massive mimo": 4, "beamforming": 3,
        "millimeter wave": 3, "mmwave": 3, "terahertz": 3, "thz": 3,
        "physical layer": 4, "phy layer": 4, "phy-layer": 4,
        "wireless": 2, "radio": 2, "rf": 2, "spectrum": 2,
        "channel estimation": 3, "channel state": 3, "csi": 3,
        "precoding": 3, "equalization": 3, "modulation": 2,
        "ris": 3, "reconfigurable intelligent surface": 4,
        "irs": 3, "intelligent reflecting surface": 4,
        "noma": 3, "non-orthogonal multiple access": 3,
        "cellular": 2, "mobile": 1, "base station": 2,
        "antenna": 2, "propagation": 2, "fading": 2,
        "interference": 2, "snr": 2, "ber": 2, "sinr": 2,
        "radar": 2, "sensing": 2, "localization": 2,
        "isac": 3, "joint radar": 3, "communication sensing": 3,
        "positioning": 2, "ranging": 2,
        "802.11": 3, "nr": 2, "ran": 2, "o-ran": 3, "open ran": 3,
        "iot": 2, "lorawan": 2, "lora": 2, "zigbee": 2,
        "bluetooth": 2, "uwb": 2, "ultra-wideband": 2,
        "resource allocation": 2, "power control": 2, "power allocation": 2,
        "spectrum efficiency": 3, "spectral efficiency": 3,
        "energy efficiency": 2, "throughput": 2, "latency": 1,
    }

    ML_KEYWORDS = {
        "deep learning": 2, "neural network": 2, "machine learning": 2,
        "reinforcement learning": 2, "deep reinforcement": 3,
        "transformer": 2, "attention mechanism": 2,
        "convolutional": 1, "cnn": 1, "rnn": 1, "lstm": 1, "gru": 1,
        "autoencoder": 2, "variational": 1, "vae": 1,
        "generative": 1, "gan": 1, "diffusion": 1,
        "federated learning": 2, "federated": 2,
        "graph neural": 2, "gnn": 2,
        "optimization": 1, "gradient": 1,
        "end-to-end": 2, "data-driven": 2,
        "model-based": 1, "model-free": 1,
    }

    def __init__(self, days_back: int = 7, max_results: int = 200):
        self.days_back = days_back
        self.max_results = max_results

    def _build_query(self) -> str:
        cat_query = " OR ".join(f"cat:{cat}" for cat in self.CATEGORIES)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        date_str = f"submittedDate:[{start_date.strftime('%Y%m%d')}0000 TO {end_date.strftime('%Y%m%d')}2359]"
        search_query = f"({cat_query}) AND {date_str}"
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        return f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

    def _parse_atom_response(self, xml_content: str) -> list[Paper]:
        papers = []
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            return papers

        for entry in root.findall("atom:entry", ns):
            try:
                id_elem = entry.find("atom:id", ns)
                if id_elem is None or id_elem.text is None:
                    continue
                arxiv_id = id_elem.text.split("/abs/")[-1]

                title_elem = entry.find("atom:title", ns)
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No Title"
                title = " ".join(title.split())

                authors = []
                for author in entry.findall("atom:author", ns):
                    name_elem = author.find("atom:name", ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text)

                summary_elem = entry.find("atom:summary", ns)
                abstract = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
                abstract = " ".join(abstract.split())

                categories = []
                for cat in entry.findall("atom:category", ns):
                    term = cat.get("term")
                    if term:
                        categories.append(term)

                primary_cat = entry.find("arxiv:primary_category", ns)
                if primary_cat is not None:
                    term = primary_cat.get("term")
                    if term and term not in categories:
                        categories.insert(0, term)

                published_elem = entry.find("atom:published", ns)
                updated_elem = entry.find("atom:updated", ns)
                published = datetime.now()
                updated = datetime.now()

                if published_elem is not None and published_elem.text:
                    try:
                        published = datetime.fromisoformat(published_elem.text.replace("Z", "+00:00"))
                    except ValueError:
                        pass

                if updated_elem is not None and updated_elem.text:
                    try:
                        updated = datetime.fromisoformat(updated_elem.text.replace("Z", "+00:00"))
                    except ValueError:
                        pass

                pdf_url = ""
                abs_url = ""
                for link in entry.findall("atom:link", ns):
                    href = link.get("href", "")
                    link_title = link.get("title", "")
                    if link_title == "pdf":
                        pdf_url = href
                    elif link.get("rel") == "alternate":
                        abs_url = href

                if not pdf_url and arxiv_id:
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                if not abs_url and arxiv_id:
                    abs_url = f"https://arxiv.org/abs/{arxiv_id}"

                paper = Paper(
                    arxiv_id=arxiv_id, title=title, authors=authors,
                    abstract=abstract, categories=categories,
                    published=published, updated=updated,
                    pdf_url=pdf_url, abs_url=abs_url,
                )
                papers.append(paper)
            except Exception:
                continue

        return papers

    def _calculate_relevance(self, paper: Paper) -> tuple[float, list[str]]:
        text = f"{paper.title} {paper.abstract}".lower()

        def word_match(keyword: str, text: str) -> bool:
            if len(keyword) <= 4 and keyword.isalpha():
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                return bool(re.search(pattern, text))
            return keyword.lower() in text

        wireless_score = 0.0
        wireless_matches = []
        for keyword, weight in self.WIRELESS_KEYWORDS.items():
            if word_match(keyword, text):
                wireless_score += weight
                wireless_matches.append(keyword)

        ml_score = 0.0
        ml_matches = []
        for keyword, weight in self.ML_KEYWORDS.items():
            if word_match(keyword, text):
                ml_score += weight
                ml_matches.append(keyword)

        if wireless_score > 0 and ml_score > 0:
            score = (wireless_score + ml_score) * 1.5
            tags = [f"W:{m}" for m in wireless_matches[:3]] + [f"ML:{m}" for m in ml_matches[:2]]
        elif wireless_score > 0:
            score = wireless_score
            tags = [f"W:{m}" for m in wireless_matches[:4]]
        elif ml_score > 0:
            score = ml_score * 0.5
            tags = [f"ML:{m}" for m in ml_matches[:4]]
        else:
            score = 0.0
            tags = []

        return score, tags

    def fetch_papers(self) -> list[Paper]:
        url = self._build_query()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CurseArXiv/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_content = response.read().decode("utf-8")
        except Exception:
            return []

        papers = self._parse_atom_response(xml_content)

        for paper in papers:
            score, tags = self._calculate_relevance(paper)
            paper.relevance_score = score
            paper.relevance_tags = tags

        relevant_papers = [p for p in papers if p.relevance_score > 0]
        relevant_papers.sort(key=lambda p: p.relevance_score, reverse=True)
        return relevant_papers


class FullPaperFetcher:
    """Fetches full paper text from ArXiv HTML view."""

    @staticmethod
    def get_html_url(arxiv_id: str) -> str:
        base_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
        return f"https://arxiv.org/html/{base_id}"

    def fetch_full_text(self, paper: Paper) -> tuple[bool, str]:
        url = self.get_html_url(paper.arxiv_id)
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "CurseArXiv/1.0 (Academic paper reader)"
            })
            with urllib.request.urlopen(req, timeout=30) as response:
                html_content = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False, f"HTML version not available.\n\nView PDF: {paper.pdf_url}"
            return False, f"Failed to fetch paper: HTTP {e.code}"
        except Exception as e:
            return False, f"Failed to fetch paper: {e}"

        return self._parse_html(html_content, paper)

    def _parse_html(self, html_content: str, paper: Paper) -> tuple[bool, str]:
        lines = []
        lines.append(f"# {paper.title}")
        lines.append("")
        lines.append(f"**Authors:** {', '.join(paper.authors)}")
        lines.append(f"**ArXiv ID:** {paper.arxiv_id}")
        lines.append(f"**Categories:** {', '.join(paper.categories)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        article_match = re.search(
            r'<article[^>]*class="ltx_document[^"]*"[^>]*>(.*?)</article>',
            html_content, re.DOTALL | re.IGNORECASE
        )

        if not article_match:
            article_match = re.search(
                r'<div[^>]*class="ltx_page_content[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</body>',
                html_content, re.DOTALL | re.IGNORECASE
            )

        if not article_match:
            lines.append("Could not extract paper content.")
            lines.append(f"\n[View PDF]({paper.pdf_url})")
            return False, "\n".join(lines)

        content = article_match.group(1)
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<nav[^>]*>.*?</nav>', '', content, flags=re.DOTALL | re.IGNORECASE)

        section_pattern = re.compile(
            r'<(h[1-6])[^>]*class="[^"]*ltx_title[^"]*"[^>]*>(.*?)</\1>',
            re.DOTALL | re.IGNORECASE
        )

        parts = section_pattern.split(content)
        current_section = []

        for i, part in enumerate(parts):
            if i % 3 == 0:
                text = self._html_to_text(part)
                if text.strip():
                    current_section.append(text)
            elif i % 3 == 2:
                if current_section:
                    lines.extend(current_section)
                    lines.append("")
                    current_section = []
                header_text = self._html_to_text(part).strip()
                if header_text:
                    lines.append(f"## {header_text}")
                    lines.append("")

        if current_section:
            lines.extend(current_section)

        if len(lines) <= 10:
            plain_text = self._html_to_text(content)
            lines.append(plain_text)

        return True, "\n".join(lines)

    def _html_to_text(self, html_content: str) -> str:
        text = html_content
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</li>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<li[^>]*>', '  - ', text, flags=re.IGNORECASE)
        text = re.sub(r'<math[^>]*alttext="([^"]*)"[^>]*>.*?</math>', r'`\1`', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<math[^>]*>.*?</math>', '`[math]`', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<figure[^>]*>.*?</figure>', '\n*[Figure]*\n', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<table[^>]*>.*?</table>', '\n*[Table]*\n', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        text = html.unescape(text)

        lines = []
        prev_blank = False
        for line in text.split('\n'):
            line = ' '.join(line.split())
            if line:
                lines.append(line)
                prev_blank = False
            elif not prev_blank:
                lines.append('')
                prev_blank = True

        return '\n'.join(lines)


# ============================================================================
# Textual UI Components
# ============================================================================

class PaperListItem(ListItem):
    """A list item representing a paper."""

    def __init__(self, paper: Paper, index: int) -> None:
        super().__init__()
        self.paper = paper
        self.index = index

    def compose(self) -> ComposeResult:
        score_str = f"[{self.paper.relevance_score:5.1f}]"
        date_str = self.paper.published.strftime("%m/%d")
        tags = " ".join(self.paper.relevance_tags[:3])

        yield Horizontal(
            Label(score_str, classes="score"),
            Label(date_str, classes="date"),
            Label(self.paper.title, classes="title"),
            Label(tags, classes="tags"),
            classes="paper-row"
        )


class PaperDetailView(Container):
    """Detailed view of a single paper."""

    def __init__(self, paper: Paper) -> None:
        super().__init__()
        self.paper = paper

    def compose(self) -> ComposeResult:
        with ScrollableContainer():
            yield Label(self.paper.title, classes="detail-title")
            yield Label(f"ArXiv: {self.paper.arxiv_id}  |  Score: {self.paper.relevance_score:.1f}", classes="detail-meta")
            yield Label(f"Published: {self.paper.published.strftime('%Y-%m-%d')}", classes="detail-meta")
            yield Label(f"Categories: {', '.join(self.paper.categories)}", classes="detail-categories")
            yield Label("")
            yield Label("Authors", classes="detail-section-header")
            yield Label(", ".join(self.paper.authors), classes="detail-authors")
            yield Label("")
            yield Label("Abstract", classes="detail-section-header")
            yield Label(self.paper.abstract, classes="detail-abstract")
            yield Label("")
            yield Label(f"Tags: {', '.join(self.paper.relevance_tags)}", classes="detail-tags")


class FullPaperView(Container):
    """Full paper reading view with markdown rendering."""

    def __init__(self, content: str, paper: Paper) -> None:
        super().__init__()
        self.content = content
        self.paper = paper

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="full-paper-scroll"):
            yield Markdown(self.content, id="full-paper-content")


class SearchModal(ModalScreen[str]):
    """Modal dialog for search input."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, prompt: str = "Search", initial: str = "") -> None:
        super().__init__()
        self.prompt = prompt
        self.initial = initial

    def compose(self) -> ComposeResult:
        with Container(classes="search-modal"):
            yield Label(self.prompt, classes="search-label")
            yield Input(value=self.initial, id="search-input")
            with Horizontal(classes="search-buttons"):
                yield Button("Search", variant="primary", id="search-ok")
                yield Button("Cancel", id="search-cancel")

    def on_mount(self) -> None:
        self.query_one("#search-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "search-ok":
            value = self.query_one("#search-input", Input).value
            self.dismiss(value)
        else:
            self.dismiss("")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss("")


class SettingsModal(ModalScreen[tuple[int, float]]):
    """Modal dialog for settings."""

    def __init__(self, days: int, min_score: float) -> None:
        super().__init__()
        self.days = days
        self.min_score = min_score

    def compose(self) -> ComposeResult:
        with Container(classes="settings-modal"):
            yield Label("Settings", classes="settings-title")
            yield Label("Days to fetch (1-30):")
            yield Input(value=str(self.days), id="days-input", type="integer")
            yield Label("Minimum relevance score:")
            yield Input(value=str(self.min_score), id="score-input", type="number")
            with Horizontal(classes="settings-buttons"):
                yield Button("Apply", variant="primary", id="settings-ok")
                yield Button("Cancel", id="settings-cancel")

    def on_mount(self) -> None:
        self.query_one("#days-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-ok":
            try:
                days = int(self.query_one("#days-input", Input).value)
                days = max(1, min(30, days))
            except ValueError:
                days = self.days
            try:
                score = float(self.query_one("#score-input", Input).value)
                score = max(0, score)
            except ValueError:
                score = self.min_score
            self.dismiss((days, score))
        else:
            self.dismiss((self.days, self.min_score))

    def action_cancel(self) -> None:
        self.dismiss((self.days, self.min_score))


class HelpScreen(ModalScreen):
    """Help screen showing keyboard shortcuts."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        help_text = """# CurseArXiv Help

## Navigation
| Key | Action |
|-----|--------|
| `j` / `↓` | Move down |
| `k` / `↑` | Move up |
| `Enter` | Open detail / full paper view |
| `Escape` | Go back |

## Views
| Key | Action |
|-----|--------|
| `f` | Load full paper text |
| `o` | Open paper in browser |

## Filtering
| Key | Action |
|-----|--------|
| `/` | Search papers |
| `c` | Clear search filter |
| `s` | Settings (days, min score) |

## Actions
| Key | Action |
|-----|--------|
| `r` | Refresh papers from ArXiv |
| `?` | Show this help |
| `q` | Quit |

## About

CurseArXiv fetches papers from ArXiv in AI, ML, Signal Processing,
and related categories. Papers are scored based on relevance to
wireless physical layer optimization and ML topics.

**Categories:** cs.AI, cs.LG, cs.IT, cs.NI, eess.SP, eess.SY, stat.ML

Press `Escape` to close this help.
"""
        with Container(classes="help-modal"):
            with ScrollableContainer():
                yield Markdown(help_text)
            yield Button("Close", variant="primary", id="help-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()


class CurseArXivApp(App):
    """Main Textual application for CurseArXiv."""

    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        height: 100%;
    }

    #paper-list {
        height: 1fr;
        border: solid $primary;
        background: $surface;
    }

    #status-bar {
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 1;
    }

    #detail-container {
        height: 100%;
        padding: 1;
    }

    #loading {
        width: 100%;
        height: 100%;
        content-align: center middle;
    }

    .paper-row {
        width: 100%;
        height: 1;
    }

    .paper-row .score {
        width: 8;
        color: $secondary;
    }

    .paper-row .date {
        width: 6;
        color: $text-muted;
    }

    .paper-row .title {
        width: 1fr;
    }

    .paper-row .tags {
        width: auto;
        max-width: 30;
        color: $success;
    }

    .detail-title {
        text-style: bold;
        color: $secondary;
        margin-bottom: 1;
    }

    .detail-meta {
        color: $text-muted;
    }

    .detail-categories {
        color: $primary;
    }

    .detail-section-header {
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }

    .detail-authors {
        color: $text;
    }

    .detail-abstract {
        color: $text;
        margin: 1 0;
    }

    .detail-tags {
        color: $success;
    }

    .search-modal, .settings-modal, .help-modal {
        width: 60;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $surface;
        border: solid $primary;
    }

    .search-modal {
        height: auto;
    }

    .search-label, .settings-title {
        text-style: bold;
        margin-bottom: 1;
    }

    .search-buttons, .settings-buttons {
        margin-top: 1;
        align: center middle;
    }

    .search-buttons Button, .settings-buttons Button {
        margin: 0 1;
    }

    .help-modal {
        width: 80;
        height: 80%;
    }

    .help-modal ScrollableContainer {
        height: 1fr;
        margin-bottom: 1;
    }

    .help-modal #help-close {
        width: 100%;
    }

    #full-paper-scroll {
        height: 100%;
        padding: 1;
    }

    #full-paper-content {
        padding: 1;
    }

    ListView > ListItem.--highlight {
        background: $accent;
    }

    ListView:focus > ListItem.--highlight {
        background: $accent;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("question_mark", "help", "Help"),
        Binding("r", "refresh", "Refresh"),
        Binding("slash", "search", "Search"),
        Binding("c", "clear_search", "Clear"),
        Binding("s", "settings", "Settings"),
        Binding("f", "full_paper", "Full Paper"),
        Binding("o", "open_browser", "Open"),
        Binding("escape", "back", "Back"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.fetcher = ArXivFetcher()
        self.paper_fetcher = FullPaperFetcher()
        self.papers: list[Paper] = []
        self.filtered_papers: list[Paper] = []
        self.current_paper: Paper | None = None
        self.search_query = ""
        self.days_back = 7
        self.min_score = 0.0
        self.view_stack: list[str] = ["list"]  # Track view history

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="main-container"):
            yield LoadingIndicator(id="loading")
            yield ListView(id="paper-list")
            yield Container(id="detail-container")
        yield Static("Loading papers...", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#detail-container").display = False
        self.query_one("#paper-list").display = False
        self.load_papers()

    @work(exclusive=True, thread=True)
    def load_papers(self) -> None:
        """Load papers in background thread."""
        worker = get_current_worker()
        self.fetcher.days_back = self.days_back
        papers = self.fetcher.fetch_papers()
        if not worker.is_cancelled:
            self.call_from_thread(self._papers_loaded, papers)

    def _papers_loaded(self, papers: list[Paper]) -> None:
        """Called when papers are loaded."""
        self.papers = papers
        self._apply_filter()
        self._update_paper_list()
        self.query_one("#loading").display = False
        self.query_one("#paper-list").display = True
        self._update_status()

    def _apply_filter(self) -> None:
        """Apply search and score filters."""
        filtered = [p for p in self.papers if p.relevance_score >= self.min_score]
        if self.search_query:
            query = self.search_query.lower()
            filtered = [
                p for p in filtered
                if (query in p.title.lower() or
                    query in p.abstract.lower() or
                    any(query in a.lower() for a in p.authors))
            ]
        self.filtered_papers = filtered

    def _update_paper_list(self) -> None:
        """Update the paper list view."""
        list_view = self.query_one("#paper-list", ListView)
        list_view.clear()
        for i, paper in enumerate(self.filtered_papers):
            list_view.append(PaperListItem(paper, i))

    def _update_status(self) -> None:
        """Update status bar."""
        status = f"Papers: {len(self.filtered_papers)}/{len(self.papers)} | Days: {self.days_back} | Min Score: {self.min_score:.1f}"
        if self.search_query:
            status += f" | Filter: '{self.search_query}'"
        self.query_one("#status-bar", Static).update(status)

    def _show_detail(self, paper: Paper) -> None:
        """Show paper detail view."""
        self.current_paper = paper
        self.view_stack.append("detail")

        detail_container = self.query_one("#detail-container")
        detail_container.remove_children()
        detail_container.mount(PaperDetailView(paper))

        self.query_one("#paper-list").display = False
        detail_container.display = True

    def _show_full_paper(self, paper: Paper) -> None:
        """Show full paper view."""
        self.current_paper = paper
        self.view_stack.append("full")

        detail_container = self.query_one("#detail-container")
        detail_container.remove_children()
        detail_container.mount(LoadingIndicator())
        detail_container.display = True
        self.query_one("#paper-list").display = False

        self._fetch_full_paper(paper)

    @work(exclusive=True, thread=True)
    def _fetch_full_paper(self, paper: Paper) -> None:
        """Fetch full paper in background."""
        worker = get_current_worker()
        success, content = self.paper_fetcher.fetch_full_text(paper)
        if not worker.is_cancelled:
            self.call_from_thread(self._full_paper_loaded, content, paper)

    def _full_paper_loaded(self, content: str, paper: Paper) -> None:
        """Called when full paper is loaded."""
        detail_container = self.query_one("#detail-container")
        detail_container.remove_children()
        detail_container.mount(FullPaperView(content, paper))

    def _back_to_list(self) -> None:
        """Return to paper list view."""
        self.view_stack = ["list"]
        self.query_one("#detail-container").display = False
        self.query_one("#paper-list").display = True
        self.query_one("#paper-list").focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle paper selection."""
        if isinstance(event.item, PaperListItem):
            self._show_detail(event.item.paper)

    def action_quit(self) -> None:
        self.exit()

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_refresh(self) -> None:
        self.query_one("#paper-list").display = False
        self.query_one("#detail-container").display = False
        self.query_one("#loading").display = True
        self.query_one("#status-bar", Static).update("Refreshing...")
        self.load_papers()

    def action_search(self) -> None:
        def handle_search(query: str) -> None:
            if query is not None:
                self.search_query = query
                self._apply_filter()
                self._update_paper_list()
                self._update_status()

        self.push_screen(SearchModal("Search papers:", self.search_query), handle_search)

    def action_clear_search(self) -> None:
        self.search_query = ""
        self._apply_filter()
        self._update_paper_list()
        self._update_status()

    def action_settings(self) -> None:
        def handle_settings(result: tuple[int, float]) -> None:
            days, score = result
            if days != self.days_back:
                self.days_back = days
                self.min_score = score
                self.action_refresh()
            elif score != self.min_score:
                self.min_score = score
                self._apply_filter()
                self._update_paper_list()
                self._update_status()

        self.push_screen(SettingsModal(self.days_back, self.min_score), handle_settings)

    def action_full_paper(self) -> None:
        if self.current_paper:
            self._show_full_paper(self.current_paper)
        elif self.filtered_papers:
            list_view = self.query_one("#paper-list", ListView)
            if list_view.highlighted_child and isinstance(list_view.highlighted_child, PaperListItem):
                self._show_full_paper(list_view.highlighted_child.paper)

    def action_open_browser(self) -> None:
        paper = self.current_paper
        if not paper and self.filtered_papers:
            list_view = self.query_one("#paper-list", ListView)
            if list_view.highlighted_child and isinstance(list_view.highlighted_child, PaperListItem):
                paper = list_view.highlighted_child.paper

        if paper:
            webbrowser.open(paper.abs_url)

    def action_back(self) -> None:
        if len(self.view_stack) > 1:
            self.view_stack.pop()
            current_view = self.view_stack[-1]
            if current_view == "list":
                self._back_to_list()
            elif current_view == "detail" and self.current_paper:
                self._show_detail(self.current_paper)

    def action_cursor_down(self) -> None:
        list_view = self.query_one("#paper-list", ListView)
        if list_view.display:
            list_view.action_cursor_down()

    def action_cursor_up(self) -> None:
        list_view = self.query_one("#paper-list", ListView)
        if list_view.display:
            list_view.action_cursor_up()


def main():
    app = CurseArXivApp()
    app.run()


if __name__ == "__main__":
    main()
