#!/usr/bin/env python3
"""
CurseArXiv - An ncurses-based ArXiv browser for AI/ML/Wireless research papers.

Focuses on:
- Wireless Physical Layer Optimization (5G, 6G, WiFi)
- Machine Learning / AI techniques for wireless and sensing
- Key CS, EE, and Signal Processing categories
"""

import curses
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import textwrap
import re
import html
import time


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

    # Categories to search
    CATEGORIES = [
        "cs.AI",      # Artificial Intelligence
        "cs.LG",      # Machine Learning
        "cs.IT",      # Information Theory
        "cs.NI",      # Networking and Internet Architecture
        "eess.SP",    # Signal Processing
        "eess.SY",    # Systems and Control
        "stat.ML",    # Machine Learning (stat)
    ]

    # Keywords for relevance scoring - wireless/physical layer focus
    WIRELESS_KEYWORDS = {
        # Core wireless terms (high weight)
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

        # Sensing/radar terms
        "radar": 2, "sensing": 2, "localization": 2,
        "isac": 3, "joint radar": 3, "communication sensing": 3,
        "positioning": 2, "ranging": 2,

        # Protocol terms
        "802.11": 3, "nr": 2, "ran": 2, "o-ran": 3, "open ran": 3,
        "iot": 2, "lorawan": 2, "lora": 2, "zigbee": 2,
        "bluetooth": 2, "uwb": 2, "ultra-wideband": 2,

        # Optimization terms relevant to wireless
        "resource allocation": 2, "power control": 2, "power allocation": 2,
        "spectrum efficiency": 3, "spectral efficiency": 3,
        "energy efficiency": 2, "throughput": 2, "latency": 1,
    }

    ML_KEYWORDS = {
        # Core ML/AI terms
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
        self.papers: list[Paper] = []

    def _build_query(self) -> str:
        """Build the ArXiv API query URL."""
        # Build category query
        cat_query = " OR ".join(f"cat:{cat}" for cat in self.CATEGORIES)

        # Add date filter
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
        """Parse Atom XML response from ArXiv."""
        papers = []

        # Define namespaces
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
                # Extract ID
                id_elem = entry.find("atom:id", ns)
                if id_elem is None or id_elem.text is None:
                    continue
                arxiv_id = id_elem.text.split("/abs/")[-1]

                # Extract title
                title_elem = entry.find("atom:title", ns)
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No Title"
                title = " ".join(title.split())  # Normalize whitespace

                # Extract authors
                authors = []
                for author in entry.findall("atom:author", ns):
                    name_elem = author.find("atom:name", ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text)

                # Extract abstract
                summary_elem = entry.find("atom:summary", ns)
                abstract = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
                abstract = " ".join(abstract.split())

                # Extract categories
                categories = []
                for cat in entry.findall("atom:category", ns):
                    term = cat.get("term")
                    if term:
                        categories.append(term)

                # Also check arxiv:primary_category
                primary_cat = entry.find("arxiv:primary_category", ns)
                if primary_cat is not None:
                    term = primary_cat.get("term")
                    if term and term not in categories:
                        categories.insert(0, term)

                # Extract dates
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

                # Extract links
                pdf_url = ""
                abs_url = ""
                for link in entry.findall("atom:link", ns):
                    href = link.get("href", "")
                    link_type = link.get("type", "")
                    link_title = link.get("title", "")

                    if link_title == "pdf" or "pdf" in link_type:
                        pdf_url = href
                    elif link.get("rel") == "alternate":
                        abs_url = href

                if not pdf_url and arxiv_id:
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                if not abs_url and arxiv_id:
                    abs_url = f"https://arxiv.org/abs/{arxiv_id}"

                paper = Paper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    categories=categories,
                    published=published,
                    updated=updated,
                    pdf_url=pdf_url,
                    abs_url=abs_url,
                )
                papers.append(paper)

            except Exception:
                continue

        return papers

    def _calculate_relevance(self, paper: Paper) -> tuple[float, list[str]]:
        """Calculate relevance score for a paper based on keywords."""
        score = 0.0
        tags = []

        # Combine title and abstract for searching
        text = f"{paper.title} {paper.abstract}".lower()

        def word_match(keyword: str, text: str) -> bool:
            """Check if keyword matches as a word (not substring)."""
            # For short acronyms, require word boundaries
            if len(keyword) <= 4 and keyword.isalpha():
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                return bool(re.search(pattern, text))
            else:
                return keyword.lower() in text

        # Check wireless keywords
        wireless_score = 0.0
        wireless_matches = []
        for keyword, weight in self.WIRELESS_KEYWORDS.items():
            if word_match(keyword, text):
                wireless_score += weight
                wireless_matches.append(keyword)

        # Check ML keywords
        ml_score = 0.0
        ml_matches = []
        for keyword, weight in self.ML_KEYWORDS.items():
            if word_match(keyword, text):
                ml_score += weight
                ml_matches.append(keyword)

        # Bonus for papers that combine wireless + ML
        if wireless_score > 0 and ml_score > 0:
            score = (wireless_score + ml_score) * 1.5
            tags = [f"W:{m}" for m in wireless_matches[:3]] + [f"ML:{m}" for m in ml_matches[:2]]
        elif wireless_score > 0:
            score = wireless_score
            tags = [f"W:{m}" for m in wireless_matches[:4]]
        elif ml_score > 0:
            score = ml_score * 0.5  # Lower priority for pure ML papers
            tags = [f"ML:{m}" for m in ml_matches[:4]]

        return score, tags

    def fetch_papers(self, progress_callback=None) -> list[Paper]:
        """Fetch papers from ArXiv and filter by relevance."""
        if progress_callback:
            progress_callback("Building query...")

        url = self._build_query()

        if progress_callback:
            progress_callback("Fetching from ArXiv API...")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CurseArXiv/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_content = response.read().decode("utf-8")
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error fetching: {e}")
            return []

        if progress_callback:
            progress_callback("Parsing response...")

        papers = self._parse_atom_response(xml_content)

        if progress_callback:
            progress_callback(f"Scoring {len(papers)} papers...")

        # Calculate relevance scores
        for paper in papers:
            score, tags = self._calculate_relevance(paper)
            paper.relevance_score = score
            paper.relevance_tags = tags

        # Filter to only papers with some relevance
        relevant_papers = [p for p in papers if p.relevance_score > 0]

        # Sort by relevance score
        relevant_papers.sort(key=lambda p: p.relevance_score, reverse=True)

        self.papers = relevant_papers

        if progress_callback:
            progress_callback(f"Found {len(relevant_papers)} relevant papers")

        return relevant_papers


class FullPaperFetcher:
    """Fetches full paper text from ArXiv HTML view."""

    @staticmethod
    def get_html_url(arxiv_id: str) -> str:
        """Get the HTML view URL for a paper."""
        # Remove version suffix if present (e.g., 2401.00001v1 -> 2401.00001)
        base_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
        return f"https://arxiv.org/html/{base_id}"

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean up extracted text."""
        # Decode HTML entities
        text = html.unescape(text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def _extract_text_from_element(element: ET.Element) -> str:
        """Recursively extract text from an XML/HTML element."""
        texts = []

        if element.text:
            texts.append(element.text)

        for child in element:
            # Skip navigation, scripts, styles
            tag = child.tag.lower() if isinstance(child.tag, str) else ""
            classes = child.get("class", "")

            if tag in ("script", "style", "nav", "header", "footer"):
                continue
            if "ltx_page_navbar" in classes or "ltx_TOC" in classes:
                continue

            texts.append(FullPaperFetcher._extract_text_from_element(child))

            if child.tail:
                texts.append(child.tail)

        return " ".join(texts)

    def fetch_full_text(self, paper: Paper, progress_callback=None) -> tuple[bool, list[str]]:
        """
        Fetch the full paper text from ArXiv HTML view.
        Returns (success, list of text sections).
        """
        url = self.get_html_url(paper.arxiv_id)

        if progress_callback:
            progress_callback(f"Fetching HTML from {url}...")

        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "CurseArXiv/1.0 (Academic paper reader)"
            })
            with urllib.request.urlopen(req, timeout=30) as response:
                html_content = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False, ["HTML version not available for this paper.",
                              "This may be an older paper or one without LaTeX source.",
                              f"You can view the PDF at: {paper.pdf_url}"]
            return False, [f"Failed to fetch paper: HTTP {e.code}"]
        except Exception as e:
            return False, [f"Failed to fetch paper: {e}"]

        if progress_callback:
            progress_callback("Parsing HTML content...")

        return self._parse_html(html_content, paper)

    def _parse_html(self, html_content: str, paper: Paper) -> tuple[bool, list[str]]:
        """Parse the HTML and extract structured text."""
        sections = []

        # Add paper header
        sections.append(f"{'=' * 60}")
        sections.append(paper.title.upper())
        sections.append(f"{'=' * 60}")
        sections.append("")
        sections.append(f"Authors: {', '.join(paper.authors)}")
        sections.append(f"ArXiv ID: {paper.arxiv_id}")
        sections.append(f"Categories: {', '.join(paper.categories)}")
        sections.append("")
        sections.append("-" * 60)
        sections.append("")

        # Extract main content using regex (more robust than HTML parsing for varied content)
        # Look for article content
        article_match = re.search(
            r'<article[^>]*class="ltx_document[^"]*"[^>]*>(.*?)</article>',
            html_content,
            re.DOTALL | re.IGNORECASE
        )

        if not article_match:
            # Fallback: try to find main content div
            article_match = re.search(
                r'<div[^>]*class="ltx_page_content[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</body>',
                html_content,
                re.DOTALL | re.IGNORECASE
            )

        if not article_match:
            sections.append("Could not extract paper content from HTML.")
            sections.append(f"Try viewing the PDF: {paper.pdf_url}")
            return False, sections

        content = article_match.group(1)

        # Remove script and style tags
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<nav[^>]*>.*?</nav>', '', content, flags=re.DOTALL | re.IGNORECASE)

        # Extract sections
        # Find section headers (h1, h2, h3, etc.)
        section_pattern = re.compile(
            r'<(h[1-6])[^>]*class="[^"]*ltx_title[^"]*"[^>]*>(.*?)</\1>',
            re.DOTALL | re.IGNORECASE
        )

        # Split content by headers
        parts = section_pattern.split(content)

        current_section = []

        for i, part in enumerate(parts):
            if i % 3 == 0:  # Content between headers
                text = self._html_to_text(part)
                if text.strip():
                    current_section.extend(text.split('\n'))
            elif i % 3 == 2:  # Header text
                # Flush current section
                if current_section:
                    sections.extend(current_section)
                    sections.append("")
                    current_section = []

                # Add header
                header_text = self._html_to_text(part).strip()
                if header_text:
                    sections.append("")
                    sections.append(f"## {header_text}")
                    sections.append("")

        # Flush remaining content
        if current_section:
            sections.extend(current_section)

        # If no content was extracted, try a simpler approach
        if len(sections) <= 10:
            sections.append("")
            sections.append("--- Fallback plain text extraction ---")
            sections.append("")
            plain_text = self._html_to_text(content)
            sections.extend(plain_text.split('\n'))

        return True, sections

    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML to plain text, preserving some structure."""
        text = html_content

        # Replace block elements with newlines
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</li>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<li[^>]*>', '  - ', text, flags=re.IGNORECASE)

        # Handle math elements - extract alt text or just mark as [math]
        text = re.sub(r'<math[^>]*alttext="([^"]*)"[^>]*>.*?</math>', r'[\1]', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<math[^>]*>.*?</math>', '[math]', text, flags=re.DOTALL | re.IGNORECASE)

        # Handle figures
        text = re.sub(r'<figure[^>]*>.*?</figure>', '\n[Figure]\n', text, flags=re.DOTALL | re.IGNORECASE)

        # Handle tables - just mark them
        text = re.sub(r'<table[^>]*>.*?</table>', '\n[Table]\n', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove remaining tags
        text = re.sub(r'<[^>]+>', '', text)

        # Decode HTML entities
        text = html.unescape(text)

        # Clean up whitespace
        lines = []
        for line in text.split('\n'):
            line = ' '.join(line.split())  # Normalize whitespace in line
            if line:
                lines.append(line)
            elif lines and lines[-1]:  # Preserve single blank lines
                lines.append('')

        # Remove multiple consecutive blank lines
        cleaned_lines = []
        prev_blank = False
        for line in lines:
            if not line:
                if not prev_blank:
                    cleaned_lines.append(line)
                prev_blank = True
            else:
                cleaned_lines.append(line)
                prev_blank = False

        return '\n'.join(cleaned_lines)


class PaperSummarizer:
    """Generates concise summaries of papers."""

    @staticmethod
    def summarize(paper: Paper, max_sentences: int = 3) -> str:
        """Create a brief summary from the abstract."""
        abstract = paper.abstract

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', abstract)

        if len(sentences) <= max_sentences:
            return abstract

        # Take first sentence(s) and try to find key contribution sentence
        summary_parts = [sentences[0]]

        # Look for sentences with key phrases
        key_phrases = ["we propose", "we present", "we introduce", "we show",
                       "our method", "our approach", "this paper", "we demonstrate",
                       "our results", "we achieve", "simulation results", "experimental results"]

        for sentence in sentences[1:]:
            lower_sent = sentence.lower()
            if any(phrase in lower_sent for phrase in key_phrases):
                if sentence not in summary_parts:
                    summary_parts.append(sentence)
                    if len(summary_parts) >= max_sentences:
                        break

        # If we don't have enough, add more from the beginning
        while len(summary_parts) < max_sentences and len(summary_parts) < len(sentences):
            next_sent = sentences[len(summary_parts)]
            if next_sent not in summary_parts:
                summary_parts.append(next_sent)

        return " ".join(summary_parts)


class CurseArXivUI:
    """NCurses-based UI for browsing ArXiv papers."""

    # Color pairs
    COLOR_NORMAL = 1
    COLOR_SELECTED = 2
    COLOR_HEADER = 3
    COLOR_HIGHLIGHT = 4
    COLOR_TAG = 5
    COLOR_SCORE = 6
    COLOR_DIM = 7

    # View modes
    VIEW_LIST = 0
    VIEW_DETAIL = 1
    VIEW_ABSTRACT = 2
    VIEW_HELP = 3
    VIEW_FULLPAPER = 4

    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.fetcher = ArXivFetcher()
        self.summarizer = PaperSummarizer()
        self.paper_fetcher = FullPaperFetcher()
        self.papers: list[Paper] = []
        self.filtered_papers: list[Paper] = []
        self.current_idx = 0
        self.scroll_offset = 0
        self.view_mode = self.VIEW_LIST
        self.detail_scroll = 0
        self.search_query = ""
        self.show_summary = True
        self.days_back = 7
        self.min_score = 0.0
        self.status_message = ""
        self.loading = False
        # Full paper content cache
        self.full_paper_lines: list[str] = []
        self.full_paper_wrapped: list[tuple[str, int]] = []  # (line, color)

        self._init_colors()
        self._init_screen()

    def _init_colors(self):
        """Initialize color pairs."""
        curses.start_color()
        curses.use_default_colors()

        curses.init_pair(self.COLOR_NORMAL, curses.COLOR_WHITE, -1)
        curses.init_pair(self.COLOR_SELECTED, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(self.COLOR_HEADER, curses.COLOR_CYAN, -1)
        curses.init_pair(self.COLOR_HIGHLIGHT, curses.COLOR_YELLOW, -1)
        curses.init_pair(self.COLOR_TAG, curses.COLOR_GREEN, -1)
        curses.init_pair(self.COLOR_SCORE, curses.COLOR_MAGENTA, -1)
        curses.init_pair(self.COLOR_DIM, curses.COLOR_WHITE, -1)

    def _init_screen(self):
        """Initialize screen settings."""
        curses.curs_set(0)  # Hide cursor
        self.stdscr.keypad(True)
        self.stdscr.timeout(100)  # Non-blocking input

    def _get_dims(self) -> tuple[int, int]:
        """Get current screen dimensions."""
        return self.stdscr.getmaxyx()

    def _draw_header(self):
        """Draw the header bar."""
        height, width = self._get_dims()

        title = " CurseArXiv - Wireless/ML Paper Browser "

        # Draw header line
        self.stdscr.attron(curses.color_pair(self.COLOR_HEADER) | curses.A_BOLD)
        self.stdscr.addstr(0, 0, "=" * width)

        # Center title
        title_x = max(0, (width - len(title)) // 2)
        self.stdscr.addstr(0, title_x, title)
        self.stdscr.attroff(curses.color_pair(self.COLOR_HEADER) | curses.A_BOLD)

        # Stats line
        stats = f" Papers: {len(self.filtered_papers)}/{len(self.papers)} | Days: {self.days_back} | Min Score: {self.min_score:.1f} "
        if self.search_query:
            stats += f"| Filter: '{self.search_query}' "

        self.stdscr.attron(curses.color_pair(self.COLOR_DIM))
        self.stdscr.addstr(1, 0, stats[:width-1].ljust(width-1))
        self.stdscr.attroff(curses.color_pair(self.COLOR_DIM))

    def _draw_footer(self):
        """Draw the footer/status bar."""
        height, width = self._get_dims()

        if self.view_mode == self.VIEW_LIST:
            keys = " [j/k] Nav | [Enter] Details | [f] Full Paper | [/] Search | [r] Refresh | [?] Help | [q] Quit "
        elif self.view_mode == self.VIEW_DETAIL:
            keys = " [j/k] Scroll | [Esc] Back | [f] Full Paper | [a] Abstract | [o] Open URL | [?] Help | [q] Quit "
        elif self.view_mode == self.VIEW_ABSTRACT:
            keys = " [j/k] Scroll | [Esc] Back | [f] Full Paper | [?] Help | [q] Quit "
        elif self.view_mode == self.VIEW_FULLPAPER:
            keys = " [j/k/PgUp/PgDn] Scroll | [g/G] Top/Bottom | [/] Search in paper | [Esc] Back | [q] Quit "
        else:
            keys = " [Enter/Esc/?] Close Help | [q] Quit "

        self.stdscr.attron(curses.color_pair(self.COLOR_HEADER))
        self.stdscr.addstr(height - 2, 0, "=" * (width - 1))
        self.stdscr.addstr(height - 1, 0, keys[:width-1].ljust(width-1))
        self.stdscr.attroff(curses.color_pair(self.COLOR_HEADER))

        # Status message
        if self.status_message:
            self.stdscr.attron(curses.color_pair(self.COLOR_HIGHLIGHT))
            msg = f" {self.status_message} "
            self.stdscr.addstr(height - 2, width - len(msg) - 2, msg)
            self.stdscr.attroff(curses.color_pair(self.COLOR_HIGHLIGHT))

    def _draw_paper_list(self):
        """Draw the paper list view."""
        height, width = self._get_dims()
        list_height = height - 5  # Account for header and footer
        start_y = 2

        if not self.filtered_papers:
            self.stdscr.addstr(start_y + 2, 2, "No papers found. Press 'r' to refresh or adjust filters.")
            return

        # Adjust scroll offset to keep current item visible
        if self.current_idx < self.scroll_offset:
            self.scroll_offset = self.current_idx
        elif self.current_idx >= self.scroll_offset + list_height:
            self.scroll_offset = self.current_idx - list_height + 1

        for i in range(list_height):
            paper_idx = self.scroll_offset + i
            if paper_idx >= len(self.filtered_papers):
                break

            paper = self.filtered_papers[paper_idx]
            y = start_y + i

            is_selected = paper_idx == self.current_idx

            # Build the display line
            score_str = f"[{paper.relevance_score:4.1f}]"
            date_str = paper.published.strftime("%m/%d")
            tags_str = " ".join(paper.relevance_tags[:3]) if paper.relevance_tags else ""

            # Calculate available width for title
            prefix_len = len(score_str) + len(date_str) + 4  # spaces
            tags_len = len(tags_str) + 2 if tags_str else 0
            title_width = width - prefix_len - tags_len - 2

            title = paper.title
            if len(title) > title_width:
                title = title[:title_width - 3] + "..."

            # Draw the line
            if is_selected:
                self.stdscr.attron(curses.color_pair(self.COLOR_SELECTED))
                self.stdscr.addstr(y, 0, " " * (width - 1))

            # Score
            if is_selected:
                self.stdscr.addstr(y, 0, score_str)
            else:
                self.stdscr.attron(curses.color_pair(self.COLOR_SCORE))
                self.stdscr.addstr(y, 0, score_str)
                self.stdscr.attroff(curses.color_pair(self.COLOR_SCORE))

            # Date
            x = len(score_str) + 1
            if is_selected:
                self.stdscr.addstr(y, x, date_str)
            else:
                self.stdscr.attron(curses.color_pair(self.COLOR_DIM))
                self.stdscr.addstr(y, x, date_str)
                self.stdscr.attroff(curses.color_pair(self.COLOR_DIM))

            # Title
            x += len(date_str) + 1
            self.stdscr.addstr(y, x, title)

            # Tags (if room)
            if tags_str and not is_selected:
                x = width - len(tags_str) - 2
                self.stdscr.attron(curses.color_pair(self.COLOR_TAG))
                self.stdscr.addstr(y, x, tags_str)
                self.stdscr.attroff(curses.color_pair(self.COLOR_TAG))

            if is_selected:
                self.stdscr.attroff(curses.color_pair(self.COLOR_SELECTED))

        # Draw scrollbar if needed
        if len(self.filtered_papers) > list_height:
            self._draw_scrollbar(start_y, list_height, len(self.filtered_papers), self.scroll_offset)

    def _draw_scrollbar(self, start_y: int, view_height: int, total_items: int, offset: int):
        """Draw a scrollbar."""
        height, width = self._get_dims()
        x = width - 1

        # Calculate scrollbar position and size
        if total_items <= view_height:
            return

        bar_height = max(1, int(view_height * view_height / total_items))
        bar_pos = int(offset * (view_height - bar_height) / (total_items - view_height))

        for i in range(view_height):
            y = start_y + i
            if bar_pos <= i < bar_pos + bar_height:
                try:
                    self.stdscr.addch(y, x, curses.ACS_CKBOARD)
                except curses.error:
                    pass
            else:
                try:
                    self.stdscr.addch(y, x, curses.ACS_VLINE)
                except curses.error:
                    pass

    def _wrap_text(self, text: str, width: int) -> list[str]:
        """Wrap text to specified width."""
        lines = []
        for paragraph in text.split('\n'):
            if paragraph.strip():
                lines.extend(textwrap.wrap(paragraph, width))
            else:
                lines.append("")
        return lines

    def _draw_paper_detail(self):
        """Draw detailed view of selected paper."""
        if not self.filtered_papers:
            return

        height, width = self._get_dims()
        paper = self.filtered_papers[self.current_idx]

        content_width = width - 4
        content_height = height - 5
        start_y = 2

        lines = []

        # Title
        lines.append(("TITLE", self.COLOR_HEADER))
        for line in self._wrap_text(paper.title, content_width):
            lines.append((line, self.COLOR_HIGHLIGHT))
        lines.append(("", 0))

        # Meta info
        lines.append(("INFO", self.COLOR_HEADER))
        lines.append((f"ArXiv ID: {paper.arxiv_id}", self.COLOR_NORMAL))
        lines.append((f"Published: {paper.published.strftime('%Y-%m-%d %H:%M')}", self.COLOR_NORMAL))
        lines.append((f"Categories: {', '.join(paper.categories)}", self.COLOR_NORMAL))
        lines.append((f"Relevance: {paper.relevance_score:.1f} - {', '.join(paper.relevance_tags)}", self.COLOR_SCORE))
        lines.append(("", 0))

        # Authors
        lines.append(("AUTHORS", self.COLOR_HEADER))
        authors_text = ", ".join(paper.authors)
        for line in self._wrap_text(authors_text, content_width):
            lines.append((line, self.COLOR_NORMAL))
        lines.append(("", 0))

        # Summary or Abstract
        if self.show_summary:
            lines.append(("SUMMARY (press 'a' for full abstract)", self.COLOR_HEADER))
            summary = self.summarizer.summarize(paper)
        else:
            lines.append(("ABSTRACT", self.COLOR_HEADER))
            summary = paper.abstract

        for line in self._wrap_text(summary, content_width):
            lines.append((line, self.COLOR_NORMAL))
        lines.append(("", 0))

        # Links
        lines.append(("LINKS (press 'o' to open)", self.COLOR_HEADER))
        lines.append((f"PDF: {paper.pdf_url}", self.COLOR_TAG))
        lines.append((f"Abstract: {paper.abs_url}", self.COLOR_TAG))

        # Handle scroll
        max_scroll = max(0, len(lines) - content_height)
        self.detail_scroll = max(0, min(self.detail_scroll, max_scroll))

        # Draw lines
        for i in range(content_height):
            line_idx = self.detail_scroll + i
            if line_idx >= len(lines):
                break

            text, color = lines[line_idx]
            y = start_y + i

            if color:
                self.stdscr.attron(curses.color_pair(color))

            try:
                self.stdscr.addstr(y, 2, text[:content_width])
            except curses.error:
                pass

            if color:
                self.stdscr.attroff(curses.color_pair(color))

        # Draw scrollbar
        if len(lines) > content_height:
            self._draw_scrollbar(start_y, content_height, len(lines), self.detail_scroll)

    def _draw_abstract_view(self):
        """Draw full abstract view."""
        if not self.filtered_papers:
            return

        height, width = self._get_dims()
        paper = self.filtered_papers[self.current_idx]

        content_width = width - 4
        content_height = height - 5
        start_y = 2

        lines = []

        # Title
        lines.append(("FULL ABSTRACT", self.COLOR_HEADER))
        lines.append(("", 0))

        for line in self._wrap_text(paper.abstract, content_width):
            lines.append((line, self.COLOR_NORMAL))

        # Handle scroll
        max_scroll = max(0, len(lines) - content_height)
        self.detail_scroll = max(0, min(self.detail_scroll, max_scroll))

        # Draw lines
        for i in range(content_height):
            line_idx = self.detail_scroll + i
            if line_idx >= len(lines):
                break

            text, color = lines[line_idx]
            y = start_y + i

            if color:
                self.stdscr.attron(curses.color_pair(color))

            try:
                self.stdscr.addstr(y, 2, text[:content_width])
            except curses.error:
                pass

            if color:
                self.stdscr.attroff(curses.color_pair(color))

        if len(lines) > content_height:
            self._draw_scrollbar(start_y, content_height, len(lines), self.detail_scroll)

    def _draw_full_paper(self):
        """Draw full paper reading view."""
        height, width = self._get_dims()
        content_width = width - 4
        content_height = height - 5
        start_y = 2

        if not self.full_paper_wrapped:
            self.stdscr.addstr(start_y + 2, 2, "No paper loaded. Press 'f' on a paper to load it.")
            return

        # Handle scroll
        max_scroll = max(0, len(self.full_paper_wrapped) - content_height)
        self.detail_scroll = max(0, min(self.detail_scroll, max_scroll))

        # Draw lines
        for i in range(content_height):
            line_idx = self.detail_scroll + i
            if line_idx >= len(self.full_paper_wrapped):
                break

            text, color = self.full_paper_wrapped[line_idx]
            y = start_y + i

            if color:
                self.stdscr.attron(curses.color_pair(color))

            try:
                self.stdscr.addstr(y, 2, text[:content_width])
            except curses.error:
                pass

            if color:
                self.stdscr.attroff(curses.color_pair(color))

        # Draw scrollbar
        if len(self.full_paper_wrapped) > content_height:
            self._draw_scrollbar(start_y, content_height, len(self.full_paper_wrapped), self.detail_scroll)

        # Draw position indicator
        if self.full_paper_wrapped:
            percent = int(100 * (self.detail_scroll + content_height) / len(self.full_paper_wrapped))
            percent = min(100, percent)
            pos_str = f" {percent}% ({self.detail_scroll + 1}-{min(self.detail_scroll + content_height, len(self.full_paper_wrapped))}/{len(self.full_paper_wrapped)}) "
            self.stdscr.attron(curses.color_pair(self.COLOR_DIM))
            try:
                self.stdscr.addstr(1, width - len(pos_str) - 2, pos_str)
            except curses.error:
                pass
            self.stdscr.attroff(curses.color_pair(self.COLOR_DIM))

    def _fetch_full_paper(self):
        """Fetch and prepare full paper for viewing."""
        if not self.filtered_papers:
            return

        paper = self.filtered_papers[self.current_idx]

        def progress(msg):
            self._draw_loading(msg)

        success, lines = self.paper_fetcher.fetch_full_text(paper, progress_callback=progress)

        self.full_paper_lines = lines

        # Wrap lines for display
        height, width = self._get_dims()
        content_width = width - 6

        self.full_paper_wrapped = []
        for line in lines:
            # Determine color based on content
            color = self.COLOR_NORMAL
            if line.startswith("=="):
                color = self.COLOR_HEADER
            elif line.startswith("##"):
                color = self.COLOR_HIGHLIGHT
            elif line.startswith("--"):
                color = self.COLOR_DIM
            elif line.startswith("Authors:") or line.startswith("ArXiv ID:") or line.startswith("Categories:"):
                color = self.COLOR_TAG
            elif "[Figure]" in line or "[Table]" in line or "[math]" in line:
                color = self.COLOR_DIM

            # Wrap long lines
            if len(line) <= content_width:
                self.full_paper_wrapped.append((line, color))
            else:
                wrapped = textwrap.wrap(line, content_width)
                for wline in wrapped:
                    self.full_paper_wrapped.append((wline, color))

        self.detail_scroll = 0
        self.view_mode = self.VIEW_FULLPAPER

        if success:
            self.status_message = f"Loaded {len(self.full_paper_wrapped)} lines"
        else:
            self.status_message = "Paper HTML not available"

    def _draw_help(self):
        """Draw help screen."""
        height, width = self._get_dims()
        start_y = 2

        help_text = """
CURSEARXIV - Help

NAVIGATION
  j / Down      Move down in list or scroll
  k / Up        Move up in list or scroll
  g / Home      Go to first item / top of paper
  G / End       Go to last item / bottom of paper
  PgUp/PgDn     Page up/down

VIEWS
  Enter         Open detail view from list
  a             Show full abstract (detail view)
  f             Read FULL PAPER text (fetches HTML)
  Esc           Go back to previous view

FULL PAPER VIEW
  f             Load full paper from ArXiv HTML
  j/k           Scroll line by line
  PgUp/PgDn     Scroll by page
  g/G           Jump to top/bottom
  /             Search within paper text
  n             Jump to next search match
  Esc           Return to detail view

ACTIONS
  r             Refresh papers from ArXiv
  o             Open paper URL in browser

FILTERING
  /             Search/filter papers by text
  d             Change number of days to fetch
  s             Set minimum relevance score
  c             Clear all filters

SETTINGS
  1-7           Quick set days (1-7 days)
  +/-           Increase/decrease min score

OTHER
  ?             Show this help
  q             Quit

ABOUT
  This tool fetches recent papers from ArXiv in AI, ML,
  Signal Processing, and related categories. Papers are
  scored based on relevance to wireless physical layer
  optimization and machine learning topics.

  The 'f' key fetches the full paper text from ArXiv's
  HTML view (available for most recent papers with
  LaTeX source).

  Categories: cs.AI, cs.LG, cs.IT, cs.NI, eess.SP,
              eess.SY, stat.ML
"""

        lines = help_text.strip().split('\n')
        content_height = height - 5

        max_scroll = max(0, len(lines) - content_height)
        self.detail_scroll = max(0, min(self.detail_scroll, max_scroll))

        for i in range(content_height):
            line_idx = self.detail_scroll + i
            if line_idx >= len(lines):
                break

            line = lines[line_idx]
            y = start_y + i

            # Highlight headers
            if line and line[0].isupper() and line == line.upper():
                self.stdscr.attron(curses.color_pair(self.COLOR_HEADER) | curses.A_BOLD)
                try:
                    self.stdscr.addstr(y, 2, line[:width-4])
                except curses.error:
                    pass
                self.stdscr.attroff(curses.color_pair(self.COLOR_HEADER) | curses.A_BOLD)
            else:
                try:
                    self.stdscr.addstr(y, 2, line[:width-4])
                except curses.error:
                    pass

    def _draw_loading(self, message: str):
        """Draw loading overlay."""
        height, width = self._get_dims()

        box_width = max(len(message) + 4, 30)
        box_height = 3
        start_x = (width - box_width) // 2
        start_y = (height - box_height) // 2

        # Draw box
        self.stdscr.attron(curses.color_pair(self.COLOR_HEADER))
        for i in range(box_height):
            self.stdscr.addstr(start_y + i, start_x, " " * box_width)

        # Draw message
        msg_x = start_x + (box_width - len(message)) // 2
        self.stdscr.addstr(start_y + 1, msg_x, message)
        self.stdscr.attroff(curses.color_pair(self.COLOR_HEADER))

        self.stdscr.refresh()

    def _get_input(self, prompt: str) -> str:
        """Get text input from user."""
        height, width = self._get_dims()

        # Draw input box
        box_width = min(60, width - 4)
        box_y = height - 4
        box_x = 2

        self.stdscr.attron(curses.color_pair(self.COLOR_HEADER))
        self.stdscr.addstr(box_y, box_x, prompt + " " * (box_width - len(prompt)))
        self.stdscr.attroff(curses.color_pair(self.COLOR_HEADER))

        curses.curs_set(1)
        curses.echo()

        self.stdscr.addstr(box_y, box_x + len(prompt), "")

        try:
            user_input = self.stdscr.getstr(box_y, box_x + len(prompt), box_width - len(prompt) - 1)
            result = user_input.decode('utf-8').strip()
        except Exception:
            result = ""

        curses.noecho()
        curses.curs_set(0)

        return result

    def _apply_filter(self):
        """Apply current search filter to papers."""
        if not self.search_query:
            self.filtered_papers = [p for p in self.papers if p.relevance_score >= self.min_score]
        else:
            query = self.search_query.lower()
            self.filtered_papers = [
                p for p in self.papers
                if (query in p.title.lower() or
                    query in p.abstract.lower() or
                    any(query in a.lower() for a in p.authors) or
                    any(query in c.lower() for c in p.categories))
                and p.relevance_score >= self.min_score
            ]

        self.current_idx = 0
        self.scroll_offset = 0

    def _refresh_papers(self):
        """Fetch fresh papers from ArXiv."""
        self.loading = True
        self.fetcher.days_back = self.days_back

        def progress(msg):
            self._draw_loading(msg)

        self.papers = self.fetcher.fetch_papers(progress_callback=progress)
        self._apply_filter()

        self.loading = False
        self.status_message = f"Loaded {len(self.papers)} papers"

    def _open_url(self):
        """Open current paper's URL."""
        if not self.filtered_papers:
            return

        paper = self.filtered_papers[self.current_idx]
        import subprocess
        import sys

        try:
            if sys.platform == "darwin":
                subprocess.run(["open", paper.abs_url], check=False)
            elif sys.platform == "win32":
                subprocess.run(["start", paper.abs_url], shell=True, check=False)
            else:
                subprocess.run(["xdg-open", paper.abs_url], check=False)
            self.status_message = "Opened in browser"
        except Exception as e:
            self.status_message = f"Failed to open: {e}"

    def draw(self):
        """Main draw function."""
        self.stdscr.clear()

        self._draw_header()

        if self.view_mode == self.VIEW_LIST:
            self._draw_paper_list()
        elif self.view_mode == self.VIEW_DETAIL:
            self._draw_paper_detail()
        elif self.view_mode == self.VIEW_ABSTRACT:
            self._draw_abstract_view()
        elif self.view_mode == self.VIEW_HELP:
            self._draw_help()
        elif self.view_mode == self.VIEW_FULLPAPER:
            self._draw_full_paper()

        self._draw_footer()

        self.stdscr.refresh()

    def handle_input(self, key: int) -> bool:
        """Handle keyboard input. Returns False to quit."""

        # Clear status message on any key
        if key != -1:
            self.status_message = ""

        # Global keys
        if key == ord('q'):
            return False

        if key == ord('?'):
            if self.view_mode == self.VIEW_HELP:
                self.view_mode = self.VIEW_LIST
            else:
                self.view_mode = self.VIEW_HELP
                self.detail_scroll = 0
            return True

        # View-specific handling
        if self.view_mode == self.VIEW_LIST:
            return self._handle_list_input(key)
        elif self.view_mode == self.VIEW_DETAIL:
            return self._handle_detail_input(key)
        elif self.view_mode == self.VIEW_ABSTRACT:
            return self._handle_abstract_input(key)
        elif self.view_mode == self.VIEW_HELP:
            return self._handle_help_input(key)
        elif self.view_mode == self.VIEW_FULLPAPER:
            return self._handle_fullpaper_input(key)

        return True

    def _handle_list_input(self, key: int) -> bool:
        """Handle input in list view."""
        if key in (ord('j'), curses.KEY_DOWN):
            if self.current_idx < len(self.filtered_papers) - 1:
                self.current_idx += 1

        elif key in (ord('k'), curses.KEY_UP):
            if self.current_idx > 0:
                self.current_idx -= 1

        elif key in (ord('g'), curses.KEY_HOME):
            self.current_idx = 0
            self.scroll_offset = 0

        elif key in (ord('G'), curses.KEY_END):
            self.current_idx = max(0, len(self.filtered_papers) - 1)

        elif key == curses.KEY_PPAGE:
            height, _ = self._get_dims()
            page_size = height - 5
            self.current_idx = max(0, self.current_idx - page_size)

        elif key == curses.KEY_NPAGE:
            height, _ = self._get_dims()
            page_size = height - 5
            self.current_idx = min(len(self.filtered_papers) - 1, self.current_idx + page_size)

        elif key in (ord('\n'), curses.KEY_ENTER, 10):
            if self.filtered_papers:
                self.view_mode = self.VIEW_DETAIL
                self.detail_scroll = 0
                self.show_summary = True

        elif key == ord('r'):
            self._refresh_papers()

        elif key == ord('/'):
            query = self._get_input("Search: ")
            self.search_query = query
            self._apply_filter()

        elif key == ord('c'):
            self.search_query = ""
            self.min_score = 0.0
            self._apply_filter()
            self.status_message = "Filters cleared"

        elif key == ord('d'):
            days_str = self._get_input("Days back (1-30): ")
            try:
                days = int(days_str)
                if 1 <= days <= 30:
                    self.days_back = days
                    self._refresh_papers()
            except ValueError:
                self.status_message = "Invalid number"

        elif key == ord('s'):
            score_str = self._get_input("Minimum score: ")
            try:
                score = float(score_str)
                if score >= 0:
                    self.min_score = score
                    self._apply_filter()
            except ValueError:
                self.status_message = "Invalid number"

        elif ord('1') <= key <= ord('7'):
            self.days_back = key - ord('0')
            self._refresh_papers()

        elif key == ord('+') or key == ord('='):
            self.min_score += 1.0
            self._apply_filter()

        elif key == ord('-'):
            self.min_score = max(0, self.min_score - 1.0)
            self._apply_filter()

        elif key == ord('o'):
            self._open_url()

        elif key == ord('f'):
            self._fetch_full_paper()

        return True

    def _handle_detail_input(self, key: int) -> bool:
        """Handle input in detail view."""
        if key in (ord('j'), curses.KEY_DOWN):
            self.detail_scroll += 1

        elif key in (ord('k'), curses.KEY_UP):
            self.detail_scroll = max(0, self.detail_scroll - 1)

        elif key == curses.KEY_PPAGE:
            height, _ = self._get_dims()
            self.detail_scroll = max(0, self.detail_scroll - (height - 5))

        elif key == curses.KEY_NPAGE:
            height, _ = self._get_dims()
            self.detail_scroll += height - 5

        elif key in (ord('\n'), curses.KEY_ENTER, 10, 27):  # Enter or Escape
            self.view_mode = self.VIEW_LIST

        elif key == ord('a'):
            self.view_mode = self.VIEW_ABSTRACT
            self.detail_scroll = 0

        elif key == ord('o'):
            self._open_url()

        elif key == ord('f'):
            self._fetch_full_paper()

        return True

    def _handle_abstract_input(self, key: int) -> bool:
        """Handle input in abstract view."""
        if key in (ord('j'), curses.KEY_DOWN):
            self.detail_scroll += 1

        elif key in (ord('k'), curses.KEY_UP):
            self.detail_scroll = max(0, self.detail_scroll - 1)

        elif key == curses.KEY_PPAGE:
            height, _ = self._get_dims()
            self.detail_scroll = max(0, self.detail_scroll - (height - 5))

        elif key == curses.KEY_NPAGE:
            height, _ = self._get_dims()
            self.detail_scroll += height - 5

        elif key in (ord('\n'), curses.KEY_ENTER, 10, 27):
            self.view_mode = self.VIEW_DETAIL
            self.detail_scroll = 0

        elif key == ord('f'):
            self._fetch_full_paper()

        return True

    def _handle_fullpaper_input(self, key: int) -> bool:
        """Handle input in full paper view."""
        height, _ = self._get_dims()
        page_size = height - 5

        if key in (ord('j'), curses.KEY_DOWN):
            self.detail_scroll += 1

        elif key in (ord('k'), curses.KEY_UP):
            self.detail_scroll = max(0, self.detail_scroll - 1)

        elif key == curses.KEY_PPAGE:
            self.detail_scroll = max(0, self.detail_scroll - page_size)

        elif key == curses.KEY_NPAGE:
            max_scroll = max(0, len(self.full_paper_wrapped) - page_size)
            self.detail_scroll = min(self.detail_scroll + page_size, max_scroll)

        elif key in (ord('g'), curses.KEY_HOME):
            self.detail_scroll = 0

        elif key in (ord('G'), curses.KEY_END):
            self.detail_scroll = max(0, len(self.full_paper_wrapped) - page_size)

        elif key == 27:  # Escape
            self.view_mode = self.VIEW_DETAIL
            self.detail_scroll = 0

        elif key == ord('/'):
            # Search within paper
            query = self._get_input("Search in paper: ")
            if query:
                self._search_in_paper(query)

        elif key == ord('n'):
            # Find next occurrence
            self._search_next_in_paper()

        elif key == ord('o'):
            self._open_url()

        return True

    def _search_in_paper(self, query: str):
        """Search for text in full paper."""
        if not self.full_paper_wrapped or not query:
            return

        self._paper_search_query = query.lower()
        self._paper_search_matches = []

        # Find all matches
        for i, (line, _) in enumerate(self.full_paper_wrapped):
            if self._paper_search_query in line.lower():
                self._paper_search_matches.append(i)

        if self._paper_search_matches:
            self._paper_search_idx = 0
            self.detail_scroll = self._paper_search_matches[0]
            self.status_message = f"Found {len(self._paper_search_matches)} matches (press 'n' for next)"
        else:
            self.status_message = f"No matches for '{query}'"

    def _search_next_in_paper(self):
        """Jump to next search match in paper."""
        if not hasattr(self, '_paper_search_matches') or not self._paper_search_matches:
            self.status_message = "No active search"
            return

        self._paper_search_idx = (self._paper_search_idx + 1) % len(self._paper_search_matches)
        self.detail_scroll = self._paper_search_matches[self._paper_search_idx]
        self.status_message = f"Match {self._paper_search_idx + 1}/{len(self._paper_search_matches)}"

    def _handle_help_input(self, key: int) -> bool:
        """Handle input in help view."""
        if key in (ord('j'), curses.KEY_DOWN):
            self.detail_scroll += 1

        elif key in (ord('k'), curses.KEY_UP):
            self.detail_scroll = max(0, self.detail_scroll - 1)

        elif key in (ord('\n'), curses.KEY_ENTER, 10, 27):
            self.view_mode = self.VIEW_LIST
            self.detail_scroll = 0

        return True

    def run(self):
        """Main run loop."""
        # Initial fetch
        self._refresh_papers()

        running = True
        while running:
            self.draw()

            try:
                key = self.stdscr.getch()
                running = self.handle_input(key)
            except KeyboardInterrupt:
                running = False


def main(stdscr):
    """Main entry point."""
    ui = CurseArXivUI(stdscr)
    ui.run()


if __name__ == "__main__":
    curses.wrapper(main)
