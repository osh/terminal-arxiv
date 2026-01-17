# TArXiv - Terminal ArXiv Browser

A modern terminal-based ArXiv paper browser built with [Textual](https://textual.textualize.io/), designed for researchers working on **wireless communications**, **physical layer optimization**, and **machine learning** for wireless systems.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TArXiv - Terminal ArXiv Browser                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ [ 22.5] 01/15 Achievable DoF Analysis in Massive MIMO...    W:mimo ML:deep  │
│ [  9.0] 01/14 Deep Learning for Channel Estimation...       W:5g ML:neural  │
│ [  7.5] 01/14 RIS-Aided Beamforming Optimization...         W:ris W:beam    │
│ [  6.0] 01/13 Federated Learning in 6G Networks...          W:6g ML:fed     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Intelligent Paper Discovery
- Fetches recent papers from 7 ArXiv categories relevant to wireless/ML research
- Scores papers based on 80+ keywords across wireless and ML domains
- Papers combining both wireless AND ML topics receive 1.5x relevance bonus
- Configurable time window (1-30 days) and minimum score threshold

### Modern Terminal UI
- Built with Textual for a rich, responsive terminal experience
- **List View** - Browse papers sorted by relevance with scores, dates, and topic tags
- **Detail View** - See title, authors, abstract, categories, and links
- **Full Paper View** - Read complete papers with Markdown rendering (fetched from ArXiv HTML)
- Modal dialogs for search and settings
- Background loading with progress indicators

### Keyboard-Driven Workflow
Vim-style navigation throughout:
- `j/k` or arrow keys - Navigate
- `Enter` - Open detail view
- `f` - Load full paper text
- `o` - Open in browser
- `/` - Search papers
- `Escape` - Go back

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tarxiv.git
cd tarxiv

# Install dependencies
pip install -r requirements.txt

# Run
python tarxiv.py
```

### Requirements
- Python 3.9+
- textual >= 0.40.0

## Usage

### Quick Start

```bash
python tarxiv.py
```

The app will immediately fetch recent papers and display the most relevant ones.

### Keyboard Controls

| Key | Action |
|-----|--------|
| `j` / `↓` | Move down |
| `k` / `↑` | Move up |
| `Enter` | Open detail view / select |
| `Escape` | Go back |
| `f` | Load full paper text |
| `o` | Open paper in browser |
| `/` | Search papers |
| `c` | Clear search filter |
| `s` | Settings (days, min score) |
| `r` | Refresh papers from ArXiv |
| `?` | Show help |
| `q` | Quit |

## How It Works

### Paper Fetching

TArXiv uses the [ArXiv API](https://info.arxiv.org/help/api/index.html) to fetch recent papers from these categories:

| Category | Description |
|----------|-------------|
| `cs.AI` | Artificial Intelligence |
| `cs.LG` | Machine Learning |
| `cs.IT` | Information Theory |
| `cs.NI` | Networking and Internet Architecture |
| `eess.SP` | Signal Processing |
| `eess.SY` | Systems and Control |
| `stat.ML` | Machine Learning (Statistics) |

### Relevance Scoring

Papers are scored based on keyword matching in titles and abstracts:

**Wireless/PHY Keywords (examples):**
- Core: `5G`, `6G`, `LTE`, `WiFi`, `OFDM`, `MIMO`, `beamforming`
- Advanced: `massive MIMO`, `RIS`, `IRS`, `mmWave`, `terahertz`
- PHY Layer: `channel estimation`, `CSI`, `precoding`, `equalization`
- Sensing: `radar`, `ISAC`, `localization`, `positioning`

**ML Keywords (examples):**
- Core: `deep learning`, `neural network`, `reinforcement learning`
- Architectures: `transformer`, `CNN`, `GNN`, `autoencoder`
- Methods: `federated learning`, `end-to-end`, `data-driven`

**Scoring Rules:**
- Each keyword has a weight (1-4 points)
- Papers with BOTH wireless AND ML keywords get a **1.5x multiplier**
- Pure ML papers without wireless context get 0.5x (lower priority)

### Full Paper Reading

Press `f` to fetch the full paper from ArXiv's HTML view (`arxiv.org/html/PAPER_ID`). This works for most papers submitted with LaTeX source.

The HTML is parsed and rendered as Markdown:
- Section headers formatted properly
- Math equations shown in code blocks
- Figures and tables marked appropriately
- Clean paragraph formatting

## Configuration

Currently, configuration is done by modifying the source code. Key areas to customize:

### Change Categories
Edit `ArXivFetcher.CATEGORIES`:

```python
CATEGORIES = [
    "cs.AI",
    "cs.LG",
    # Add more categories here
]
```

### Add Keywords
Edit `ArXivFetcher.WIRELESS_KEYWORDS` or `ArXivFetcher.ML_KEYWORDS`:

```python
WIRELESS_KEYWORDS = {
    "your_keyword": 3,  # keyword: weight
    # ...
}
```

## Why TArXiv?

If you're a researcher in wireless communications, signal processing, or ML for wireless systems, you know the challenge: **hundreds of papers are published weekly** across multiple ArXiv categories. Finding the ones relevant to your specific interests—like physical layer optimization for 5G/6G, MIMO systems, or ML-based channel estimation—requires manually checking multiple categories and reading countless abstracts.

TArXiv solves this by:
- **Automated relevance scoring** - Papers are ranked based on your research interests
- **Cross-category search** - Searches multiple relevant categories simultaneously
- **Smart filtering** - Papers combining wireless + ML topics get bonus scores
- **Terminal-native** - Read papers without leaving your terminal

## Tech Stack

- **[Textual](https://textual.textualize.io/)** - Modern TUI framework for Python
- **ArXiv API** - Paper metadata and search
- **ArXiv HTML** - Full paper content

## Contributing

Contributions are welcome! Some ideas:

- [ ] Add configuration file support (YAML/JSON)
- [ ] Bookmark/save favorite papers
- [ ] Export paper lists to BibTeX
- [ ] Add more keyword categories
- [ ] Cached paper storage for offline reading
- [ ] Custom color themes

## License

MIT License - feel free to use, modify, and distribute.

## Acknowledgments

- [ArXiv](https://arxiv.org/) for providing open access to scientific papers and their API
- [Textual](https://textual.textualize.io/) for the excellent TUI framework
- The wireless communications and ML research community

---

**Happy paper hunting!**
