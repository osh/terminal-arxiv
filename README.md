# CurseArXiv

A terminal-based ArXiv paper browser built with Python ncurses, designed for researchers working on **wireless communications**, **physical layer optimization**, and **machine learning** for wireless systems.

```
╔════════════════════════════════════════════════════════════════════════╗
║              CurseArXiv - Wireless/ML Paper Browser                    ║
╠════════════════════════════════════════════════════════════════════════╣
║ [22.5] 01/15 Achievable DoF Analysis in Massive MIMO...    W:mimo ML:  ║
║ [ 9.0] 01/14 Deep Learning for Channel Estimation...       W:5g ML:dee ║
║ [ 7.5] 01/14 RIS-Aided Beamforming Optimization...         W:ris W:bea ║
║ [ 6.0] 01/13 Federated Learning in 6G Networks...          W:6g ML:fed ║
╚════════════════════════════════════════════════════════════════════════╝
```

## Why CurseArXiv?

If you're a researcher in wireless communications, signal processing, or ML for wireless systems, you know the challenge: **hundreds of papers are published weekly** across multiple ArXiv categories. Finding the ones relevant to your specific interests—like physical layer optimization for 5G/6G, MIMO systems, or ML-based channel estimation—requires manually checking multiple categories and reading countless abstracts.

CurseArXiv solves this by:

- **Automated relevance scoring** - Papers are ranked based on keyword matching for wireless/PHY and ML topics
- **Cross-category search** - Searches cs.AI, cs.LG, cs.IT, cs.NI, eess.SP, eess.SY, and stat.ML simultaneously
- **Smart filtering** - Papers combining wireless + ML topics get bonus scores (the sweet spot!)
- **Full paper reading** - Read entire papers directly in your terminal via ArXiv's HTML view
- **Zero dependencies** - Pure Python standard library, works anywhere Python runs

## Features

### Intelligent Paper Discovery
- Fetches recent papers from 7 ArXiv categories relevant to wireless/ML research
- Scores papers based on 80+ keywords across wireless and ML domains
- Papers combining both wireless AND ML topics receive 1.5x relevance bonus
- Configurable time window (1-30 days) and minimum score threshold

### Terminal-Native Reading Experience
- **List View** - Browse papers sorted by relevance with scores, dates, and topic tags
- **Detail View** - See title, authors, abstract summary, categories, and links
- **Full Paper View** - Read the complete paper text directly in ncurses (fetched from ArXiv HTML)
- **Search** - Filter papers or search within full paper text

### Keyboard-Driven Workflow
Vim-style navigation throughout:
- `j/k` - Navigate/scroll
- `g/G` - Jump to top/bottom
- `/` - Search
- `f` - Load full paper text
- `o` - Open in browser

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cursearxiv.git
cd cursearxiv

# No dependencies to install! Just run:
python3 cursearxiv.py
```

### Requirements
- Python 3.9+
- No external packages required (uses only standard library)
- Works on macOS, Linux, and WSL

## Usage

### Quick Start

```bash
python3 cursearxiv.py
```

The tool will immediately fetch recent papers and display the most relevant ones.

### Keyboard Controls

#### List View
| Key | Action |
|-----|--------|
| `j` / `↓` | Move down |
| `k` / `↑` | Move up |
| `Enter` | Open detail view |
| `f` | Load full paper text |
| `/` | Search/filter papers |
| `r` | Refresh from ArXiv |
| `d` | Change days to fetch |
| `s` | Set minimum score |
| `1-7` | Quick set days (1-7) |
| `+/-` | Adjust min score |
| `c` | Clear filters |
| `?` | Help |
| `q` | Quit |

#### Detail View
| Key | Action |
|-----|--------|
| `j/k` | Scroll |
| `a` | Show full abstract |
| `f` | Load full paper |
| `o` | Open in browser |
| `Esc` | Back to list |

#### Full Paper View
| Key | Action |
|-----|--------|
| `j/k` | Scroll line by line |
| `PgUp/PgDn` | Scroll by page |
| `g/G` | Jump to top/bottom |
| `/` | Search in paper |
| `n` | Next search match |
| `o` | Open in browser |
| `Esc` | Back to detail view |

## How It Works

### Paper Fetching

CurseArXiv uses the [ArXiv API](https://info.arxiv.org/help/api/index.html) to fetch recent papers. It queries these categories:

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

When you press `f`, CurseArXiv fetches the paper from ArXiv's HTML view (`arxiv.org/html/PAPER_ID`). This works for most papers submitted with LaTeX source (the vast majority of recent papers).

The HTML is parsed and converted to readable text:
- Section headers are highlighted
- Math equations show LaTeX alt-text when available
- Figures and tables are marked with `[Figure]` and `[Table]`
- Clean paragraph formatting

## Examples

### Find papers on RIS + Deep Learning
```
1. Launch: python3 cursearxiv.py
2. Press / and type: ris
3. Papers mentioning RIS/IRS will be filtered
4. Top results will have both W:ris and ML: tags
```

### Read a paper on MIMO channel estimation
```
1. Navigate to a relevant paper with j/k
2. Press Enter to see details and abstract
3. Press f to load the full paper
4. Use / to search for "channel estimation"
5. Press n to jump between matches
```

### Adjust search parameters
```
1. Press d and enter 3 for last 3 days only
2. Press s and enter 5 to require score >= 5
3. Press + repeatedly to increase min score
4. Press c to clear all filters
```

## Configuration

Currently, configuration is done by modifying the source code. Key areas to customize:

### Change Categories
Edit `ArXivFetcher.CATEGORIES` to add/remove ArXiv categories:

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

### Defaults
Modify `CurseArXivUI.__init__()` to change default settings:

```python
self.days_back = 7      # Default time window
self.min_score = 0.0    # Default minimum score
```

## Troubleshooting

### "No papers found"
- Try increasing the time window with `d` (some days have fewer submissions)
- Lower the minimum score with `s` or `-`
- Check your internet connection

### "HTML version not available"
- Some older papers or those without LaTeX source don't have HTML versions
- Press `o` to open the PDF in your browser instead

### Display issues
- Ensure your terminal supports colors and is at least 80 columns wide
- Try resizing your terminal window

## Contributing

Contributions are welcome! Some ideas:

- [ ] Add configuration file support (YAML/JSON)
- [ ] Bookmark/save favorite papers
- [ ] Export paper lists to BibTeX
- [ ] Add more keyword categories (computer vision, NLP, etc.)
- [ ] Cached paper storage for offline reading
- [ ] Custom color schemes

## License

MIT License - feel free to use, modify, and distribute.

## Acknowledgments

- [ArXiv](https://arxiv.org/) for providing open access to scientific papers and their API
- The Python `curses` library for terminal UI capabilities
- The wireless communications and ML research community

---

**Happy paper hunting!** If you find CurseArXiv useful for your research, consider giving it a star.
