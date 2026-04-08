# ⚾ Baseball Player Similarity Tool

A Streamlit app that finds the most statistically similar MLB players using batting stats from 2015–2025.

Built as a learning project to explore baseball analytics and data science.

---

## Features

- **Single Season mode** — find the most similar players to any batter in a given season
- **Player vs Past Self** — track how a player's profile has changed across seasons
- **Historical Twin** — search across all 10 years to find the closest statistical match ever
- **Radar charts** — visual comparison of up to 4 players at once
- **Player headshots** — pulled live from MLB's image CDN

---

## How It Works

Stats are pulled from Baseball Reference via the `pybaseball` library. For each qualified batter (min. 200 PA), six features are computed:

| Feature | Description |
|---------|-------------|
| BA      | Batting average |
| OBP     | On-base percentage |
| SLG     | Slugging percentage |
| K%      | Strikeout rate |
| BB%     | Walk rate |
| ISO     | Isolated power (SLG − BA) |

Features are normalized using `StandardScaler` and similarity is measured with Euclidean distance.

---

## Project Structure

```
baseball-similarity/
├── app.py              # Streamlit entry point
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── data.py         # Data loading and headshot URLs
    ├── model.py        # Normalization and similarity scoring
    └── viz.py          # Radar chart
```

---

## Getting Started

**1. Clone the repo**

```bash
git clone https://github.com/yourusername/baseball-similarity.git
cd baseball-similarity
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the app**

```bash
streamlit run app.py
```

The first run will take a minute to pull 10 years of data. After that it's cached and instant.

---

## Example Results

**Most similar players to Shohei Ohtani (2025):**

| Rank | Player | Similarity |
|------|--------|------------|
| 1 | Yordan Alvarez (2025) | 100% |
| 2 | Aaron Judge (2024) | 91% |
| 3 | Ronald Acuña Jr. (2023) | 73% |

---

## Possible Extensions

- Add pitching stats and pitcher similarity
- Weight features by importance (e.g. OBP over BA)
- Include minor league and NCAA data for prospect comps
- Deploy to Streamlit Cloud

---

## Data Source

All stats via [Baseball Reference](https://www.baseball-reference.com/) through the [`pybaseball`](https://github.com/jldbc/pybaseball) library.

---

## Author

Built by Luis Gomez
