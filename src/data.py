import re
import pandas as pd
import streamlit as st
from pybaseball import batting_stats_bref

SEASONS = list(range(2015, 2026))
MIN_PA = 200

def fix_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Fix literal escape sequences like \\xc3\\xad in player names."""
    def decode_escapes(s):
        if not isinstance(s, str):
            return s
        fixed = re.sub(
            r'\\x([0-9a-fA-F]{2})',
            lambda m: bytes.fromhex(m.group(1)).decode('latin-1'),
            s
        )
        try:
            return fixed.encode('latin-1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            return fixed
    df['Name'] = df['Name'].apply(decode_escapes)
    return df

@st.cache_data
def load_positions(season: int) -> pd.DataFrame:
    """Pull fielding stats to get primary position per player."""
    try:
        pos_df = fielding_stats_bref(season)
        pos_df = fix_encoding(pos_df)
        # Keep only the first (primary) position per player
        pos_df = pos_df.drop_duplicates(subset='Name', keep='first')[['Name', 'Pos']]
        return pos_df
    except Exception:
        return pd.DataFrame(columns=['Name', 'Pos'])

@st.cache_data
def load_season(season: int) -> pd.DataFrame:
    """Pull one season from Baseball Reference and compute derived stats."""
    df = batting_stats_bref(season)
    df = fix_encoding(df)
    df = df[df['Lev'].str.startswith('Maj')]
    df = df[df['PA'] >= MIN_PA]
    df = df.dropna(subset=['BA', 'OBP', 'SLG'])

    # Base derived stats
    df['K%']       = df['SO'] / df['PA']
    df['BB%']      = df['BB'] / df['PA']
    df['ISO']      = df['SLG'] - df['BA']
    df['HR_per_PA'] = df['HR'] / df['PA']
    df['CS_pct']   = df['CS'] / (df['SB'] + df['CS'] + 1e-9)

    df['Season'] = season
    df['Label']  = df['Name'] + f' ({season})'
    return df.reset_index(drop=True)

@st.cache_data
def load_all_seasons() -> pd.DataFrame:
    """Load all seasons and concatenate."""
    frames = []
    progress = st.progress(0, text="Loading seasons...")
    for i, season in enumerate(SEASONS):
        frames.append(load_season(season))
        progress.progress((i + 1) / len(SEASONS), text=f"Loading {season}...")
    progress.empty()
    return pd.concat(frames, ignore_index=True)

def get_headshot_url(mlb_id) -> str | None:
    if pd.isna(mlb_id):
        return None
    return (
        f"https://img.mlbstatic.com/mlb-photos/image/upload/"
        f"d_people:generic:headshot:67:current.png/"
        f"w_213,q_auto:best/v1/people/{int(mlb_id)}/headshot/67/current"
    )

def get_player_mlbid(name: str, season: int, df: pd.DataFrame):
    row = df[(df['Name'] == name) & (df['Season'] == season)]
    if row.empty or pd.isna(row.iloc[0]['mlbID']):
        return None
    return int(row.iloc[0]['mlbID'])