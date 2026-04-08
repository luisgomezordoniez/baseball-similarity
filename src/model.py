import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

FEATURES_DEFAULT = ['BA', 'OBP', 'SLG', 'K%', 'BB%', 'ISO']
FEATURES_CONTACT = ['BA', 'OBP', 'K%', 'BB%', 'SB', 'OPS']
FEATURES_POWER   = ['SLG', 'ISO', 'HR_per_PA', 'BB%', 'K%', 'OBP']
FEATURES_SPEED   = ['BA', 'OBP', 'K%', 'BB%', 'SB', 'CS_pct']

POSITION_FEATURES = {
    'C  — Catcher':           FEATURES_DEFAULT,
    '1B — First Base':        FEATURES_POWER,
    '2B — Second Base':       FEATURES_CONTACT,
    '3B — Third Base':        FEATURES_POWER,
    'SS — Shortstop':         FEATURES_CONTACT,
    'LF — Left Field':        FEATURES_DEFAULT,
    'CF — Center Field':      FEATURES_SPEED,
    'RF — Right Field':       FEATURES_DEFAULT,
    'DH — Designated Hitter': FEATURES_POWER,
    'All — No Position Bias': FEATURES_DEFAULT,
}

def get_features_for_position(pos_label: str) -> list:
    return POSITION_FEATURES.get(pos_label, FEATURES_DEFAULT)

def build_model_for_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    available = [f for f in features if f in df.columns]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[available])
    return pd.DataFrame(scaled, columns=available, index=df['Label'])

def find_similar(
    target_label: str,
    scaled_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    n: int = 5,
    season_only: bool = False,
    season: int = None,
) -> pd.DataFrame:
    if target_label not in scaled_df.index:
        return pd.DataFrame()

    if season_only and season is not None:
        season_labels = raw_df[raw_df['Season'] == season]['Label'].tolist()
        search_df = scaled_df[scaled_df.index.isin(season_labels)]
    else:
        search_df = scaled_df

    target = scaled_df.loc[target_label].values
    features = scaled_df.columns.tolist()

    distances = {
        label: euclidean(target, row.values)
        for label, row in search_df.iterrows()
        if label != target_label
    }

    if not distances:
        return pd.DataFrame()

    similar  = sorted(distances.items(), key=lambda x: x[1])[:n]
    min_d    = similar[0][1]
    max_d    = similar[-1][1]
    available = [f for f in features if f in raw_df.columns]

    results = []
    for rank, (label, dist) in enumerate(similar, 1):
        #similarity = round(100 - ((dist - min_d) / (max_d - min_d + 1e-9) * 40), 1)
        similarity = round(100 / (1 + dist), 1)
        row = raw_df[raw_df['Label'] == label]
        if row.empty:
            continue
        stats = row[available].iloc[0]
        entry = {'Rank': rank, 'Player': label, 'Similarity': f"{similarity}%"}
        for f in available:
            val = stats[f]
            if f in ['K%', 'BB%', 'CS_pct']:
                entry[f] = f"{round(val * 100, 1)}%"
            elif f == 'HR_per_PA':
                entry[f] = round(val, 4)
            else:
                entry[f] = round(val, 3)
        results.append(entry)

    return pd.DataFrame(results)    