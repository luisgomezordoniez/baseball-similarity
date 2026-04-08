import streamlit as st
from src.data import load_all_seasons, get_headshot_url, get_player_mlbid, SEASONS
from src.model import (build_model_for_features, find_similar,
                       POSITION_FEATURES, get_features_for_position, FEATURES_DEFAULT)
from src.viz import radar_chart

st.set_page_config(page_title="Baseball Similarity Tool", layout="wide")
st.title("⚾ Baseball Player Similarity Tool")
st.caption("2015–2025 · Min 200 PA · Stats via Baseball Reference")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

with st.spinner("Loading 10 years of data..."):
    all_df = load_all_seasons()

all_names = sorted(all_df['Name'].unique().tolist())
all_teams = sorted(all_df['Tm'].dropna().unique().tolist())

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Mode",
        ["Single Season", "Cross-Year: Player vs Past Self", "Cross-Year: Find Historical Twin"]
    )
    n_similar = st.slider("Similar players to show", 3, 10, 5)

    st.divider()
    st.subheader("Position Lens")
    position_label = st.selectbox(
        "Compare as if playing...",
        options=list(POSITION_FEATURES.keys()),
        index=list(POSITION_FEATURES.keys()).index('All — No Position Bias'),
        help="Changes which stats are weighted in the similarity model. Try a different position to see how a player's profile fits elsewhere."
    )
    features = get_features_for_position(position_label)
    st.caption(f"Using: {', '.join(features)}")

    st.divider()
    st.subheader("Filters")
    team_filter = st.multiselect(
        "Filter by team",
        options=all_teams,
        default=[],
        placeholder="All teams"
    )

# ── HELPERS ───────────────────────────────────────────────────────────────────

def apply_team_filter(df, teams):
    if not teams:
        return df
    return df[df['Tm'].isin(teams)]

def show_player_card(name: str, season: int, df):
    mlb_id = get_player_mlbid(name, season, df)
    url    = get_headshot_url(mlb_id)
    if url:
        st.image(url, width=120)
    row = df[(df['Name'] == name) & (df['Season'] == season)]
    tm  = row.iloc[0].get('Tm', '') if not row.empty else ''
    st.markdown(f"**{name}** · {season}")
    st.caption(f"{tm} · {position_label.split(' — ')[0]}")

def show_comp_headshots(results, all_df):
    top  = results.head(4)
    cols = st.columns(len(top))
    for col, (_, row) in zip(cols, top.iterrows()):
        with col:
            label  = row['Player']
            name   = label.rsplit(' (', 1)[0]
            season = int(label.rsplit(' (', 1)[1].rstrip(')'))
            mlb_id = get_player_mlbid(name, season, all_df)
            url    = get_headshot_url(mlb_id)
            if url:
                st.image(url, width=90)
            st.caption(f"{name}\n{season} · {row['Similarity']}")

# ── SINGLE SEASON ─────────────────────────────────────────────────────────────

if mode == "Single Season":
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Search")
        season       = st.selectbox("Season", SEASONS[::-1])
        season_df    = apply_team_filter(all_df[all_df['Season'] == season], team_filter)
        season_names = sorted(season_df['Name'].unique().tolist())

        if not season_names:
            st.warning("No players match the selected team filter.")
        else:
            player       = st.selectbox("Player", season_names)
            target_label = f"{player} ({season})"

            # Build model scoped to this season with selected position features
            available    = [f for f in features if f in all_df.columns]
            season_scaled = build_model_for_features(
                all_df[all_df['Season'] == season], available
            )

            st.divider()
            show_player_card(player, season, all_df)

            results = find_similar(
                target_label, season_scaled, all_df,
                n=n_similar, season_only=True, season=season
            )

            if not results.empty:
                st.divider()
                st.markdown("**Most Similar Players**")
                show_comp_headshots(results, all_df)
                st.dataframe(results, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Radar Chart")
        if 'results' in dir() and not results.empty:
            top_labels = results['Player'].tolist()
            fig = radar_chart([target_label] + top_labels[:3], all_df)
            st.pyplot(fig)

# ── CROSS-YEAR: PAST SELF ─────────────────────────────────────────────────────

elif mode == "Cross-Year: Player vs Past Self":
    st.subheader("Track a Player Across Seasons")

    filtered_df = apply_team_filter(all_df, team_filter)
    player      = st.selectbox("Player", sorted(filtered_df['Name'].unique().tolist()))
    player_df   = all_df[all_df['Name'] == player].sort_values('Season')

    if len(player_df) < 2:
        st.warning("Not enough seasons of data for this player.")
    else:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            any_season = int(player_df.iloc[-1]['Season'])
            show_player_card(player, any_season, all_df)
            st.metric("Seasons in dataset", len(player_df))
            display_cols = [f for f in features if f in player_df.columns]
            st.dataframe(
                player_df[['Season'] + display_cols].round(3).reset_index(drop=True),
                use_container_width=True, hide_index=True
            )

        with col_right:
            labels = [f"{player} ({s})" for s in player_df['Season'].tolist()]
            fig = radar_chart(labels, all_df)
            st.pyplot(fig)

# ── CROSS-YEAR: HISTORICAL TWIN ───────────────────────────────────────────────

elif mode == "Cross-Year: Find Historical Twin":
    st.subheader("Find a Player's Historical Twin")
    st.caption("Who from any season between 2015–2025 played most like this player?")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        season       = st.selectbox("Season", SEASONS[::-1])
        season_df    = apply_team_filter(all_df[all_df['Season'] == season], team_filter)
        season_names = sorted(season_df['Name'].unique().tolist())

        if not season_names:
            st.warning("No players match the selected team filter.")
        else:
            player       = st.selectbox("Player", season_names)
            target_label = f"{player} ({season})"

            # Build full cross-year model with selected position features
            available   = [f for f in features if f in all_df.columns]
            full_scaled = build_model_for_features(all_df, available)

            st.divider()
            show_player_card(player, season, all_df)

            results = find_similar(
                target_label, full_scaled, all_df,
                n=n_similar, season_only=False
            )

            if not results.empty:
                st.divider()
                st.markdown("**Historical Twins**")
                show_comp_headshots(results, all_df)
                st.dataframe(results, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Radar Chart")
        if 'results' in dir() and not results.empty:
            top_labels = results['Player'].tolist()
            fig = radar_chart([target_label] + top_labels[:3], all_df)
            st.pyplot(fig)