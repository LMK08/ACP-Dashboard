# AI Agent Instructions for Soccer Match Analysis Dashboard

## Project Overview
This is a Python-based soccer match analysis dashboard that processes and visualizes Wyscout event data for Liga 3 Portugal matches. The project consists of:
- Data fetching and processing pipeline (`process_data.py`)
- Interactive Streamlit dashboard (`app.py`) 
- Supporting data files (CSV, Parquet, and Pickle formats)

## Key Components and Data Flow

### Data Pipeline (`process_data.py`)
- **Authentication**: Uses Wyscout API credentials for data access
- **Data Fetching**: Sequential API calls to get match IDs and event data
- **Data Processing**: Multi-step pipeline that generates:
  - Raw events (`.parquet`)
  - Match summaries (`.parquet`)
  - Match-specific analytics (`.pkl`)
  - Season-long team statistics (`.pkl`)

### Dashboard (`app.py`)
- **Data Loading**: Uses `@st.cache_data` for efficient data management
- **Analysis Types**: Supports both match-level and season-long analysis
- **Visualization Components**: Custom shot maps and statistical tables

## Development Workflows

### Setup and Dependencies
```bash
pip install -r requirements.txt
# Key dependencies: streamlit, pandas, numpy, matplotlib, mplsoccer
```

### Data Processing
1. Run full data pipeline:
```bash
python process_data.py
```
2. Verify output files:
- `raw_events.parquet`
- `matches_summary.parquet`
- `all_match_data.pkl`
- `season_team_stats.pkl`

### Running the Dashboard
```bash
streamlit run app.py
```

## Project-Specific Patterns

### Data Structures
- Match event data uses Wyscout's schema (locations in x/y coordinates, nested event types)
- Shot maps use `mplsoccer` pitch coordinates (0-100 x, 0-100 y)
- Statistical tables follow consistent format with categories:
  - General
  - Attacks
  - Defence
  - Transitions
  - Duels
  - Possession
  - Passes

### Key File/Directory References
- `app.py`: Main dashboard interface and visualization logic
- `process_data.py`: Data pipeline and processing functions
- Generated files in root directory (`.parquet`, `.pkl`)

### Common Operations
- Data fetching uses retry logic for API resilience
- Shotmaps handle both match and season-level visualizations
- Statistical calculations preserve team context (home/away)

## Integration Points
- Wyscout API v3 endpoints:
  - `/competitions/{competition_id}/matches`
  - `/matches/{match_id}/events`
- Streamlit components for UI rendering
- Matplotlib/mplsoccer for visualization

## Testing and Validation
- Monitor API response codes and retry mechanisms
- Validate data completeness before processing
- Check for expected output files after pipeline execution