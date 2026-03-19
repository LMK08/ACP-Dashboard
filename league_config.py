"""
Central league configuration for all pipeline scripts and the dashboard.
Maps competition IDs to metadata, credentials, and season info.
"""

import os

COMPETITIONS = {
    43324: {
        "name": "Liga 3",
        "env_user": "WYSCOUT_USER",
        "env_pass": "WYSCOUT_PASS",
        "seasons": {
            191782: "2025/26",
            190090: "2024/25",
            189147: "2023/24",
            188222: "2022/23",
            188221: "2021/22",
        },
        "current_season": 191782,
    },
    702: {
        "name": "Campeonato",
        "env_user": "WYSCOUT_USER_NEW",
        "env_pass": "WYSCOUT_PASS_NEW",
        "seasons": {
            191779: "2025/26",
            190962: "2024/25",
        },
        "current_season": 191779,
    },
}


def get_credentials(comp_id):
    """Return (username, password) for a competition from environment variables."""
    comp = COMPETITIONS[comp_id]
    user = os.environ.get(comp["env_user"], "")
    password = os.environ.get(comp["env_pass"], "")
    return user, password


def all_season_ids(comp_id=None):
    """Return all season IDs, optionally filtered by competition."""
    if comp_id is not None:
        return list(COMPETITIONS[comp_id]["seasons"].keys())
    return [sid for comp in COMPETITIONS.values() for sid in comp["seasons"]]


def current_season_ids():
    """Return dict of {comp_id: current_season_id}."""
    return {cid: comp["current_season"] for cid, comp in COMPETITIONS.items()}


def competition_for_season(season_id):
    """Return the competition ID that owns a given season ID."""
    for comp_id, comp in COMPETITIONS.items():
        if season_id in comp["seasons"]:
            return comp_id
    return None


def competition_name(comp_id):
    """Return the display name for a competition."""
    return COMPETITIONS[comp_id]["name"]


def season_display_name(season_id):
    """Return the display name for a season (e.g. '2025/26')."""
    for comp in COMPETITIONS.values():
        if season_id in comp["seasons"]:
            return comp["seasons"][season_id]
    return str(season_id)


def all_season_id_map():
    """Return a flat {season_id: display_name} map across all competitions."""
    result = {}
    for comp in COMPETITIONS.values():
        result.update(comp["seasons"])
    return result
