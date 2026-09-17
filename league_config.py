"""
Central league configuration for all pipeline scripts and the dashboard.
Maps competition IDs to metadata, credentials, and season info.
"""

import os

COMPETITIONS = {
    43324: {
        "name": "Liga 3",
        # The original WYSCOUT_USER account lost API access in Aug 2026 (403 on
        # every season, including historical). The _NEW account covers Liga 3 too.
        "env_user": "WYSCOUT_USER_NEW",
        "env_pass": "WYSCOUT_PASS_NEW",
        "seasons": {
            192831: "2026/27",
            191782: "2025/26",
            190090: "2024/25",
            189147: "2023/24",
            188222: "2022/23",
            188221: "2021/22",
        },
        "current_season": 192831,
    },
    702: {
        "name": "Campeonato",
        "env_user": "WYSCOUT_USER_NEW",
        "env_pass": "WYSCOUT_PASS_NEW",
        "seasons": {
            192925: "2026/27",
            191779: "2025/26",
            190230: "2023/24",
        },
        "current_season": 192925,
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


# ============================================================================
# PLAYER_ID_ALIASES — Wyscout sometimes splits one real-world player across
# two playerIds (different scrapes, different sources, mistyped name, etc.).
# Map FROM the duplicate pid → TO the canonical pid we want to keep.
#
# After alias resolution every downstream pipeline (GPA, raw_events,
# player_details, valuations, reported_fees) sees only the canonical pid.
# Player_details for the canonical pid wins the bio, so put the pid whose
# bio you want to keep on the RIGHT side of the mapping.
#
# Add a new entry by appending a line like:  <wrong_pid>: <canonical_pid>,
# with a comment naming the player + reason.
PLAYER_ID_ALIASES = {
    # Mamadu Camará at Brito (25-26 Camp). Wyscout has two records:
    # pid 71835 holds the GPA stats but DOB 1991-12-31 is the wrong
    # (older) profile; pid 1322978 has the correct DOB 2001-11-20 but
    # no stats. Remap 71835 → 1322978 so the GPA flows under the
    # correct younger bio.
    71835: 1322978,
    # Alisson Calegari at Atlético CP (26-27 Liga 3). Wyscout opened a
    # second record mid-season: pid 614969 is the established profile
    # (Guimarães II 24/25, Sp. Covilhã 25/26, full bio, DOB 2001-01-03)
    # and still carries the EVENTS of the MW3 Covilhã match; pid 1361357
    # ("A. Calegari", empty bio, same DOB) carries the lineups and the
    # season minutesOnField (164') for BOTH his matches plus the events
    # from MW4 on. Remap 1361357 → 614969 so the history, the bio and
    # this season's events land on one profile. NB the 61' minutes row
    # under 614969 is only precompute_minutes' event-timestamp estimate
    # of the SAME Covilhã appearance (he came on at 36'), so the
    # loader's keep-FROM / drop-TO minutes rule yields the correct 164',
    # not 225' (verified against the match sheets, 2026-09).
    1361357: 614969,
}
