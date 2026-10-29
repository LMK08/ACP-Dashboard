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
            191779: "2025/26",
            190230: "2023/24",
        },
        # The club's Wyscout subscription dropped the Campeonato (2026-10):
        # every Campeonato request now 403s and 2026/27 has no data. Keep the
        # seasons we already have; inactive = never fetched again, never
        # simulated, and the newest season with data is "current" so the
        # context bar never lands on an empty season.
        "current_season": 191779,
        "active": False,
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


# Season -> start year and the Campeonato season set, derived from COMPETITIONS
# so a new season added above reaches every engine script automatically.
# (Until 2026-10 the engine kept its own hard-coded maps and 2026/27 was
# silently never rated: rows with no year were dropped.)
SEASON_YEAR = {sid: int(label[:4]) for comp in COMPETITIONS.values()
               for sid, label in comp["seasons"].items()}
CAMP_SEASON_IDS = frozenset(COMPETITIONS[702]["seasons"])
# Seasons still being played: RATED (live ACP Index) but never used to fit
# anything built for completed seasons — the projection panel/refit and the
# league-conversion estimates (half a season of 500'+ "pairs" would shift
# them every matchday). A season leaves this set when the next one is added
# and becomes "current_season" at the rollover.
IN_PROGRESS_SEASON_IDS = frozenset(c["current_season"] for c in COMPETITIONS.values()
                                   if c.get("active", True))


def is_active(comp_id):
    """False for a competition we can no longer fetch (see "active")."""
    return COMPETITIONS[comp_id].get("active", True)


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


def is_reserve_team(team_name):
    """A club's B side ('<club> II': Sporting CP II, Sporting Braga II,
    Vitória Guimarães II, FC Alverca II, Chaves II, Portimonense II,
    Marítimo II — Wyscout names every one this way). Few Liga 3 minutes
    there often mean first-team football this data can't see, so the
    engine value's playing-time factor exempts them (models/value/availability.py)."""
    return isinstance(team_name, str) and team_name.endswith(' II')


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
    # UPDATE 2026-10: Wyscout moved the API minutes back onto 614969 (357'
    # through MW6). Minutes now come only from the per-match API
    # (match_minutes.py), where 1361357 has none.
    1361357: 614969,
    # Nicolás Souza at Atlético CP (26-27 Liga 3): the same mid-season
    # duplicate as Calegari (adjacent pid, created together). 1361362
    # ("N. Cristian de Souza", same DOB 2001-03-10) only held events; the API
    # minutes (149' through MW6) are all on 598545.
    1361362: 598545,
}
