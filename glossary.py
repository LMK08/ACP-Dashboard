"""One glossary for the numbers the dashboard shows (2026-11, UX "Next" item 7).

RULES (Lucas, 2026-11-13):
* A metric's NAME is never changed: every entry is keyed by the name exactly
  as the pages show it. The glossary explains; it renames nothing.
* An entry says only what the code, or the data source, makes certain. A
  metric whose meaning could not be read from the code is in `PENDING` with
  what is known, and is shown nowhere until Lucas defines it. Never invent a
  definition to fill a gap.
* One sentence in football terms, and whether a high value is good where the
  code says so (see `scored_as`). The method stays in the pages' own
  explainers and in the code.

Where it shows: the "Glossary" page (views/glossary.py) and an expander at the
foot of every page that shows metrics (`render_page_footer`, called once by
app.py after the view). The ACP Index entries are read from
`app.ENGINE_METRIC_HELP` and the coach entries from
`models/coaches/coach_metrics.py`, so those texts stay in ONE place.

tests/test_glossary.py holds the gap honest: every metric named in
config.yaml (role templates, stat categories) and on the team radars is
either defined here or listed in `PENDING`.
"""
import functools
import os
import re

HIGHER, LOWER = 'Higher is better', 'Lower is better'
NOT_SCORED = '—'
GPA = "GPA, the club's own action-value model"

# --------------------------------------------------------------------------
# PLAYER metrics. (name as shown, one sentence). Counts are per 90 minutes on
# the pages unless the name holds '%', 'per', 'index' or 'Avg' (app.py's
# per-90 rule) or the page says totals. The direction of a player metric is
# NOT typed here: it is read from config.yaml (see `scored_as`).
# --------------------------------------------------------------------------
PLAYER_GROUPS = {
    'Attacking output': [
        ('Goals', "Goals scored."),
        ('Assists', "Passes Wyscout records as the assist for a goal."),
        ('Second assists', "The pass before the assist, as tagged by Wyscout."),
        ('Shots', "Shots taken. A penalty is a separate event and is not counted as a shot."),
        ('Shots on target', "Shots on target."),
        ('xG', "Expected goals: the chance quality of his shots added up (Wyscout's xG), penalties included."),
        ('npxG', "xG without penalties."),
        ('xG per Shot', "xG divided by shots: the average quality of the chances he shoots from."),
        ('xA', "Expected assists: for each of his passes that led to a shot, the xG of that shot."),
        ('xAOP', "xA from open play."),
        ('xASP', "xA from set pieces: corners, free kicks, throw-ins and goal kicks."),
        ('xT', "Expected threat: how far his accurate passes, his carries and his touches moved the ball "
               "into more dangerous zones of the club's threat grid. Only gains count."),
        ('xTOP', "xT from open play."),
        ('xTSP', "xT from corners, free kicks and throw-ins."),
        ('Shooting Value', f"On-ball value from his shots, in goals ({GPA})."),
        ('Total Value', "All of his on-ball value, every category together, in goals (GPA)."),
        ('Total Offensive Value', "The attacking part of his on-ball value, in goals (GPA)."),
    ],
    'Passing': [
        ('Passing Value', "On-ball value from his passes, in goals (GPA). Creating Value and Linking Value "
                          "split its open-play part."),
        ('Creating Value', "Value from open-play passes into dangerous areas: into the final third or the box, "
                           "crosses, through balls and shot assists (GPA)."),
        ('Linking Value', "Value from all other open-play passes: build-up and circulation (GPA)."),
        ('Receiving Value', "Value from receiving the ball in dangerous positions (GPA)."),
        ('Passes', "Passes attempted."),
        ('Passes successful', "Passes that reached a team-mate (Wyscout's accurate passes)."),
        ('Passes successful %', "Share of his passes that were accurate."),
        ('Long passes', "Passes Wyscout tags as long."),
        ('Long passes successful', "Accurate long passes."),
        ('Long passes successful %', "Share of his long passes that were accurate."),
        ('Crosses', "Passes Wyscout tags as crosses."),
        ('Crosses successful', "Accurate crosses."),
        ('Crosses successful %', "Share of his crosses that were accurate."),
        ('Through passes', "Passes Wyscout tags as through passes."),
        ('Through passes successful', "Accurate through passes."),
        ('Progressive Passes', "Accurate passes that move the ball well forward: at least 30% of the pitch length "
                               "inside his own half, 15% when the pass crosses halfway, 10% inside the opponent's "
                               "half (about 31, 16 and 10 metres). The radar labels it "
                               "\"Progressive passes successful\"."),
        ('Passes to final third', "Passes Wyscout tags as played into the final third."),
        ('Passes to final third successful', "Accurate passes into the final third."),
        ('Forward passes', "Passes Wyscout tags as forward."),
        ('Forward passes successful', "Accurate forward passes."),
        ('Back passes', "Passes Wyscout tags as backward."),
        ('Back passes successful', "Accurate back passes."),
        ('Passes to penalty area', "Passes Wyscout tags as played into the penalty area."),
        ('Passes to penalty area successful', "Accurate passes into the penalty area."),
        ('Deep Completions', "Accurate passes, crosses excluded, that end within 20 metres of the centre of the "
                             "opponent's goal."),
        ('Throw-ins', "Throw-ins taken."),
        ('Avg max throw-in distance', "Average length of his ten longest throw-ins."),
        ('Throw-ins into box', "Throw-ins that land in the opponent's penalty-area zone (the last 16% of the "
                               "pitch, the middle 60% of its width)."),
        ('Avg max throw-in into box distance', "Average length of his ten longest throw-ins into that zone."),
        ('Avg max throw-in into box aerial distance', "The same, counting only throw-ins that were followed by "
                                                      "an aerial duel."),
    ],
    'Defending': [
        ('Possessions won', "Times he won the ball and started a new possession for his team: an interception, "
                            "a duel or a recovery that is the first action of that possession."),
        ('Interceptions', "Interceptions, as recorded by Wyscout."),
        ('Clearances', "Clearances, as recorded by Wyscout."),
        ('Recoveries', "Ball recoveries, as tagged by Wyscout."),
        ('Recoveries Opp Half', "Recoveries made in the opponent's half."),
        ('Counterpressing Recoveries', "Recoveries Wyscout tags as counter-pressing."),
        ('Defensive duels', "Ground duels against the player on the ball, as tagged by Wyscout."),
        ('Defensive duels successful', "Defensive duels in which he won the ball or stopped the attacker's "
                                       "progress."),
        ('Defensive duels successful %', "Share of his defensive duels that were successful."),
        ('Defensive duels vs offensive duel', "Defensive duels against an attacker who shields, holds up or "
                                              "carries the ball, not one who tries to dribble past him."),
        ('Defensive duels vs offensive duel successful %', "Share of those duels in which he won the ball or "
                                                           "stopped the attacker's progress."),
        ('Dribbles faced', "Defensive duels against an attacker who tries to dribble past him."),
        ('Dribbled past %', "Share of the dribbles he faced in which he neither stopped the attacker nor won "
                            "the ball."),
        ('Dribbled past % (proj)', "Dribbled past %, pulled toward the league average when he has faced few "
                                   "dribbles: it takes about 54 before the raw share can be trusted."),
        ('Aerial duels', "Aerial duels contested."),
        ('Aerial duels successful', "Aerial duels in which he got the first touch."),
        ('Aerial duels successful %', "Share of his aerial duels in which he got the first touch."),
        ('Aerial duels def box', "Aerial duels contested in his own penalty-area zone."),
        ('Aerial duels def box successful', "Aerial duels in his own penalty-area zone in which he got the "
                                            "first touch."),
        ('Aerial duels def box successful %', "Share of his aerial duels in his own penalty-area zone in which "
                                              "he got the first touch."),
        ('Aerial duels att box', "Aerial duels contested in the opponent's penalty-area zone."),
        ('Aerial duels att box successful', "Aerial duels in the opponent's penalty-area zone in which he got "
                                            "the first touch."),
        ('Aerial duels att box successful %', "Share of his aerial duels in the opponent's penalty-area zone "
                                              "in which he got the first touch."),
        ('Sliding tackles', "Duels Wyscout tags as sliding tackles."),
        ('Sliding tackles successful', "Sliding tackles in which he won the ball or stopped the attacker's "
                                       "progress."),
        ('Sliding tackles successful %', "Share of his sliding tackles that were successful."),
        ('Fouls', "Fouls committed."),
        ('Yellow cards', "Yellow cards."),
        ('Red cards', "Red cards."),
        ('Avg defensive action height', "How far from his own goal line he defends on average, in metres: the "
                                        "mean position of his interceptions, clearances, defensive and aerial "
                                        "duels and sliding tackles."),
    ],
    'Dribbling and ball security': [
        ('Touches in penalty area', "Touches in the opponent's penalty area, as tagged by Wyscout."),
        ('Progressive runs', "Carries Wyscout tags as progressive."),
        ('Fouls suffered', "Fouls won."),
        ('Dribbling Value', "Value from carries and take-ons (GPA)."),
        ('Losses', "Times he lost the ball, as tagged by Wyscout."),
        ('Loss index', "Ball losses divided by his successful attacking actions: shots on target, accurate "
                       "crosses and Dribbles successful."),
    ],
    'Goalkeeping': [
        ('shotsOnTargetAgainst', "Shots on target he faced."),
        ('goalsConceded', "Goals he conceded from shots on target."),
        ('savePercentage', "Share of the shots on target he faced that he kept out."),
        ('goalsPrevented', "The post-shot expected goals of the shots on target he faced, minus the goals he "
                           "conceded. Above zero: he kept out more than an average goalkeeper would."),
        ('goalsPreventedPerSOT', "goalsPrevented divided by the shots on target he faced."),
        ('exits', "Times he came off his line, as recorded by Wyscout (goalkeeper exits)."),
        ('recoveries_gk', "Ball recoveries by the goalkeeper, as tagged by Wyscout."),
        ('passes_gk', "Passes attempted by the goalkeeper."),
        ('passesSuccessful_gk', "Accurate passes by the goalkeeper."),
        ('GK Passes successful %', "Share of the goalkeeper's passes that were accurate."),
        ('longPasses_gk', "Long passes attempted by the goalkeeper (Wyscout's long-pass tag)."),
        ('longPassesSuccessful_gk', "Accurate long passes by the goalkeeper."),
        ('GK Long passes successful %', "Share of the goalkeeper's long passes that were accurate."),
        ('Shot-Stopping Value', "Goalkeeper value from shot-stopping, in goals (GPA)."),
        ('GK Distribution Value', "Goalkeeper value from his distribution, in goals (GPA)."),
        ('GK Total Value', "All of his goalkeeper value together, in goals (GPA)."),
    ],
}

PLAYER_GROUPS['Dribbling and ball security'][:0] = [
    ('Dribbles', "Dribble attempts: the duels Wyscout tags as his own dribble."),
    ('Dribbles successful', "Dribbles after which his team has the ball closer to goal within 4 seconds (he "
                            "carries on, he passes it forward accurately, or a team-mate picks it up), or on "
                            "which he is fouled."),
    ('Dribbles successful %', "Share of his dribbles that were successful."),
]
PLAYER_GROUPS['Off-ball defending'] = [
    ('Defensive Area', "Size in square metres of the zone he defends: the ellipse that holds about two thirds "
                       "(68%) of his open-play defensive actions at his main position (interceptions, "
                       "clearances, recoveries, defensive and aerial duels, sliding tackles)."),
    ('Opp xT into Def Area', "Expected threat of the opposition moves that end inside his Defensive Area."),
    ('Opp xT from Def Area', "Expected threat of the opposition moves that start inside his Defensive Area."),
    ('Opp xT into Def Area OE', "Opp xT into Def Area divided by the average threat-grid value of the zone he "
                                "defends."),
    ('Opp xT from Def Area OE', "Opp xT from Def Area divided by the average threat-grid value of the zone he "
                                "defends."),
    ('Opp Pass Success % into Def Area', "Share of the opposition passes into his Defensive Area that were "
                                         "accurate."),
    ('Territorial Dominance', "Opp xT into Def Area per square metre of his Defensive Area (multiplied by "
                              "100,000 to be readable)."),
]
PLAYER_GROUPS['Defensive responsibility (DefR)'] = [
    ('DefR Total', "Defensive actions he made minus the number the club's defensive-responsibility model "
                   "expected of him on those plays."),
    ('DefR Interceptions', "DefR counting interceptions only."),
    ('DefR Clearances', "DefR counting clearances only."),
    ('DefR Tackles', "DefR counting tackles only."),
    ('DefR Recoveries', "DefR counting recoveries only."),
    ('DefR Aerials', "DefR counting defensive aerial actions only."),
    ('Def Wins Above Exp', "Defensive duels he won minus the number he was expected to win."),
    ('Tackle Wins AE', "Def Wins Above Exp counting tackles only."),
    ('Aerial Wins AE', "Def Wins Above Exp counting aerial duels only."),
    ('DefR Value Conceded', "Opposition on-ball value conceded where he is the responsible defender. Lower "
                            "means he suppresses more."),
]

# --------------------------------------------------------------------------
# TEAM, MATCH and LEAGUE metrics: (name as shown, one sentence, direction).
# The direction is typed here only where the code states it: the radar's
# inverted list (app.py), SEASON_REPORT_DIMENSIONS' flags, a column's help.
# --------------------------------------------------------------------------
_15S = "in the 15 seconds after"
TEAM_GROUPS = {
    'Team radars': [
        ('Goals', "Goals per match.", HIGHER),
        ('xG', "Expected goals per match.", HIGHER),
        ('xG per Shot', "xG divided by shots: the average quality of the team's chances.", HIGHER),
        ('Shots', "Shots per match.", HIGHER),
        ('Actions in Box', "Actions on the ball in the opponent's penalty area, per match.", HIGHER),
        ('Passes into Box', "Passes that end in the opponent's penalty area, per match. Always counted from "
                            "match events.", HIGHER),
        ('Dribbles', "Dribble attempts per match: the duels Wyscout tags as a dribble by the team's own "
                     "player. Always counted from match events.", HIGHER),
        ('Progressive Passes', "Passes that take the ball clearly closer to the opponent's goal, per match: by "
                               "30 or more (on Wyscout's 0 to 100 pitch scale) inside the own half, 15 when "
                               "the pass crosses halfway, 10 inside the opponent's half. Always counted from "
                               "match events.", HIGHER),
        ('Directness', "Share of the team's passes played forward, in %. A style measure.", None),
        ('Crosses', "Crosses per match.", HIGHER),
        ('Passes', "Passes per match.", HIGHER),
        ('Ball Possession', "Share of possession, in %.", HIGHER),
        ('Losses', "Ball losses per match.", LOWER),
        ('Goals Against', "Goals conceded per match.", LOWER),
        ('xG Against', "Expected goals conceded per match.", LOWER),
        ('xG per Shot Against', "xG Against divided by the shots the team concedes.", LOWER),
        ('Shots Against', "Shots conceded per match.", LOWER),
        ('Aerial Duel Win %', "Share of the team's aerial duels it won.", HIGHER),
        ('Defensive Duel Win %', "Share of the team's defensive duels it won.", HIGHER),
        ('Interceptions', "Interceptions per match.", HIGHER),
        ('Fouls', "Fouls committed per match. The radar draws more fouls as a higher percentile.", None),
        ('PPDA', "Opposition passes allowed per defensive action. Lower means the team presses more.", LOWER),
    ],
    'Set pieces': [
        ('Corners', "Corners taken.", None),
        ('xG per Corner', f"xG of the team's shots {_15S} a corner, per corner.", HIGHER),
        ('Goals per Corner', f"Goals scored {_15S} a corner, per corner.", HIGHER),
        ('Short Corner %', "Share of the team's corners played short (20 metres or less).", None),
        ('Long Throws', "Throw-ins of 25 metres or more that land in or just outside the penalty area.", None),
        ('Long Throw %', "Long throws as a share of the team's throw-ins in the attacking third.", None),
        ('xG per Long Throw', f"xG of the team's shots {_15S} a long throw, per long throw.", HIGHER),
        ('First Contact %', "Share of the team's deliveries (corners that are not short, long throws, free "
                            "kicks of more than 20 metres from the attacking third) after which the next "
                            "action is by its own player.", HIGHER),
        ('xG per FK Delivery', f"xG of the team's shots {_15S} a free kick in the attacking third, per such "
                               "free kick.", HIGHER),
        ('Penalties', "Penalties taken.", None),
        ('Non-Pen SP Goals', f"Goals scored {_15S} a corner, a throw-in in the attacking third or a free kick "
                             "in the attacking third. Penalties are not included.", HIGHER),
        ('xG For', f"Set-piece table: xG of the team's shots {_15S} its own set piece of that kind.", None),
        ('Goals For', f"Set-piece table: goals the team scored {_15S} its own set piece of that kind.", None),
        ('xG Conceded', f"Set-piece table: xG the opponent created {_15S} its set piece of that kind.", None),
        ('Goals Conceded', f"Set-piece table: goals conceded {_15S} an opposition set piece of that kind.",
         None),
    ],
    'Season report': [
        ('PPDA', "Opposition passes in their own 60% of the pitch per defensive action the team makes "
                 "there. Lower means more pressing.", LOWER),
        ('Defensive intensity', "Defensive actions per 90.", HIGHER),
        ('Defensive action height (m)', "How far from its own goal line the team defends on average, in "
                                        "metres.", HIGHER),
        ('Defensive duels won %', "Share of the team's defensive duels it won.", HIGHER),
        ('Opp. xG per match', "Expected goals conceded per match.", LOWER),
        ('Opp. Goals per match', "Goals conceded per match.", LOWER),
        ('Turnovers / 90', "Ball losses per 90 (Wyscout's loss tag).", LOWER),
        ('Recoveries / 90', "Ball recoveries per 90 (Wyscout's recovery tag).", HIGHER),
        ('Final-third recoveries %', "Share of the team's recoveries made in the final third.", HIGHER),
        ('Recovery line height (m)', "How far from its own goal line the team recovers the ball on average, "
                                     "in metres.", HIGHER),
        ('Counter-press recoveries / 90', "Recoveries Wyscout tags as counter-pressing, per 90.", HIGHER),
        ('Opp. Shots / 90', "Shots conceded per 90.", LOWER),
        ('Opp. Shots on target / 90', "Shots on target conceded per 90.", LOWER),
        ('Opp. xG / 90', "Expected goals conceded per 90.", LOWER),
        ('Opp. xG per shot', "Average quality of the chances the team concedes.", LOWER),
        ('Opp. Box touches / 90', "Opposition touches in the team's penalty area per 90.", LOWER),
        ('Opp. High-opportunity shots / 90', "Opposition shots worth 0.25 xG or more, per 90.", LOWER),
        ('Successful long passes / 90', "Accurate long passes per 90 (Wyscout's long-pass tag).", HIGHER),
        ('Ball possession %', "Share of possession, in %. Under a stage filter: the team's share of the two "
                              "sides' passes.", HIGHER),
        ('Field tilt %', "The team's share of the two sides' touches in the final third: its own in the "
                         "attacking third against the opponent's in theirs.", HIGHER),
        ('Pass tempo (passes / min of poss.)', "Passes per minute of possession. Possession time is estimated "
                                               "from the team's share of the passes.", HIGHER),
        ('Long ball %', "Share of the team's passes that are long. A style measure: neither end is better.",
         None),
        ('Passes / 90', "Passes per 90.", HIGHER),
        ('Progressive passes / 90', "Passes Wyscout tags as progressive, per 90.", HIGHER),
        ('Shots / 90', "Shots per 90.", HIGHER),
        ('High-opportunity shots / 90', "Shots worth 0.25 xG or more, per 90.", HIGHER),
        ('xG / 90', "Expected goals per 90.", HIGHER),
        ('Goals / 90', "Goals per 90.", HIGHER),
        ('xG per shot', "Average quality of the team's chances.", HIGHER),
        ('Box touches / 90', "Touches in the opponent's penalty area per 90.", HIGHER),
        ('Shots on target / 90', "Shots on target per 90.", HIGHER),
        ('Points / match', "Points per match.", HIGHER),
        ('xPoints / match', "Expected points per match: from each match's xG for and against, the chance of "
                            "a win (3 points) and of a draw (1).", HIGHER),
        ('Points − xPoints', "Points per match minus expected points per match. Above zero: results have "
                             "been better than the chances.", HIGHER),
        ('Goals / match', "Goals per match.", HIGHER),
        ('Opp. Goals / match', "Goals conceded per match.", LOWER),
        ('xG / match', "Expected goals per match.", HIGHER),
        ('Opp. xG / match', "Expected goals conceded per match.", LOWER),
    ],
    'On-ball value and phases': [
        ('OBV', "On-ball value: the value of a team's or a player's actions on the ball, added up, in the "
                "club's action-value model.", None),
        ('Passing', "OBV chart: value from passes, per match.", None),
        ('Shooting', "OBV chart: value from shots, per match.", None),
        ('Carrying', "OBV chart: value from carries and touches, per match.", None),
        ('Set pieces', "OBV chart: value from corners, free kicks, throw-ins, goal kicks and penalties, per "
                       "match.", None),
        ('Defending', "OBV chart: value from duels, interceptions and clearances, per match.", None),
        ('Goalkeeping', "OBV chart: value from goalkeeper exits, per match.", None),
        ('Discipline', "OBV chart: value from fouls, offsides and own goals, per match.", None),
        ('Buildup', "Phase: an organised possession while the ball is in the team's own third. It succeeds "
                    "when it reaches the middle or final third.", None),
        ('Progression', "Phase: an organised possession while the ball is in the middle third. It succeeds "
                        "when it reaches the final third.", None),
        ('Finishing', "Phase: an organised possession while the ball is in the final third. It succeeds with "
                      "a shot or an entry into the penalty area.", None),
        ('Fast break', "Phase: a possession Wyscout tags as a counter-attack. It succeeds with a shot or an "
                       "entry into the penalty area.", None),
        ('Set piece', "Phase: a possession that starts from a corner, a free kick or a penalty. It succeeds "
                      "with a shot or an entry into the penalty area.", None),
        ('Volume (per 90)', "How many phases of that kind the team has per 90.", None),
        ('Success rate', "Share of the team's phases of that kind that succeed.", None),
        ('OBV per phase', "Average on-ball value of the actions in one phase of that kind.", None),
    ],
    'Match report': [
        ('xG', "Sum of the xG of the team's shots in the match.", HIGHER),
        ('PSxG', "Post-shot expected goals (Wyscout): the chance of a goal once the shot is on its way.", None),
        ('From penalty area / on target', "Shots from inside the penalty area, and how many were on target.",
         None),
        ('Outside penalty area / on target', "Shots from outside the penalty area, and how many were on "
                                             "target.", None),
        ('Attacking free kicks', "Free kicks taken within 45 of the opponent's goal on Wyscout's 0 to 100 "
                                 "pitch scale.", None),
        ('Positional attacks', "Possessions Wyscout marks as an attack down the left, the centre or the "
                               "right.", None),
        ('Final 1/3 Entries', "Possessions that start outside the final third and reach it.", None),
        ('Box Entries', "Possessions with an action in the opponent's penalty area.", None),
        ('Counterattacks', "Balls won at least 40 from the opponent's goal (0 to 100 pitch scale) after which "
                           "the team gets 30 or more closer to it within 5 seconds.", None),
        ('Passes allowed per def. action (PPDA)', "The opponent's passes in its own 60% of the pitch per "
                                                  "defensive action of the team there (interceptions, "
                                                  "clearances, defensive duels, sliding tackles). Lower "
                                                  "means more pressing.", LOWER),
        ('Average shot distance (m)', "Average distance of the team's shots from the centre of the goal, in "
                                      "metres.", None),
        ('Recoveries / low / medium / high', "Interceptions, clearances and ground duels won: the total, "
                                             "then by third of the pitch (own, middle, final).", None),
        ('Opponent half recoveries', "Those recoveries made in the opponent's half.", None),
        ('Losses / low / medium / high', "Times the team's possession ended, fouls won excluded: the total, "
                                         "then by third of the pitch. A misplaced pass is counted where it "
                                         "ended.", None),
        ('Total duels / won', "Every duel, and those won: ground duels in which the team kept or won the "
                              "ball, and aerial duels in which it got the first touch.", None),
        ('Offensive duels / won', "Duels with the ball, and those in which the team kept or won it.", None),
        ('Defensive duels / won', "Duels against the player on the ball, and those in which the team won it "
                                  "or stopped his progress.", None),
        ('Aerial duels / won', "Aerial duels, and those in which the team got the first touch.", None),
        ('Challenge intensity', "Defensive duels and interceptions per minute of opposition possession.", None),
        ('Dribbles / successful', "Dribble attempts (the duels Wyscout tags as a dribble by the team's own "
                                  "player) and the successful ones. Team table: the player got past with the "
                                  "ball. Players table: the team has the ball closer to goal within 4 seconds, "
                                  "or the player is fouled.", None),
        ('Possession %', "The team's share of the two sides' time on the ball.", None),
        ('Pure possession time', "Time the team had the ball.", None),
        ('Dead time', "Match time in which neither side had the ball.", None),
        ('Open play possessions', "Possessions that did not start from a set piece, by how long they lasted "
                                  "(short up to 10 seconds, medium to 20, long to 45, very long beyond).", None),
        ('Progressive passes / accurate', "Passes that take the ball into the last 40% of the pitch, or "
                                          "travel at least 10 metres inside it. Not the rule the player "
                                          "pages use.", None),
        ('Long passes / accurate', "Passes longer than 40 metres.", None),
        ('Passes to final third / accurate', "Passes from outside the final third into it.", None),
        ('Passes to penalty area / accurate', "Passes that end in the opponent's penalty area.", None),
        ('Smart passes / accurate', "Passes Wyscout tags as smart passes.", None),
        ('Shot assists', "Passes Wyscout tags as shot assists.", None),
        ('Crosses: low / high / blocked', "Crosses by height. Blocked crosses are not in the data and show "
                                          "N/A.", None),
        ('Deep completions', "Accurate passes, crosses excluded, that end close to the centre of the "
                             "opponent's goal.", None),
        ('Match tempo', "Passes per minute of the team's own possession.", None),
        ('Minutes', "Players table: exact minutes on the pitch, from Wyscout.", None),
        ('Actions / successful', "Players table: every action of his that Wyscout recorded; successful means "
                                 "accurate passes and shots on target.", None),
        ('Recoveries / opponent half', "Players table: interceptions, clearances and ground duels won, and "
                                       "those in the opponent's half.", None),
        ('Losses / own half', "Players table: inaccurate passes, and those played from his own half. Not the "
                              "Losses of the player pages.", None),
    ],
    'League and predictions': [
        ('P', "Matches played.", None),
        ('W', "Matches won.", None),
        ('D', "Matches drawn.", None),
        ('L', "Matches lost.", None),
        ('GF', "Goals for.", None),
        ('GA', "Goals against.", None),
        ('GD', "Goal difference.", None),
        ('Pts', "Points.", None),
        ('Attacking Strength', "0.3 × goals per match + 0.7 × xG per match.", HIGHER),
        ('Defending Strength', "0.3 × goals conceded per match + 0.7 × xG conceded per match.", LOWER),
        ('Att Strength', "0.3 × goals per match + 0.7 × xG per match.", HIGHER),
        ('Def Strength', "0.3 × goals conceded per match + 0.7 × xG conceded per match.", LOWER),
        ('SOS Att', "Att Strength adjusted for the opponents played: raised after a schedule of sides that "
                    "concede little.", HIGHER),
        ('SOS Def', "Def Strength adjusted for the opponents played: lowered after a schedule of sides that "
                    "score a lot.", LOWER),
        ('SOS Factor', "Above 1: the team has had a tougher schedule than average.", None),
        ('SOS Net', "SOS Att minus SOS Def.", None),
        ('Scores ×', "Goals scored relative to the average of the sides shown (1.00), from the scoreline "
                     "model.", None),
        ('Concedes ×', "Goals conceded relative to the average of the sides shown (1.00), from the scoreline "
                       "model.", LOWER),
        ('xGD vs avg', "Expected goal difference against an average side of this league at a neutral venue.",
         HIGHER),
        ('xPts', "Expected final points over the season simulations.", None),
        ('Promo Series %', "Liga 3, first phase: share of the season simulations in which the team reaches "
                           "the promotion series.", None),
        ('Playoff %', "Campeonato séries: share of the season simulations in which the team finishes in the "
                      "top two of its série. Liga 3 promotion series: the share in which it finishes third.",
         None),
        ('Promo %', "Share of the season simulations in which the team takes a promotion place.", None),
        ('Releg %', "Share of the season simulations in which the team goes down. For a Campeonato série "
                    "without a simulated relegation phase: the share in which it finishes in the bottom five.",
         None),
    ],
}

# --------------------------------------------------------------------------
# NOT DEFINED YET: (group, name as shown) -> what is known. Shown nowhere.
# Lucas defines these (or decides what the number should be); then the entry
# moves into a group above.
# --------------------------------------------------------------------------
PENDING = {
    ('Goalkeeping', 'saves'):
        "Listed in the goalkeeping stats, but no code computes a column of that name.",
    ('Goalkeeping', 'Handling Value'):
        "A goalkeeper category of the action-value model; which actions it covers is not in this repository.",
    ('Goalkeeping', 'Sweeping Value'):
        "A goalkeeper category of the action-value model; which actions it covers is not in this repository.",
}

INTRO = ("Names are shown exactly as on the pages. On the player pages a count is per 90 minutes unless the "
         "page says totals; a name with %, \"per\", \"index\" or \"Avg\" is a rate or an average. "
         "\"Tagged by Wyscout\" means the data provider classifies the action. GPA is the club's own "
         "action-value model: it values each action on the ball, in goals.")
PLAYER_SCORING_NOTE = ("Better: how the role templates treat the metric. \"Higher\" means a template "
                       "weights it, \"Lower\" that it counts against the player, \"—\" that no template "
                       "uses it.")

ENGINE_GROUP = 'ACP Index and value'
COACH_GROUP = 'Coaches'
PLAYER_PAGE_GROUPS = [ENGINE_GROUP] + list(PLAYER_GROUPS)
TEAM_PAGE_GROUPS = ['Team radars', 'Set pieces', 'Season report', 'On-ball value and phases']
# the foot of each page lists the groups of the numbers that page shows
PAGE_GROUPS = {
    'Player Profile': PLAYER_PAGE_GROUPS,
    'Player Comparison': PLAYER_PAGE_GROUPS,
    'Player Analysis': PLAYER_PAGE_GROUPS,
    'Shadow Team': PLAYER_PAGE_GROUPS,
    'Team Analysis': TEAM_PAGE_GROUPS,
    'Opposition Report': TEAM_PAGE_GROUPS,
    'Match Analysis': ['Match report'],
    'League Analysis': ['League and predictions', 'Team radars', 'Set pieces'],
    'Match Predictor': ['League and predictions'],
    'Coach Profile': [COACH_GROUP],
    'Coach Comparison': [COACH_GROUP],
}
# the Glossary page's areas
AREAS = {
    'Players': PLAYER_PAGE_GROUPS,
    'Teams': TEAM_PAGE_GROUPS,
    'Matches': ['Match report'],
    'League and predictions': ['League and predictions'],
    COACH_GROUP: [COACH_GROUP],
}
PAGE_NAME = 'Glossary'
FOOTER_LABEL = "What the numbers on this page mean"


@functools.lru_cache(maxsize=1)
def config_sets():
    """(metrics the role templates weight, metrics config.yaml inverts)."""
    try:
        import yaml
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'), encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return frozenset(), frozenset()
    weighted = {metric for template in (cfg.get('weights') or {}).values() for metric in template}
    return frozenset(weighted), frozenset(cfg.get('invert_metrics') or ())


def scored_as(name):
    """How the role templates score a PLAYER metric, read from config.yaml:
    'Lower is better' when config inverts it, 'Higher is better' when a role
    template weights it (the templates rank every weighted metric high-first
    unless it is inverted), '—' when no template uses it."""
    weighted, inverted = config_sets()
    if name in inverted:
        return LOWER
    return HIGHER if name in weighted else NOT_SCORED


# --------------------------------------------------------------------------
# Reading the glossary
# --------------------------------------------------------------------------
_ROLE_PCT_TAIL = re.compile(r"\s*Within-role percentile.*$", re.S)


def _engine_rows(engine_help):
    """The ACP Index entries: app.ENGINE_METRIC_HELP, the text the '?' tips
    already show (one source). The shared within-role-percentile sentence is
    said once, in the group's note, not on every row."""
    rows = []
    for name, text in (engine_help or {}).items():
        is_role_pct = bool(_ROLE_PCT_TAIL.search(text))
        rows.append((name, _ROLE_PCT_TAIL.sub('', text).strip(),
                     HIGHER if is_role_pct or name.startswith('ACP ') or 'WOE' in name else NOT_SCORED))
    return rows


def _coach_rows():
    """The coach pages' measures, from models/coaches/coach_metrics.py (the
    labels and descriptions those pages show)."""
    try:
        from models.coaches import coach_metrics as cm
    except Exception:
        return []
    # "a high PERCENTILE", not "a high value": PPDA is ranked inverted there
    rows = [(label, f"Style measure. A high percentile means: {meaning}.", NOT_SCORED)
            for label, meaning, _ in cm.STYLE_METRICS.values()]
    rows += [(label, f"Tendency. It counts: {what}.", NOT_SCORED)
             for label, what, _ in cm.TENDENCIES.values()]
    return rows


def groups(engine_help=None):
    """{group: [(name as shown, one sentence, 'Higher is better' | 'Lower is
    better' | '—')]}, in display order. Only defined metrics: nothing from
    `PENDING`."""
    out = {}
    engine = _engine_rows(engine_help)
    if engine:
        out[ENGINE_GROUP] = engine
    for group, rows in PLAYER_GROUPS.items():
        out[group] = [(name, text, scored_as(name)) for name, text in rows]
    for group, rows in TEAM_GROUPS.items():
        out[group] = [(name, text, direction or NOT_SCORED) for name, text, direction in rows]
    coach = _coach_rows()
    if coach:
        out[COACH_GROUP] = coach
    return out


def search(rows, query):
    """Rows whose name or sentence holds every word of `query` (case-blind)."""
    words = [w for w in str(query or '').lower().split() if w]
    if not words:
        return list(rows)
    return [row for row in rows if all(w in f"{row[0]} {row[1]}".lower() for w in words)]


def _md(text):
    """Literal Markdown inside a table cell (a name like passes_gk or
    'Scores ×' must print as typed)."""
    return re.sub(r'([\\`*_\[\]$~|])', r'\\\1', str(text))


_SHORT = {HIGHER: 'Higher', LOWER: 'Lower'}


def table_markdown(rows):
    """Metric | What it is | Better (Higher / Lower / —)."""
    lines = ["| Metric | What it is | Better |", "|---|---|---|"]
    lines += [f"| **{_md(name)}** | {_md(text)} | {_SHORT.get(direction, direction)} |"
              for name, text, direction in rows]
    return "\n".join(lines)


GROUP_NOTES = {
    ENGINE_GROUP: ("The \"%\" grades rank a player among players of the same role in the same league and "
                   "season: 0 to 100, 50 is the typical player of that role. It is a rank, not a success rate."),
    'Team radars': ("Without a stage filter these are Wyscout's published season averages per match, except "
                    "Passes into Box, Progressive Passes and Dribbles, which Wyscout's team file does not hold "
                    "and which are always counted from match events. With a stage filter every axis is "
                    "counted from that stage's events by the club's own rules, close to Wyscout's but not "
                    "the same."),
    'Season report': ("Without a stage filter a figure Wyscout publishes is Wyscout's season average per match; "
                      "the others, and every figure under a stage filter, are counted from the events."),
    COACH_GROUP: "How a coach's teams play, not how well. The coach pages carry the fuller notes.",
}


def render_groups(st, wanted, query='', engine_help=None):
    """Draw the wanted groups as tables; returns how many metrics were shown."""
    shown = 0
    for group, rows in groups(engine_help).items():
        if group not in wanted:
            continue
        rows = search(rows, query)
        if not rows:
            continue
        shown += len(rows)
        st.markdown(f"**{group}**")
        notes = [GROUP_NOTES.get(group), PLAYER_SCORING_NOTE if group in PLAYER_GROUPS else None]
        for note in notes:
            if note:
                st.caption(note)
        st.markdown(table_markdown(rows))
    return shown


def render_page_footer(page, engine_help=None):
    """The explainer at the foot of a page (app.py calls it once, after the
    view): the metrics that page shows, behind one expander, and a way to
    the full glossary. Nothing on a page without metrics of its own."""
    wanted = PAGE_GROUPS.get(page)
    if not wanted:
        return
    import streamlit as st

    import navigation
    with st.expander(FOOTER_LABEL):
        st.caption(INTRO)
        query = st.text_input("Find a metric", key=f"glossary_find_{page}", placeholder="for example: PPDA")
        if not render_groups(st, wanted, query, engine_help):
            st.caption("No metric of this page matches. The full glossary covers every page.")
        if st.button("Open the full glossary", key=f"glossary_open_{page}"):
            navigation.go_to(PAGE_NAME)
