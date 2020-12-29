
from dicewars.client.game.board import Board


def probability_of_successful_attack(board, atk_area, target_area):
    """ Calculate probability of attack success

    """
    atk = board.get_area(atk_area)
    target = board.get_area(target_area)
    atk_power = atk.get_dice()
    def_power = target.get_dice()
    return attack_succcess_probability(atk_power, def_power)


def attack_succcess_probability(atk, df):
    """Dictionary with pre-calculated probabilities for each combination of dice

    Parameters
    ----------
    atk : int
        Number of dice the attacker has
    df : int
        Number of dice the defender has

    Returns
    -------
    float
    """
    return {
        2: {
            1: 0.83796296,
            2: 0.44367284,
            3: 0.15200617,
            4: 0.03587963,
            5: 0.00610497,
            6: 0.00076625,
            7: 0.00007095,
            8: 0.00000473,
        },
        3: {
            1: 0.97299383,
            2: 0.77854938,
            3: 0.45357510,
            4: 0.19170096,
            5: 0.06071269,
            6: 0.01487860,
            7: 0.00288998,
            8: 0.00045192,
        },
        4: {
            1: 0.99729938,
            2: 0.93923611,
            3: 0.74283050,
            4: 0.45952825,
            5: 0.22044235,
            6: 0.08342284,
            7: 0.02544975,
            8: 0.00637948,
        },
        5: {
            1: 0.99984997,
            2: 0.98794010,
            3: 0.90934714,
            4: 0.71807842,
            5: 0.46365360,
            6: 0.24244910,
            7: 0.10362599,
            8: 0.03674187,
        },
        6: {
            1: 0.99999643,
            2: 0.99821685,
            3: 0.97529981,
            4: 0.88395347,
            5: 0.69961639,
            6: 0.46673060,
            7: 0.25998382,
            8: 0.12150697,
        },
        7: {
            1: 1.00000000,
            2: 0.99980134,
            3: 0.99466336,
            4: 0.96153588,
            5: 0.86237652,
            6: 0.68516499,
            7: 0.46913917,
            8: 0.27437553,
        },
        8: {
            1: 1.00000000,
            2: 0.99998345,
            3: 0.99906917,
            4: 0.98953404,
            5: 0.94773146,
            6: 0.84387382,
            7: 0.67345564,
            8: 0.47109073,
        },
    }[atk][df]


def possible_attacks(board: Board, player_name: int):
    for area in board.get_player_border(player_name):
        if not area.can_attack():
            continue
        neighbours = area.get_adjacent_areas()
        for adj in neighbours:
            adjacent_area = board.get_area(adj)
            if adjacent_area.get_owner_name() != player_name:
                yield area, adjacent_area


def get_largest_region_size(board, player_name):
    """Get size of the largest region for given player
    """
    regions = board.get_players_regions(player_name)
    return max(len(region) for region in regions)


def get_danger_area_rating(board, player_name):
    """ Get danger area rating. Danger area is an area that is connected
        to any enemy area with more dice. Each point of danger rating
        corresponds to dice strength difference.
    """
    rating = 0
    count = 0
    for area in board.get_player_border(player_name):
        adjacent_areas = area.get_adjacent_areas()
        for adj in adjacent_areas:
            adj = board.get_area(adj)
            if adj.get_owner_name() != player_name:
                if adj.get_dice() > area.get_dice():
                    rating += (adj.get_dice() - area.get_dice())
                    count += 1
    return rating, count


def get_juicy_area_rating(board, player_name):
    """ Get "juicy" area rating. Intuitively it's the opposite of dangerous
        area. Checks how many enemy areas can be attacked with favorable
        chances. The rating gains points for difference of dice strength.
    """
    rating = 0
    count = 0
    attacks = possible_attacks(board, player_name)
    for source, target in attacks:
        if source.get_dice() > target.get_dice():
            rating += (source.get_dice() - target.get_dice())
            count += 1
    return rating, count


def get_split_rating(board, player_name):
    """ Get split rating. This rating measures how divided the bot's regions are
        on the board. Each point of this rating is the difference between total
        area count and region size, for all regions. Player with only 1 region
        will have split rating equal to zero.

        The incentive is to minimize this split rating, so the bot tries to
        choose a move that connects 2 (or even more) regions into one, thus
        reducing this rating by a lot.
    """
    rating = 0
    area_count = len(board.get_player_areas(player_name))
    regions = board.get_players_regions(player_name)
    for region in regions:
        rating += area_count - len(region)
    return rating


def eval_heuristic(board, player_name):
    """ Get evaluation of the board for a given player based on heuristic
        functions and their weighted sum.
    """

    # Area/region counts and sizes
    area_count = len(board.get_player_areas(player_name))
    region_count = len(board.get_players_regions(player_name))
    largest_region_size = get_largest_region_size(board, player_name)

    # Ratings
    danger_rating, danger_count = get_danger_area_rating(board, player_name)
    juicy_rating, juicy_count = get_juicy_area_rating(board, player_name)
    split_rating = get_split_rating(board, player_name)

    # Heuristic weights
    w = {
        "area_count": 0.35,
        "region_count": -0.1,
        "largest_region_size": 1.0,
        "danger_rating": -0.03,
        "danger_count": -0.01,
        "juicy_rating": 0.15,
        "juicy_count": 0.01,
        "split_rating": -0.1
    }

    evaluation = (
            w["area_count"] * area_count
            + w["region_count"] * region_count
            + w["largest_region_size"] * largest_region_size
            + w["danger_rating"] * danger_rating
            + w["danger_count"] * danger_count
            + w["juicy_rating"] * juicy_rating
            + w["juicy_count"] * juicy_count
            + w["split_rating"] * split_rating
    )

    hf = (area_count, region_count, largest_region_size, danger_rating,
          danger_count, juicy_rating, juicy_count, split_rating)

    return evaluation, hf
