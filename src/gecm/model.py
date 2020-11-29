import numpy as np
import pandas as pd


class ConceptualModel(object):
    """
    Implements the conceptual model class of the game.
    """

    def __init__(self, model_parameters, model_calculations):
        self.model_parameters = model_parameters
        self.model_calculations = model_calculations

        # bank accounts
        self.bank_account_farmers1 = {}
        self.bank_account_farmers2 = {}

def create_dummy_matrix():
    """
    Returns: an 80 x 80 numpy matrix and a 4 x 4 numpy matrix
    """
    # create dummy matrix
    n_plots = 4
    rows = cols = 80

    plot_length = int(rows / n_plots)
    dummy_plot = np.ones(plot_length ** 2).reshape((plot_length, plot_length))
    A1 = dummy_plot
    A2 = dummy_plot * 2
    B1 = dummy_plot * 3
    B2 = dummy_plot * 4
    large_matrix = np.block([[A1, A2], [B1, B2]])
    large_dummy_matrix = np.ones_like(large_matrix)
    n = plot_length
    coef_matrix = np.array([[1, 2], [3, 4]])
    result = np.multiply(
        large_dummy_matrix, np.kron(coef_matrix, np.ones((n, n)))
    )
    matrix_indizes = np.indices((n_plots, n_plots), dtype="uint8") + 1
    row_indizes, column_indizes = matrix_indizes[0], matrix_indizes[1]
    plot_definition_matrix = np.char.add(
        row_indizes.astype(np.str), column_indizes.astype(np.str)
    ).astype(np.uint8)
    dummy_playing_field_matrix = np.ones(shape=(rows, cols), dtype=np.uint8)
    large_plot_definition_matrix = np.multiply(
        dummy_playing_field_matrix,
        np.kron(
            plot_definition_matrix, np.ones(shape=(plot_length, plot_length))
        ),
    )

    return (
        dummy_playing_field_matrix,
        plot_definition_matrix,
        large_plot_definition_matrix,
    )


# assumes four players only / if the four corner player say yes then it's true.
def teamwork(cooperation_matrix):
    """
    assumes four players.  if the four corner player say yes then it's true.

    Args:
        cooperation_matrix: a numpy matrix where the corner elements are a boolean that is true if the associated player wants teamwork.

    Returns:
        a boolean - true if each corner player wants teamwork.
    """
    teamwork = False
    row, col = cooperation_matrix.shape
    if (
        cooperation_matrix[0][0]
        == cooperation_matrix[0][col - 1]
        == cooperation_matrix[row - 1][0]
        == cooperation_matrix[row - 1][col - 1]
        == True
    ):
        teamwork = True
    return teamwork


# get the total yield for the current map
def calculate_yield(field, lulc_mapping, relative=True):
    """

    Args:
        field: a two dimensional numpy array filled with integers
            corresponding to the simplified_lulc_mapping dictionary

    Returns:
        the number of elements of field responding to
            the keys 'Cattle Farming', 'Sheep Farming', 'Native Forest' and 'Commercial Forest' respectively.

    """
    yield_dict = {}
    n_pixels = np.count_nonzero(field.flatten())

    for lulc_class, lulc_id in lulc_mapping.items():

        # calc yield per class
        class_yield = np.count_nonzero(field == lulc_mapping[lulc_class])

        # append
        yield_dict[lulc_id] = (
            class_yield / n_pixels if relative else class_yield
        )

    # TODO: assert check on percentages
    return yield_dict


def crop_field(field):
    """
    Devides an 2D numpy array of quadratric form into four parts.
    Args:
        field: a 2D array

    Returns: 4 2d arrays of 1/4 the size

    """
    m, n = field.shape
    indices1 = list(range(0, int(n / 2)))
    indices2 = list(range(int(n / 2), n))
    m1 = field[np.ix_(indices1, indices1)]
    m2 = field[np.ix_(indices2, indices1)]
    m3 = field[np.ix_(indices1, indices2)]
    m4 = field[np.ix_(indices2, indices2)]
    return m1, m2, m3, m4


def number_of_workers(
    estimate_farmland,
    estimate_forest,
    income_farmland_sheep,
    income_forest_commercial,
    gdp_pc_scotland,
    unempl_rate_scotland,
    farmer,
):
    """

    Args:
        estimate_farmland:    how many pixel of land are farmland
        estimate_forest:        how many pixel of land are forest
        income_farmland_sheep: how much money is earned by a pixel of sheep
        income_forest_commercial: how much money is earned by a pixel of commercial forest
        gdp_pc_scotland:       gdp per capita of scotland
        unempl_rate_scotlnd:  unemployment rate of scotland
        farmer:                 boolean - if its a farmer : true, if its a forester: false


    Returns:    The number f workers suitable for that area.
    """
    if farmer == True:
        number_of_worker = (
            estimate_farmland
            * income_farmland_sheep
            / gdp_pc_scotland
            / (1 - unempl_rate_scotland)
        )
    else:
        number_of_worker = (
            estimate_forest
            * income_forest_commercial
            / gdp_pc_scotland
            * estimate_forest
            / (1 - unempl_rate_scotland)
        )
    return number_of_worker


def unemployment(money, gdp_pc_scottland, number_of_worker):
    """
    uses the number_of_workers and the total amount of money to calculate how many can be employed
    Args:
        money:              amount of money earned
        gdp_pc_scottland:   gdp_pc_of scotland
        number_of_worker:   number of potential workers

    Returns:                unemployment rate

    """
    unempl_rate_scotland = int(money / gdp_pc_scottland) / number_of_workers
    return unempl_rate_scotland


# adapt the prices
def profit_pp(
    round,
    brexit,
    increased_timber_prices,
    tot_sheep,
    tot_cattle,
    tot_n_forest,
    tot_c_forest,
    cattle_pp_0,
    sheep_pp_0,
    n_forest_pp_0,
    c_forest_pp_0,
    income_farmland_cattle,
    income_farmland_sheep,
    income_forest_commercial,
    income_forest_native,
):
    """
    calculates current prices depending on demand based on an estimate on what's produced in the beginning.
    Args:
        round:                      timeline of the game
        brexit:                     time at which brexit happens
        increased_timber_prices:    factor for higher timber prices after brexit
        tot_sheep:                  a list of the total amount of sheep for each timestep
        tot_cattle:
        tot_n_forest:
        tot_c_forest:
        cattle_pp_0:                  the price of the product at the beginning
        sheep_pp_0:
        n_forest_pp_0:
        c_forest_pp_0:
        income_farmland_cattle:
        income_farmland_sheep:          income per pixel
        income_farmland_cattle:
        income_forest_commercial:
        income_forest_commercial:

    Returns: the price of the current round for cattle, sheep, native forest and commercial forest

    """
    # doesn't take tourism effects into account yet. and the equations are pretty random.
    cattle_pp_new = (
        tot_sheep[round]
        / (
            1
            + tot_cattle[round] * income_farmland_cattle / income_farmland_sheep
            + tot_sheep[round]
        )
        * cattle_pp_0
    )  # a certain demand - + 1 so that its never going to infinity should all land become forest
    sheep_pp_new = sheep_pp_0  # assume sheep can go everywhere, eat everything and no degradation and its profit only influences cattle by competition
    c_forest_pp_new = (
        (tot_c_forest[0] + tot_n_forest[0])
        / (
            1
            + tot_c_forest[round]
            * income_forest_commercial
            / income_forest_commercial
            + tot_n_forest[round]
        )
        * c_forest_pp_0
    )
    n_forest_pp_new = n_forest_pp_0  # assumes native forest can grow everywhere and its profit only influences the commercial forest through competition in the timber market
    if brexit > round:
        n_forest_pp_new *= increased_timber_prices  # less import of wood.
        c_forest_pp_new *= increased_timber_prices

    return cattle_pp_new, sheep_pp_new, n_forest_pp_new, c_forest_pp_new


def money_pp_farmer(
    round,
    tourism,
    teams,
    brexit,
    teamwork,
    area_sheep,
    area_cattle,
    area_c_forest,
    area_n_forest,
    sheep_pp,
    cattle_pp,
    n_forest_pp,
    c_forest_pp,
    nf_to_s=0,
    nf_to_c=0,
    s_to_c=0,
    s_to_nf=0,
    c_to_s=0,
    c_to_nf=0,
    subsidies=0,
    starting_capital=0,
):
    """

    Args:
        round:      timeline of the game
        tourism:    a factor which the profit can be multiplied with
        teams:      integer after which round is teamwork allowed
        brexit:     at which round happens brexit
        teamwork:   boolea - does teamwork take place
        area_sheep: list of how many sheep does the farmer own
        area_cattle:
        area_c_forest:
        area_n_forest:
        sheep_pp:   list of what are the prices for each sheep
        cattle_pp:
        n_forest_pp:
        c_forest_pp:
        nf_to_s:        The prices for landuse change. I just copy pasted them at the begining of the function.
        nf_to_c:
        s_to_c:
        s_to_nf:
        c_to_s:
        c_to_nf:
        subsidies:      same
        starting_capital:   integer - how much does he own in the beginning

    Returns:            how much money does a farmer recieve at the currently

    """
    cf_to_nf = 0.5
    nf_to_cf = 0.5
    s_to_c = 0.5
    s_to_nf = c_to_nf = 1  # must stay the same (brexit calculations for farmer)
    c_to_s = 0.5
    nf_to_s = (
        -0.1
    )  # assuming farmers can convert native forest to farmland but not commercial forest (sell wood)
    nf_to_c = 0.8
    subsidies = 0.8

    if round == 0:
        money = starting_capital
    else:
        # costs of landscape change
        try:
            d_sheep = area_sheep[round] - area_sheep[round - 1]
        except:
            pass
        d_cattle = area_cattle[round] - area_cattle[round - 1]
        d_n_forest = (
            area_n_forest[round] - area_n_forest[round - 1]
        )  # necessary to potentially allow two changes (i.e. a rise or native forests and cattle on cost of sheep )
        m_change = 0
        m_brexit = 0
        if d_n_forest < 0:
            m_change += (
                min([d_cattle, d_n_forest], key=abs)
                * nf_to_c
                * cattle_pp[round]
            )
            m_change += (
                min([d_sheep, d_n_forest], key=abs) * nf_to_s * sheep_pp[round]
            )
        if d_sheep < 0:
            m_change += (
                min([d_cattle, d_sheep], key=abs) * s_to_c * cattle_pp[round]
            )
            m_change += (
                min([d_sheep, d_n_forest], key=abs)
                * s_to_nf
                * n_forest_pp[round]
            )
        if d_cattle < 0:
            m_change += (
                min([d_cattle, d_sheep], key=abs) * c_to_s * sheep_pp[round]
            )
            m_change += (
                min([d_cattle, d_n_forest], key=abs)
                * c_to_nf
                * n_forest_pp[round]
            )
        # money from the area
        m_area = (area_sheep[round] * sheep_pp[round]) + (
            area_cattle[round] * cattle_pp[round]
        )
        if teamwork == True and teams > round:
            m_teamwork = (
                area_c_forest[round] * c_forest_pp[round]
                + area_n_forest[round] * n_forest_pp[round]
            )
        if brexit > round:
            m_brexit = (subsidies - 1) * (area_sheep[round] * sheep_pp[round])
            if d_n_forest > 0:
                m_brexit += d_n_forest * (subsidies - 1)
        m_tourism = tourism * m_area
        # maybe return later on the performance of each landuse/industrie --> append() so that its easy to plot?
        money = m_area + m_change + m_tourism + m_teamwork + brexit

    return money


def money_pp_forester(
    round,
    tourism,
    teams,
    brexit,
    teamwork,
    area_sheep,
    area_c_forest,
    area_n_forest,
    sheep_pp,
    n_forest_pp,
    c_forest_pp,
    nf_to_cf,
    cf_to_nf,
    subsidies,
    starting_capital,
):
    """
    how much money does a forester earn per round - similar to farmer.
    area_c_forest: number of pixel displaying commercial forest
    area_n_forest: number of pixel displaying native forest
    round: round of the game (starting at 0)
    """

    cf_to_nf = 0.5
    nf_to_cf = 0.5
    s_to_c = 0.5
    s_to_nf = c_to_nf = 1  # must stay the same (brexit calculations for farmer)
    c_to_s = 0.5
    nf_to_s = (
        -0.1
    )  # assuming farmers can convert native forest to farmland but not commercial forest (sell wood)
    nf_to_c = 0.8
    subsidies = 0.8

    if round == 0:
        money = starting_capital
    else:
        d_n_forest = (
            area_n_forest[round] - area_n_forest[round - 1]
        )  # necessary to potentially allow two changes (i.e. a rise or native forests and cattle on cost of sheep )
        m_change = 0
        m_brexit = 0
        # ich habe momentan gemacht, dass man nur etwas verkleinern darf!
        if d_n_forest < 0:
            m_change += d_n_forest * nf_to_cf * c_forest_pp[round]
        if d_n_forest > 0:
            m_change += d_n_forest * cf_to_nf * n_forest_pp[round]
        # money from the area
        m_area = (area_n_forest[round] * n_forest_pp[round]) + (
            (area_c_forest[round] * c_forest_pp[round])
        )
        if teamwork == True and teams > round:
            m_teamwork = area_sheep[round] * sheep_pp[round]
        if brexit > round:
            if d_n_forest > 0:
                m_brexit = d_n_forest * (subsidies - 1) + (subsidies - 1) * (
                    area_sheep[round] * sheep_pp[round]
                )
        m_tourism = tourism * m_area
        # maybe return later on the performance of each landuse/industrie --> append() so that its easy to plot?
        money = m_area + m_change + m_tourism + m_teamwork + brexit

    return money


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # create dummy data
    (
        dummy_playing_field_matrix,
        plot_definition_matrix,
        large_plot_definition_matrix,
    ) = create_dummy_matrix()
    dummy_playing_field_matrix = np.random.randint(
        1, 4, dummy_playing_field_matrix.shape, np.int8
    )

    # plot dummy data
    plt.imshow(dummy_playing_field_matrix)

    # yield_map --> WORKS
    #tot_cattle, tot_sheep, tot_n_forest, tot_c_forest = calculate_yield(
    #    dummy_playing_field_matrix
    #)
    #print(tot_cattle, tot_sheep, tot_n_forest, tot_c_forest)

    # crop_field --> WORKS
    m1, m2, m3, m4 = crop_field(dummy_playing_field_matrix)
    print(m1.shape, m2.shape, m3.shape, m4.shape)
