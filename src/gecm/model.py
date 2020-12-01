import numpy as np
import pandas as pd
from src.gecm.dicts import simplified_lulc_mapping

# dictionary isn't necessary because the conceptual model contains that information
LANDUSE_CHANGE = (
    0.25  # how long does it take until the new landuse returns profit.
)
SUBSIDIES = 0.8
COSTS_LANDUSE_CHANGE = 10

# anzahl touristen als factor für tourism_factor
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
        == cooperation_matrix[col - 1][0]
        == cooperation_matrix[col - 1][col - 1]
        == True
    ):
        teamwork = True
    return teamwork


def unemployment_rate(earning, earning_new, unemployment):
    """
    Args:
        earning:          how much money the player earned last round
        earning_new:      how much money the player earned this round
        unemployment:   unemployment rate of the player had last round

    Returns:            unemployment rate of the player has this round

    """

    # based on okuns law
    delta_unemployment = ((earning - earning_new) / earning) / 1.8
    new_unemployment = max(0, unemployment + delta_unemployment)

    return new_unemployment


# get the total yield for the current map
# not sure why this is necessary - the yield map function calculates the number of pixel for a matrix with values of simplified_lulc_mapping
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


def yield_map(field):
    """
    Args:
        field:  a matrix with values corresponding to simplified_lulc_mapping
    Returns:    the number of pixel for each value of simplified_lulc_mapping

    """

    tot_cattle = np.count_nonzero(
        field == simplified_lulc_mapping["Cattle Farming"]
    )
    tot_sheep = np.count_nonzero(
        field == simplified_lulc_mapping["Sheep Farming"]
    )
    tot_n_forest = np.count_nonzero(
        field == simplified_lulc_mapping["Commercial Forest"]
    )
    tot_c_forest = np.count_nonzero(
        field == simplified_lulc_mapping["Cattle Farming"]
    )
    return tot_cattle, tot_sheep, tot_n_forest, tot_c_forest


# maybe calculate the tourist
def tourism_factor(tourism_factor_matrix, gdp_tourism_factor):
    """
    Args:
        tourism_factor_matrix:     part of lulc mapping at which tourism_factor takes place (20x20)
    Returns: number of tourists and a factor "tourism_factor" that improves earnings.
    """

    cattle, sheep, n_forest, c_forest = yield_map(tourism_factor_matrix)
    sum = cattle + sheep + n_forest + c_forest
    m, n = tourism_factor_matrix.shape
    sum = max(sum, m * n)

    number_tourists = sheep * 2 + n_forest * 3 - c_forest * 5
    # a minimum number of tourits always find their way
    number_tourists = max(number_tourists, sum * 0.75)
    # no more increased sale - tourists start thinking this is too expensive and full
    tourism_factor = min(number_tourists / sum, 1 + gdp_tourism_factor / 100)
    # beach bonus
    if sum < m ** 2 - 10:
        tourism_factor = max(0.95, tourism_factor * 1.2)
    return number_tourists, tourism_factor


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
    fa1 = field[np.ix_(indices1, indices1)]
    fo1 = field[np.ix_(indices2, indices1)]
    fa2 = field[np.ix_(indices1, indices2)]
    fo2 = field[np.ix_(indices2, indices2)]
    return fa1, fa2, fo1, fo2


# adapt the prices
def price_per_pixel(
    current_round,
    brexit,
    tot_sheep_0,
    tot_cattle,
    tot_n_forest,
    tot_c_forest,
    income_farmland_cattle,
    income_farmland_sheep,
    income_forest_native,
    income_forest_commercial,
):
    """
    calculates current prices depending on demand based on an estimate on what's produced in the beginning.
    Args:
        current_round:                  timeline of the game
        brexit:                         time at which brexit happens
        tot_c_forest_0:
        tot_n_forest_0:
        tot_sheep_0

        tot_cattle:                     total amount of pixel for each the current round
        tot_n_forest:
        tot_c_forest:

        income_farmland_cattle,
        income_farmland_sheep,
        income_forest_native,
        income_forest_commercial,
    Returns: the price of the current round for cattle, sheep, native forest and commercial forest

    """
    # doesn't take tourism_factor effects into account yet. and the equations are pretty random.
    cattle_price_new = income_farmland_cattle + (
        tot_cattle
        / (income_farmland_cattle / income_farmland_sheep)
        / tot_sheep_0
    ) * (income_farmland_sheep - income_farmland_cattle)
    sheep_price_new = income_farmland_sheep  # assume sheep can go everywhere, eat everything and no degradation and its profit only influences cattle by competition
    c_forest_price_new = income_forest_commercial + tot_c_forest / (
        income_forest_commercial / income_forest_native
    ) / (tot_n_forest + tot_c_forest) * (
        income_forest_native - income_forest_commercial
    )
    n_forest_price_new = income_forest_native  # assumes native forest can grow everywhere and its profit only influences the commercial forest through competition in the timber market

    if brexit > current_round:
        n_forest_price_new = (
            n_forest_price_new / (1 + SUBSIDIES) * 2
        )  # less import of wood.
        c_forest_price_new = c_forest_price_new / SUBSIDIES

    return (
        cattle_price_new,
        sheep_price_new,
        n_forest_price_new,
        c_forest_price_new,
    )


def money_farmer(
    current_round,
    tourism_factor,
    teams,
    brexit,
    teamwork,
    area_sheep,
    area_cattle,
    area_c_forest,
    area_n_forest,
    sheep_price,
    cattle_price,
    n_forest_price,
    c_forest_price,
    bank_account_farmer_1,
    bank,
    gdp_pc_scotland,
):
    """

    Args:
        current_round:
        tourism_factor:             tourism_factor (from tourism function
        teams:                      in which round are teams allowed
        brexit:                     in which round does brexit happen
        teamwork:                   is teamwork true or false
        area_sheep:                 number of pixel with the respective landuse
        area_cattle:
        area_c_forest:
        area_n_forest:
        sheep_price:                list of prices
        cattle_price:
        n_forest_price:
        c_forest_price:
        bank_account_farmer_1:      assumes same bank_account at the begining for both farmer
        bank:                       how much money is in the bank-account (in round 0 bank_account_farmer_1
        gdp_pc_scotland:

    Returns:                        earnings and bank-account

    """
    if current_round == 0:
        earning = gdp_pc_scotland
        bank_current = bank_account_farmer_1
    else:
        # costs of landscape change
        try:
            d_sheep = area_sheep[current_round] - area_sheep[current_round - 1]
        except:
            pass
        d_cattle = area_cattle[current_round] - area_cattle[current_round - 1]
        d_n_forest = (
            area_n_forest[current_round] - area_n_forest[current_round - 1]
        )  # necessary to potentially allow two changes (i.e. a rise or native forests and cattle on cost of sheep )
        m_change = 0
        m_brexit = 0
        m_teamwork = 0
        if d_n_forest < 0:
            m_change += (
                min([d_cattle, d_n_forest], key=abs)
                * LANDUSE_CHANGE
                * cattle_price[current_round]
            )
            m_change += (
                min([d_sheep, d_n_forest], key=abs)
                * LANDUSE_CHANGE
                * sheep_price[current_round]
            )
        if d_sheep < 0:
            m_change += (
                min([d_cattle, d_sheep], key=abs)
                * LANDUSE_CHANGE
                * cattle_price[current_round]
            )
            m_change += (
                min([d_sheep, d_n_forest], key=abs)
                * LANDUSE_CHANGE
                * n_forest_price[current_round]
            )
        if d_cattle < 0:
            m_change += (
                min([d_cattle, d_sheep], key=abs)
                * LANDUSE_CHANGE
                * sheep_price[current_round]
            )
            m_change += (
                min([d_cattle, d_n_forest], key=abs)
                * LANDUSE_CHANGE
                * n_forest_price[current_round]
            )
        # earning from the area
        m_area = (area_sheep[current_round] * sheep_price[current_round]) + (
            area_cattle[current_round] * cattle_price[current_round]
        )
        # divided by 10 to reduce that profit
        if teamwork == True and teams <= current_round:
            m_teamwork = (
                area_c_forest[current_round]
                * c_forest_price[current_round]
                / 10
                + area_n_forest[current_round]
                * n_forest_price[current_round]
                / 10
            )
        if brexit <= current_round:
            m_brexit = (SUBSIDIES - 1) * (
                area_sheep[current_round] * sheep_price[current_round]
            )
            # if d_n_forest > 0:
            #    m_brexit += d_n_forest * (SUBSIDIES - 1)
        m_tourism = tourism_factor * m_area - m_area

        earning = (
            (m_area + m_change + m_tourism + m_teamwork + m_brexit)
            * gdp_pc_scotland
            / (50 * (area_sheep[current_round] + area_cattle[current_round]))
        )

        bank_current = (
            bank
            - (d_cattle + d_n_forest + d_sheep) / 2 * COSTS_LANDUSE_CHANGE
            + earning
            - gdp_pc_scotland
        )
    return (
        earning,
        bank_current,
    )  # , m_area, m_change, m_tourism, m_brexit, m_teamwork


def money_forester(
    current_round,
    tourism_factor,
    teams,
    brexit,
    teamwork,
    area_sheep,
    area_c_forest,
    area_n_forest,
    sheep_price,
    n_forest_price,
    c_forest_price,
    bank_account_forestry_1,
    bank,
    gdp_pc_scotland,
):
    """

    Args:
        current_round:
        tourism_factor:             tourism_factor (from tourism function
        teams:                      in which round are teams allowed
        brexit:                     in which round does brexit happen - doesn't include higher timber prices here.
        teamwork:                   is teamwork true or false
        area_sheep:                 number of pixel with the respective landuse
        area_cattle:
        area_c_forest:
        area_n_forest:
        sheep_price:                list of prices
        cattle_price:
        n_forest_price:
        c_forest_price:
        bank_account_farmer_1:      assumes same bank_account at the begining for both farmer
        bank:                       how much money is in the bank-account (in round 0 bank_account_farmer_1
        gdp_pc_scotland:

    Returns:                        earnings and bank-account

    """

    if current_round == 0:
        earning = gdp_pc_scotland
        bank_current = bank_account_forestry_1
    else:
        d_n_forest = (
            area_n_forest[current_round] - area_n_forest[current_round - 1]
        )  # necessary to potentially allow two changes (i.e. a rise or native forests and cattle on cost of sheep )
        m_change = 0
        m_brexit = 0
        m_teamwork = 0
        # ich habe momentan gemacht, dass man nur etwas verkleinern darf!
        if d_n_forest < 0:
            m_change += (
                d_n_forest * LANDUSE_CHANGE * c_forest_price[current_round]
            )
        if d_n_forest > 0:
            m_change += (
                d_n_forest * LANDUSE_CHANGE * n_forest_price[current_round]
            )
        # earning from the area
        m_area = (
            area_n_forest[current_round] * n_forest_price[current_round]
        ) + ((area_c_forest[current_round] * c_forest_price[current_round]))
        if teamwork == True and teams <= current_round:
            m_teamwork = area_sheep[current_round] * sheep_price[current_round]

        if brexit <= current_round:
            if d_n_forest > 0:
                m_brexit = d_n_forest * (SUBSIDIES - 1)
            m_brexit += (SUBSIDIES - 1) * (
                area_sheep[current_round] * sheep_price[current_round]
            )

        m_tourism_factor = tourism_factor * m_area
        # maybe return later on the performance of each landuse/industrie --> append() so that its easy to plot?
        # has to make more - maybe beacause he has less land? has still much more...
        earning = (
            (m_area + m_change + m_tourism_factor + m_teamwork + m_brexit)
            * gdp_pc_scotland
            / (
                100
                * (area_n_forest[current_round] + area_c_forest[current_round])
            )
        )
        bank_current = (
            bank
            - abs(d_n_forest) * COSTS_LANDUSE_CHANGE
            + earning
            - gdp_pc_scotland
        )

    return earning, bank_current


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

    # yield_map --> WORKS
    tot_cattle, tot_sheep, tot_n_forest, tot_c_forest = calculate_yield(
        dummy_playing_field_matrix
    )
    print(tot_cattle, tot_sheep, tot_n_forest, tot_c_forest)

    # crop_field --> WORKS
    m1, m2, m3, m4 = crop_field(dummy_playing_field_matrix)
    print(m1.shape, m2.shape, m3.shape, m4.shape)

    # plt.show()
