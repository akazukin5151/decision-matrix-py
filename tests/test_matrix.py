from unittest.mock import Mock, call

import numpy as np
from scipy import interpolate
import pytest
from hypothesis import given
from hypothesis.strategies import text, floats


import matrix


non_nan_floats = floats(allow_nan=False)
non_nan_non_inf_floats = floats(allow_nan=False, allow_infinity=False)
floats_for_weights = floats(min_value=1, allow_nan=False, allow_infinity=False)


@given(text())
def test_init_with_one_choice(choice):
    m = matrix.Matrix(choice)
    assert choice in m.df.index
    assert 'Weight' in m.df.index
    assert m.all_choices == [choice]


@given(text(), text())
def test_init_with_multiple_choices(choice1, choice2):
    m = matrix.Matrix(choice1, choice2)
    assert choice1 in m.df.index
    assert choice2 in m.df.index
    assert 'Weight' in m.df.index
    assert m.all_choices == [choice1, choice2]


@given(text(), text(), text())
def test_init_with_choices_and_criteria_but_not_weight(choice1, choice2, crit1):
    crit2 = crit1 + '1'  # Criteria cannot be the same
    m = matrix.Matrix(choices=(choice1, choice2), criteria=(crit1, crit2))
    assert choice1 in m.df.index
    assert choice2 in m.df.index
    assert 'Weight' in m.df.index
    assert list(m.df.columns) == m.all_criteria == []  # Criteria not added


@given(text(), text(), text(), floats_for_weights, floats_for_weights)
def test_init_with_choices_criteria_and_weight(choice1, choice2, crit1, weight1, weight2):
    crit2 = crit1 + '1'  # Criteria cannot be the same
    m = matrix.Matrix(choices=(choice1, choice2), criteria=(crit1, crit2), weights=(weight1, weight2))
    assert choice1 in m.df.index
    assert choice2 in m.df.index

    assert m.df.loc['Weight', crit1] == weight1
    assert m.df.loc['Weight', crit2] == weight2


@given(text(), floats_for_weights, floats_for_weights)
def test_all_criteria_simple(crit1, weight1, weight2):
    crit2 = crit1 + '1'  # Criteria cannot be the same
    m = matrix.Matrix()
    m.add_criterion(crit1, weight=weight1)
    m.add_criterion(crit2, weight=weight2)
    assert m.all_criteria == [crit1, crit2]


@given(text())
def test_all_criteria_percentage(choice1):
    choice2 = choice1 + '1'
    m = matrix.Matrix(choices=(choice1, choice2), criteria=('taste', 'color'), weights=(7, 3))
    m.rate_choice(choice1, taste=1, color=2)
    assert m.df.loc[choice1, 'Percentage'] == 13


@given(text(), text())
def test_add_choices(choice1, choice2):
    m = matrix.Matrix()
    m.add_choices(choice1)
    assert choice1 in m.df.index

    m.add_choices(choice2)
    assert choice2 in m.df.index
    assert 'Weight' in m.df.index


@given(text(), floats_for_weights, floats_for_weights)
def test_add_criteria(crit1, weight1, weight2):
    crit2 = crit1 + '1'
    m = matrix.Matrix()
    m.add_criteria(crit1, crit2, weights=(weight1, weight2))
    assert crit1 in m.df.columns
    assert crit2 in m.df.columns
    assert m.df.loc['Weight', crit1] == weight1
    assert m.df.loc['Weight', crit2] == weight2


@given(floats_for_weights, floats_for_weights, non_nan_floats, non_nan_floats,
        non_nan_floats, non_nan_floats)
def test_add_criteria_with_rating(weight1, weight2, rating1, rating2, rating3, rating4):
    m = matrix.Matrix('apple', 'orange')
    m.add_criteria(
        'taste', 'color',
        weights=(weight1, weight2),  # Not used here (yet?)
        apple=(rating1, rating2),
        orange=(rating3, rating4)
    )
    assert m.df.loc['apple']['taste'] == rating1
    assert m.df.loc['apple']['color'] == rating2
    assert m.df.loc['orange']['taste'] == rating3
    assert m.df.loc['orange']['color'] == rating4


@given(text(), floats_for_weights, floats_for_weights)
def test_add_criterion(crit1, weight1, weight2):
    crit2 = crit1 + '1'
    m = matrix.Matrix()
    m.add_criterion(crit1, weight=weight1)
    assert crit1 in m.df.columns
    assert m.df.loc['Weight', crit1] == weight1

    m.add_criterion(crit2, weight=weight2)
    assert crit2 in m.df.columns
    assert m.df.loc['Weight', crit2] == weight2


@given(text(), floats_for_weights, non_nan_floats, non_nan_floats)
def test_add_criteria_and_rating_shortcut(crit, weight, rating1, rating2):
    m = matrix.Matrix('apple', 'orange')
    m.add_criterion(crit, weight=weight, apple=rating1, orange=rating2)
    assert crit in m.df.columns
    assert m.df.loc['Weight', crit] == weight
    assert m.df.loc['apple', crit] == rating1
    assert m.df.loc['orange', crit] == rating2


@given(non_nan_floats, non_nan_floats, floats_for_weights, floats_for_weights)
def test_rate_choice(rating1, rating2, weight1, weight2):
    m = matrix.Matrix(choices=('apple', 'orange'), criteria=('taste', 'color'), weights=(weight1, weight2))
    m.rate_choice('apple', taste=rating1, color=rating2)
    assert m.df.loc['apple', 'taste'] == rating1
    assert m.df.loc['apple', 'color'] == rating2
    assert m.df.loc['apple', 'Percentage'] == pytest.approx(
        (rating1 * weight1 + rating2 * weight2) / ((weight1 + weight2) * 10) * 100
    ) or np.nan  # For near-infinite values
    # Orange is 0 as it hasn't been rated yet


@given(non_nan_floats, non_nan_floats, floats_for_weights)
def test_rate_criterion(rating1, rating2, weight):
    m = matrix.Matrix(choices=('apple', 'orange'), criteria=('taste',), weights=(weight))
    m.rate_criterion('taste', apple=rating1, orange=rating2)
    assert m.df.loc['apple', 'taste'] == rating1
    assert m.df.loc['orange', 'taste'] == rating2
    assert m.df.loc['apple', 'Percentage'] == pytest.approx(
        (weight * rating1) / (weight * 10) * 100
    ) or np.nan
    assert m.df.loc['orange', 'Percentage'] == pytest.approx(
        (weight * rating2) / (weight * 10) * 100
    ) or np.nan


def test_rate_criterion_continuous_criterion_raises():
    m = matrix.Matrix('apple')
    m.add_continuous_criterion('test', weight=3)
    with pytest.raises(ValueError, match='Cannot assign a rating to a continuous criterion!'):
        m.rate_criterion('test', apple=7)


def test_rate_criterion_criterion_not_added_raises():
    m = matrix.Matrix('apple', 'orange')
    with pytest.raises(ValueError, match='Criterion has not been added yet, weight is unknown!'):
        m.rate_criterion('unknown', apple=1, orange=2)


def test_rate_choice_before_adding_criterion_raises():
    m = matrix.Matrix()
    with pytest.raises(ValueError, match='Criterion has not been added yet, weight is unknown!'):
        m.rate_choice('unknown', taste=8)


@given(text())
def test_add_continuous_criteria(crit):
    m = matrix.Matrix('apple', 'orange')
    m.add_continuous_criterion(crit, weight=9)
    assert crit in m.df.columns
    assert crit in m.continuous_criteria


@given(non_nan_floats, non_nan_floats, non_nan_floats, non_nan_floats, non_nan_floats, non_nan_floats)
def test_if_then_chain(cost1, cost2, cost3, score1, score2, score3):
    m = matrix.Matrix()
    m.add_continuous_criterion('cost', weight=9)
    m.if_(cost=cost1).then(score=score1)
    m.if_(cost=cost2).then(score=score2)
    m.if_(cost=cost3).then(score=score3)
    assert list(m._criterion_value_to_score.loc[:, 'cost']) == [cost1, cost2, cost3]
    assert list(m._criterion_value_to_score.loc[:, 'cost_score']) == [score1, score2, score3]


def test_if_then_not_a_criteria_raises():
    m = matrix.Matrix(choices=('apple', 'orange'), criteria=('taste', 'color'), weights=(7, 3))
    with pytest.raises(ValueError, match='Criterion has not been added yet, weight is unknown!'):
        m.if_(hardness=2).then(score=7)


def test_if_then_criteria_not_continuous_raises():
    m = matrix.Matrix(choices=('apple', 'orange'), criteria=('taste', 'color'), weights=(7, 3))
    with pytest.raises(ValueError, match='Criterion is not continuous!'):
        m.if_(color=2).then(score=20)


def test_invalid_usage_of_then():
    m = matrix.Matrix(choices=('apple', 'orange'), criteria=('taste', 'color'), weights=(7, 3))
    with pytest.raises(SyntaxError) as excinfo:
        m.then(score=10)

    assert excinfo.value.args[0] == 'then() method called before an if_() method!'


def test_if_without_then_raises():
    m = matrix.Matrix(choices=('apple', 'orange'), criteria=('taste', 'color'), weights=(7, 3))
    m.add_continuous_criterion('cost', weight=9)
    m.if_(cost=40)
    with pytest.raises(SyntaxError) as excinfo:
        m.add_choices('unadded')

    assert excinfo.value.args[0] == 'if_() method not followed by a then() method!'


def test_criterion_value_to_score():
    m = matrix.Matrix(choices=('apple', 'orange'), criteria=('taste', 'color'), weights=(7, 3))
    m.add_continuous_criterion('cost', weight=9)
    m.criterion_value_to_score('cost', {0: 10, 10: 5, 30: 0})
    assert list(m._criterion_value_to_score.loc[:, 'cost']) == [0, 10, 30]
    assert list(m._criterion_value_to_score.loc[:, 'cost_score']) == [10, 5, 0]


def test_criterion_value_to_score_before_adding_criterion_raises():
    m = matrix.Matrix()
    with pytest.raises(ValueError, match='Criterion has not been added yet, weight is unknown!'):
        m.criterion_value_to_score('price', {0: 10, 5: 5, 30: 0})


@given(non_nan_non_inf_floats, non_nan_non_inf_floats, non_nan_non_inf_floats,
        non_nan_non_inf_floats, non_nan_non_inf_floats, non_nan_non_inf_floats,
        non_nan_non_inf_floats, non_nan_non_inf_floats)
def test_add_data(cost1, cost2, cost3, score1, score2, score3, actual1, actual2):
    m = matrix.Matrix(choices=('apple', 'orange'))
    m.add_continuous_criterion('cost', weight=9)
    m.criterion_value_to_score('cost', {cost1: score1, cost2: score2, cost3: score3})

    # This exception is allowed; occurs when all floats are equal
    try:
        m.add_data('apple', cost=actual1)
    except ValueError as e:
        if e.args[0] == 'x and y arrays must have at least 2 entries':
            return None
        raise e

    m.add_data('orange', cost=actual2)

    calculated_score_1 = m._interpolators['cost'](actual1)
    calculated_score_2 = m._interpolators['cost'](actual2)
    assert m.df.loc['apple', 'cost'] == pytest.approx(calculated_score_1) or np.nan
    assert m.df.loc['orange', 'cost'] == pytest.approx(calculated_score_2) or np.nan
    assert m.df.loc['apple', 'Percentage'] == pytest.approx(calculated_score_1 * 10) or np.nan
    assert m.df.loc['orange', 'Percentage'] == pytest.approx(calculated_score_2 * 10) or np.nan


def test_repr(capsys):
    m = matrix.Matrix(choices=('apple', 'orange'), criteria=('taste', 'color'), weights=(7, 3))
    m.add_continuous_criterion('cost', weight=9)
    m.criterion_value_to_score('cost', {0: 10, 10: 5, 30: 0})
    m.add_data('apple', cost=2)
    m.add_data('orange', cost=7)

    print(m)
    assert capsys.readouterr().out == '|        | taste   | color   |   cost | Percentage         |\n|:-------|:--------|:--------|-------:|:-------------------|\n| Weight | 7       | 3       |    9   |                    |\n| apple  |         |         |    9   | 42.63157894736842  |\n| orange |         |         |    6.5 | 30.789473684210527 |\n'


def test_plot_interpolator(monkeypatch):
    mocked_plot = Mock()
    monkeypatch.setattr('matrix.matrix.plt.plot', mocked_plot)
    m = matrix.Matrix(choices=('apple', 'orange'), criteria=('taste', 'color'), weights=(7, 3))
    m.add_continuous_criterion('cost', weight=9)
    m.criterion_value_to_score('cost', {0: 10, 10: 5, 30: 0})
    m.add_data('apple', cost=2)
    m.add_data('orange', cost=7)

    m.plot_interpolator('cost', 0, 30)

    # It's nested for some reason
    assert list(mocked_plot.call_args[0][0]) == list(range(0, 30))
    assert list(mocked_plot.call_args[0][1]) == [10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4.75, 4.5, 4.25, 4, 3.75, 3.5, 3.25, 3, 2.75, 2.5, 2.25, 2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25]


def test_calculate():
    m = matrix.Matrix(choices=('apple', 'orange'), criteria=('taste', 'color'), weights=(7, 3))
    m.add_continuous_criterion('cost', weight=9)
    m.criterion_value_to_score('cost', {0: 10, 10: 5, 30: 0})
    m.add_data('apple', cost=2)
    m.add_data('orange', cost=7)

    assert list(m.df['Percentage'])[1:] == [42.63157894736842, 30.789473684210527]
    assert str(list(m.df['Percentage'])[0]) == 'nan'


def test_percentage_column_always_last():
    m = matrix.Matrix('apple', 'orange')
    m.add_criterion('taste', weight=7, apple=1, orange=2)
    assert m.df.columns[-1] == 'Percentage'

    m.add_criterion('color', weight=3, apple=3, orange=4)
    assert m.df.columns[-1] == 'Percentage'


def test_add_criterion_also_adds_unknown_choices():
    m = matrix.Matrix()
    m.add_criterion('taste', weight=7, apple=1, orange=2)
    assert 'taste' in m.df.columns
    assert m.df.loc['Weight', 'taste'] == 7
    assert 'apple' in m.df.index
    assert 'orange' in m.df.index


def test_add_criteria_also_adds_unknown_choices():
    m = matrix.Matrix()
    m.add_criteria('color', 'taste', weights=(4, 7), apple=(1, 2), orange=(3, 4))
    assert m.df.loc['Weight', 'color'] == 4
    assert m.df.loc['Weight', 'taste'] == 7
    assert m.df.loc['apple', 'color'] == 1
    assert m.df.loc['apple', 'taste'] == 2
    assert m.df.loc['orange', 'color'] == 3
    assert m.df.loc['orange', 'taste'] == 4


def test_update_weight_no_percentage():
    m = matrix.Matrix()
    m.add_criteria('color', 'taste', weights=(4, 7))
    assert 'Percentage' not in m.df.index
    m.update_weight('color', 9)
    assert m.df.loc['Weight', 'color'] == 9
    assert 'Percentage' not in m.df.index


def test_update_weight():
    m = matrix.Matrix(choices=('apple', 'orange'), criteria=('taste', 'color'), weights=(7, 3))
    m.rate_choice('apple', taste=1, color=2)
    m.rate_choice('orange', taste=3, color=4)
    old_percentages = m.df.loc[:, 'Percentage']

    m.update_weight('color', 9)
    assert m.df.loc['Weight', 'color'] == 9
    assert (m.df.loc[:, 'Percentage'] != old_percentages).any()


def test_rename_criteria():
    m = matrix.Matrix()
    m.add_criteria('color', 'taste', weights=(4, 7))
    m.rename_criteria('color', 'colour')
    assert 'colour' in m.df.columns
    assert 'color' not in m.df.columns


def test_rename_criteria_multiple():
    m = matrix.Matrix()
    m.add_criteria('color', 'taste', weights=(4, 7))
    m.rename_criteria(color='colour', taste='flavour')
    assert 'colour' in m.df.columns
    assert 'color' not in m.df.columns
    assert 'flavour' in m.df.columns
    assert 'taste' not in m.df.columns


def test_rename_choices():
    m = matrix.Matrix('apple', 'orange')
    m.rename_choices('apple', 'pear')
    assert 'pear' in m.df.index
    assert 'apple' not in m.df.index


def test_rename_choices_multiple():
    m = matrix.Matrix('apple', 'orange')
    m.rename_choices(apple='pear', orange='lemon')
    assert 'pear' in m.df.index
    assert 'apple' not in m.df.index
    assert 'lemon' in m.df.index
    assert 'orange' not in m.df.index


def test_rename_choices_raise():
    m = matrix.Matrix('apple', 'orange')
    msg = (
        'Either a choice and a name should be given, or keyword arguments '
        'that maps the old names to the new names'
    )
    with pytest.raises(TypeError, match=msg):
        m.rename_choices()

    with pytest.raises(TypeError, match=msg):
        m.rename_choices('apple')


def test_rename_criteria_raise():
    m = matrix.Matrix()
    msg = (
        'Either a criterion and a name should be given, or keyword arguments '
        'that maps the old names to the new names'
    )
    with pytest.raises(TypeError, match=msg):
        m.rename_criteria()

    with pytest.raises(TypeError, match=msg):
        m.rename_criteria('taste')


@given(non_nan_floats, non_nan_floats, floats_for_weights)
def test_update_rating(rating1, new_rating, weight):
    m = matrix.Matrix(choices=('apple',), criteria=('taste',), weights=(weight))
    m.rate_criterion('taste', apple=rating1)
    old_percentage = m.df.loc['apple', 'Percentage']

    m.update_rating('apple', 'taste', new_rating)
    assert m.df.loc['apple', 'taste'] == new_rating
    # Ignore errors that uses absurd numbers
    if (
        rating1 != new_rating
        and 0 <= rating1 <= 100 and
        0 <= new_rating <= 100 and
        0 <= weight <= 100
    ):
        assert m.df.loc['apple', 'Percentage'] != old_percentage


def test_update_criterion_value_to_score():
    m = matrix.Matrix('apple')
    m.add_continuous_criterion('price', weight=8)
    m.criterion_value_to_score('price', {
        0: 10,
        3: 5,
        10: 0
    })
    assert m.value_score_df.loc[0, 'price_score'] == 10

    m.update_criterion_value_to_score('price', value=0, new_score=3)
    assert m.value_score_df.loc[0, 'price_score'] == 3


def test_remove_criterion_value_to_score():
    m = matrix.Matrix()
    m.add_continuous_criterion('price', weight=8)
    m.criterion_value_to_score('price', {0: 10, 10: 5, 15: 0})
    m.remove_criterion_value_to_score(1)
    assert m.value_score_df.loc[1, 'price'] == 15
    assert m.value_score_df.loc[1, 'price_score'] == 0
