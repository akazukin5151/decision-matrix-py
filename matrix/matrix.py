"""Module docstring

Warning
-------
Avoid nonsensical values for all floats, such as nan, inf, and extremely large
numbers that can cause an overflow in numpy.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


class Matrix:
    def __init__(self, *choices: str, **kwargs):
        """Add any choices, criteria, and their weights from the constructor into the matrix.

        Parameters
        ----------
        *choices : str
            Competiting items to choose from.

        Keyword args
        ------------
        **kwargs : tuple[Union[str, float]]
            If choices given, then add those choices. If both criteria and weights are given,
            then add those criteria with their weights.
            Choices should be in a tuple of ``str``, criteria should be in a tuple of ``str``,
            and weights should be in a tuple of ``float``.


        Attributes
        ----------
        df : DataFrame
            The pandas DataFrame of the decision matrix.
        continuous_criteria : list[str]
            The names of the criteria that are added as continuous.

        Raises
        ------
        ValueError
            If any given weight value is zero.


        Examples
        --------
        >>> import matrix
        >>> m = matrix.Matrix('apple', 'orange')
        >>> m
        |:-------|
        | Weight |
        | apple  |
        | orange |


        >>> m = matrix.Matrix(
        ...     choices=('apple', 'orange'),
        ...     criteria=('price', 'size'),
        ...     weights=(4, 8)
        ... )
        >>> m
        |        | price   | size   |
        |:-------|:--------|:-------|
        | Weight | 4       | 8      |
        | apple  |         |        |
        | orange |         |        |

        Warning
        -------
        If the length of the criteria tuple and the weight tuple is not the same,
        then the 'extra' values of the longer tuple will be ignored silently, like ``zip``.
        """
        self.df = pd.DataFrame(index=('Weight',))
        self.continuous_criteria: 'list[str]' = []

        # Columns: str   ==> <continuous_criterion>       , <continuous_criterion>_score
        # Rows   : float ==> if <criterion value> is this, then <score> is this
        self._criterion_value_to_score: 'pd.DataFrame[float]'
        self._criterion_value_to_score = pd.DataFrame()

        # Key  : str      ==> criterion name
        # Value: interp1d ==> interpolator function(v: float) -> float
        self._interpolators: dict[str, interpolate.interp1d] = {}

        # Defined in self.if_() and self.then()
        self._if_method_active = False
        self._given_criterion_name: 'Optional[str]' = None
        self._given_criterion_value: 'Optional[float]' = None

        self._setup(*choices, **kwargs)

    def __repr__(self) -> str:
        """Provides a view to the final decision matrix

        Returns
        -------
        DataFrame
            The pandas DataFrame, with NaN values removed and in prettier markdown formatting.
        """
        return self.df.fillna('').to_markdown()

    @property
    def all_choices(self) -> list[str]:
        """Returns a view of the current choices in the matrix

        Returns
        -------
        list[str]
            A list of the choices names.
        """
        self._reject_if_if_method_active()
        return list(self.df.index)[1:]

    @property
    def all_criteria(self) -> list[str]:
        """Returns a view of the current criteria in the matrix

        Returns
        -------
        list[str]
            A list of the criteria names.
        """
        self._reject_if_if_method_active()
        return list(self.df.columns.drop('Percentage', errors='ignore'))

    @property
    def value_score_df(self) -> pd.DataFrame:
        """Provides read-only access to the final decision matrix

        Returns
        -------
        DataFrame
            The pandas DataFrame containing criterion values and their corresponding scores.
        """
        self._reject_if_if_method_active()
        return self._criterion_value_to_score

    def add_choices(self, *choices: str):
        """Add items to choose from into the matrix.

        Parameters
        ----------
        *choices : str
            Competiting items to choose from.

        Examples
        --------
        >>> import matrix
        >>> m = matrix.Matrix()
        >>> m.add_choices('apple', 'orange')
        >>> m
        |:-------|
        | Weight |
        | apple  |
        | orange |
        """
        self._reject_if_if_method_active()
        self.df = self.df.append(pd.DataFrame(index=choices))

    def add_criteria(
        self,
        *criteria: str,
        weights: tuple[float],
        **choices_to_scores: tuple[float]
    ):
        """Add multiple criteria into the matrix to evaluate each choice against.

        Parameters
        ----------
        *criteria : str
            Names of the criteria to add.

        Keyword args
        ------------
        weights : tuple[float]
            How important the criteria are (usually on a 0-10 scale), in order of declaration.
        **choices_to_scores : tuple[float], optional
            Immediately assign scores (dictionary values) to given choices (dictionary keys).
            The tuples must be in the same order of the criteria.

        Raises
        ------
        ValueError
            If any given weight value is zero.

        Examples
        --------
        >>> import matrix
        >>> m = matrix.Matrix()
        >>> m.add_criteria('color', 'taste', weights=(5, 2))
        >>> m
        |        |   color |   taste |
        |:-------|--------:|--------:|
        | Weight |       5 |       2 |


        >>> m = matrix.Matrix('apple', 'orange')
        >>> m.add_criteria('color', 'taste', weights=(5, 2), apple=(4, 8), orange=(7, 5))
        >>> m
        |        |   color |   taste | Percentage        |
        |:-------|--------:|--------:|:------------------|
        | Weight |       5 |       2 |                   |
        | apple  |       4 |       8 | 51.42857142857142 |
        | orange |       7 |       5 | 64.28571428571429 |
        """
        self._reject_if_if_method_active()
        if np.any(weights == 0):
            raise ValueError('Weights cannot be equal to zero!')

        self.df = self.df.append(pd.DataFrame(columns=criteria))
        self.df.loc['Weight', [*criteria]] = weights

        if choices_to_scores:
            new = pd.DataFrame.from_dict(
                choices_to_scores, columns=criteria, orient='index'
            )

            # If some choices has not been added first, add them now anyway
            if len(self.df.index) < len(new.index):
                unadded_choices = {
                    choice: np.nan
                    for choice in set(new.index).difference(self.df.index)
                }
                self.df = self.df.combine_first(
                    pd.DataFrame.from_dict(unadded_choices, orient='index')
                )

            self.df.update(new)
            self._calculate_percentage()

    def add_continuous_criteria(self, *criteria: str, weights: tuple[float]):
        """Add multiple continuous criteria into the matrix to evaluate each choice against.

        Parameters
        ----------
        *criteria : str
            Names of the criteria to add.

        Keyword args
        ------------
        weights : tuple[float]
            How important the criteria are (usually on a 0-10 scale), in order of declaration.
        **choices_to_scores : tuple[float], optional
            Immediately assign scores (dictionary values) to given choices (dictionary keys).
            The tuples must be in the same order of the criteria.

        Raises
        ------
        ValueError
            If any given weight value is zero.

        Examples
        --------
        >>> import matrix
        >>> m = matrix.Matrix()
        >>> m.add_continuous_criteria('price', 'size', weights=(5, 2))
        >>> m
        |        |   price |   size |
        |:-------|--------:|-------:|
        | Weight |       5 |      2 |

        >>> m.continuous_criteria
        ['price', 'size']
        """
        self.add_criteria(*criteria, weights=weights)
        self.continuous_criteria += list(criteria)

    def add_criterion(
        self, criterion: str, *, weight: float, **choices_to_scores: float
    ):
        """Add a criterion into the matrix to evaluate each choice against.

        Parameters
        ----------
        criterion : str
            Name of the criterion to add.

        Keyword args
        ------------
        weight : float
            How important this criterion is (usually on a 0-10 scale).
        **choices_to_scores : float, optional
            Immediately assign a score (dictionary values) to given choices (dictionary keys).

        Raises
        ------
        ValueError
            If a weight value is zero.

        Examples
        --------
        >>> import matrix
        >>> m = matrix.Matrix()
        >>> m.add_criterion('taste', weight=7)
        >>> m
        |        |   taste |
        |:-------|--------:|
        | Weight |       7 |


        >>> m = matrix.Matrix('apple', 'orange')
        >>> m.add_criterion('color', weight=3, apple=4, orange=6)
        >>> m
        |        |   color | Percentage   |
        |:-------|--------:|:-------------|
        | Weight |       3 |              |
        | apple  |       4 | 40.0         |
        | orange |       6 | 60.0         |
        """
        if np.any(weight == 0):
            raise ValueError('Weights cannot be equal to zero!')

        self.add_criteria(criterion, weights=weight, **choices_to_scores)

    def add_continuous_criterion(self, criterion: str, *, weight: float):
        """Add a continuous criterion into the matrix to evaluate each choice against.

        Parameters
        ----------
        criterion : str
            Name of the criterion to add.

        Keyword args
        ------------
        weight : float
            How important this criterion is (usually on a 0-10 scale).
        **choices_to_scores : float, optional
            Immediately assign a score (dictionary values) to given choices (dictionary keys).

        Raises
        ------
        ValueError
            If a weight value is zero.

        Examples
        --------
        >>> import matrix
        >>> m = matrix.Matrix()
        >>> m.add_continuous_criterion('price', weight=7)
        >>> m
        |        |   price |
        |:-------|--------:|
        | Weight |       7 |

        >>> m.continuous_criteria
        ['price']
        """
        self.add_criterion(criterion, weight=weight)
        self.continuous_criteria.append(criterion)

    def score_criterion(self, criterion: str, **choices_to_scores: float):
        """Given a criterion, assign scores (dictionary values) to given choices (dictionary keys).

        Parameters
        ----------
        criterion : str
            The name of the criterion.

        Keyword args
        ------------
        **choices_to_scores : float
            The choice-score pairs.

        Raises
        ------
        ValueError
            If the given criterion is not continuous.

        Examples
        --------
        >>> import matrix
        >>> m = matrix.Matrix(
        ...     choices=('apple', 'orange'),
        ...     criteria=('taste',),
        ...     weights=(7,)
        ... )
        >>> m.score_criterion('taste', apple=7, orange=9)
        >>> m
        |        |   taste | Percentage   |
        |:-------|--------:|:-------------|
        | Weight |       7 |              |
        | apple  |       7 | 70.0         |
        | orange |       9 | 90.0         |
        """
        self._reject_if_if_method_active()
        if criterion in self.continuous_criteria:
            raise ValueError('Cannot assign a score to a continuous criterion!')
        if criterion not in self.df.columns:
            raise ValueError('Criterion has not been added yet, weight is unknown!')

        self.df.update(
            pd.DataFrame.from_dict(
                choices_to_scores, columns=[criterion], orient='index'
            )
        )

        self._calculate_percentage()

    def score_choice(self, choice: str, **criteria_to_scores: float):
        """Given a choice, assign scores (dictionary values) to given criteria (dictionary keys).

        Parameters
        ----------
        choice : str
            The name of the choice to be evaluated.

        Keyword args
        ------------
        **criteria_to_scores : float
            The criterion-score pairs.

        Raises
        ------
        ValueError
            If any of the given criteria is not continuous.

        Examples
        --------
        >>> import matrix
        >>> m = matrix.Matrix(
        ...     choices=('apple',),
        ...     criteria=('taste', 'color'),
        ...     weights=(7, 3)
        ... )
        >>> m.score_choice('apple', taste=7, color=5)
        >>> m
        |        |   taste |   color | Percentage   |
        |:-------|--------:|--------:|:-------------|
        | Weight |       7 |       3 |              |
        | apple  |       7 |       5 | 64.0         |
        """
        self.score_choices({choice: criteria_to_scores})

    def score_choices(
        self, choices_and_criteria_to_scores: dict[str, dict[str, float]]
    ):
        """Given some choices, assign scores (dictionary values) to given criteria (dictionary keys).

        Parameters
        ----------
        choices_and_criteria_to_scores : dict[str, dict[str, float]]
            The nested dictionary containing the choices and the scores for each criteria.

        Raises
        ------
        ValueError
            * If any of the given criteria is not continuous.


            * If any of the given criteria has not been added yet; its weight is unknown.


        Examples
        --------
        >>> import matrix
        >>> m = matrix.Matrix(
        ...     choices=('apple', 'orange'),
        ...     criteria=('taste', 'color'),
        ...     weights=(7, 3)
        ... )
        >>> m.score_choices({
        ...     'apple': {'taste': 7, 'color': 5},
        ...     'orange': {'taste': 9, 'color': 3}
        ... })
        >>> m
        |        |   taste |   color | Percentage   |
        |:-------|--------:|--------:|:-------------|
        | Weight |       7 |       3 |              |
        | apple  |       7 |       5 | 64.0         |
        | orange |       9 |       3 | 72.0         |
        """
        self._reject_if_if_method_active()

        new = pd.DataFrame(choices_and_criteria_to_scores).T

        for criterion in new.columns:
            if criterion in self.continuous_criteria:
                raise ValueError('Cannot assign a score to a continuous criterion!')
            if criterion not in self.df.columns:
                raise ValueError('Criterion has not been added yet, weight is unknown!')

        self.df.update(new)
        self._calculate_percentage()

    def if_(self, **criterion_to_value: float) -> Matrix:
        """
        The first method in the if-then chain syntatic sugar for declaring
        what score should a choice receive given the values for the criterion.

        Keyword args
        ------------
        **criterion_to_value : float
            The criterion (dictionary key) and a given value (dictionary value).

        Raises
        ------
        ValueError
            * If the criterion is not continuous.


            * If the criterion has not been added yet; its weight is unknown.


        Returns
        -------
        Matrix
            Returns the instance so that it can be chained with the then() method.

        Examples
        --------
        This example shows the declaration of a continuous criterion called cost.
        The usage is declarative in nature, mimicking natural language:
        "if the cost is 0 (dollars), then award a score of 10"


        >>> import matrix
        >>> m = matrix.Matrix()
        >>> m.add_continuous_criterion('cost', weight=9)
        >>> m.if_(cost=0).then(score=10)
        >>> m.if_(cost=10).then(score=5)
        >>> m.if_(cost=30).then(score=0)
        >>> m.value_score_df
           cost  cost_score
        0     0          10
        1    10           5
        2    30           0

        If a choice does indeed has a price of 10$, a score of 5
        will be assigned in its cost column

        >>> m.add_choices('apple')
        >>> m.add_data('apple', cost=10)
        >>> m
        |        |   cost | Percentage   |
        |:-------|-------:|:-------------|
        | Weight |      9 |              |
        | apple  |      5 | 50.0         |

        If a choice has a price between a specified value, such as 5$,
        then a simple linear interpolating function will be used as shown below.

        >>> m.add_choices('orange')
        >>> m.add_data('orange', cost=5)
        >>> m
        |        |   cost | Percentage   |
        |:-------|-------:|:-------------|
        | Weight |    9   |              |
        | apple  |    5   | 50.0         |
        | orange |    7.5 | 75.0         |


        Notes
        -----
        The interpolating function is calculated by assuming a linear function
        with points at :math:`(0, 10)` and :math:`(10, 5)`. The x-axis is the cost,
        or the criterion value in general; the y-axis is the score to be assigned.

        .. math::
            \\text{The gradient is: } m = \\frac{(5-10)}{(10-0)} = -0.5

            \\therefore y = (-0.5)x + c

            10 = (-0.5)(0) + c

            \\therefore c = 10

        Thus the interpolating function is :math:`y = (-0.5)x + 10` *for the domain* :math:`0 \le x \le 10`

        If the price is 5$, then :math:`score = (-0.5)(5) + 10 = 7.5`

        See also
        -------
        then : The last method in the chain
        """
        self._reject_if_if_method_active()
        criterion = list(criterion_to_value.keys())[0]
        value = list(criterion_to_value.values())[0]

        if criterion not in self.df.columns:
            raise ValueError('Criterion has not been added yet, weight is unknown!')
        if criterion not in self.continuous_criteria:
            raise ValueError('Criterion is not continuous!')

        self._if_method_active = True
        self._given_criterion_name = criterion
        self._given_criterion_value = value
        return self

    def then(self, *, score: float):
        """
        The last method in the if-then chain syntatic sugar for declaring
        what score should a choice receive given the values for the criterion.

        Keyword args
        ------------
        score : float
            The score to assign given the criterion value from the if_() method.
            Must be a keyword argument.

        Raises
        ------
        SyntaxError
            If this method is called without calling the if_() method first.

        See also
        -------
        if_ : The first method in the chain
        """
        if self._given_criterion_value is None or self._given_criterion_name is None:
            raise SyntaxError('then() method called before an if_() method!')

        self._if_method_active = False
        self.criterion_value_to_score(
            self._given_criterion_name, {self._given_criterion_value: score}
        )

        self._given_criterion_name = None
        self._given_criterion_value = None

    def criteria_values_to_scores(
        self,
        criteria_names: list[str],
        all_values: list[list[float]],
        all_scores: list[list[float]],
    ):
        """
        For multiple continuous criteria, declare what score should a choice receive
        given values for that criterion.

        Parameters
        ----------
        criteria_names : list[str]
            The names of the criteria.
        all_values : list[list[float]]
            The collection of criterion values. The outer list should have a length
            equal to criteria_names.
        all_scores: list[list[float]]
            The collection of criterion scores. The outer list should have a length
            equal to criteria_names.

        Examples
        --------
        >>> import matrix
        >>> m = matrix.Matrix()
        >>> m.add_continuous_criterion('size', weight=4)
        >>> m.add_continuous_criterion('cost', weight=7)
        >>> m.criteria_values_to_scores(
        ...     ['size', 'cost'],
        ...     all_values=[[0, 10, 15], [0, 10]],
        ...     all_scores=[[10, 5, 0], [10, 0]]
        ... )
        >>> m.value_score_df
           size  size_score  cost  cost_score
        0     0          10   0.0        10.0
        1    10           5  10.0         0.0
        2    15           0   NaN         NaN
        """
        for criterion, value_lst, score_lst in zip(
            criteria_names, all_values, all_scores
        ):
            self._criterion_value_to_score = pd.concat([
                self._criterion_value_to_score,
                pd.Series(value_lst, name=criterion),
                pd.Series(score_lst, name=criterion + '_score'),
            ], axis=1)

    def criterion_value_to_score(
        self, criterion_name: str, value_to_scores: dict[float, float]
    ):
        """
        Declare what score should a choice receive for a continuous criterion,
        given the values for that criterion.

        Parameters
        ----------
        criterion_name : str
            The name of the criterion.
        value_to_scores : dict[float, float]
            The criterion value (dictionary key) to score mapping (dictionary value).
            If a choice has a criterion value that matches exactly,
            the corresponding score must be assigned.

        Examples
        --------
        >>> import matrix
        >>> m = matrix.Matrix()
        >>> m.add_continuous_criterion('size', weight=4)
        >>> m.criterion_value_to_score('size', {
        ...     # criterion value: score
        ...     0: 10,
        ...     10: 5,
        ...     15: 0,
        ... })
        >>> m.value_score_df
           size  size_score
        0     0          10
        1    10           5
        2    15           0
        """
        self._reject_if_if_method_active()
        if criterion_name not in self.df.columns:
            raise ValueError('Criterion has not been added yet, weight is unknown!')

        other = pd.DataFrame(
            list(value_to_scores.items()),
            index=range(len(value_to_scores.keys())),
            columns=[criterion_name, criterion_name + '_score'],
        )

        self._criterion_value_to_score = pd.concat(
            [self._criterion_value_to_score, other], ignore_index=True
        )

    def add_data(
        self,
        choice: str,
        values_dict: dict[str, float] = None,
        **criteria_to_values: float
    ):
        """Adds criterion data for the given choice, which is used to calculate their score.

        Parameters
        ----------
        choice : str
            The name of the choice.
        values_dict : Optional[dict[str, float]]
            The criterion-value pairs in the form of a dictionary.

        Keyword args
        ------------
        **criteria_to_values : float
            The criterion-value pairs (str, float) as keyword arguments.

        Raises
        ------
        TypeError
            If neither ``values_dict`` nor ``criteria_to_values`` is given.

        Examples
        --------
        >>> import matrix
        >>> m = matrix.Matrix('apple', 'orange')
        >>> m.add_continuous_criterion('size', weight=4)
        >>> m.add_continuous_criterion('price', weight=8)
        >>> m.if_(price=0).then(score=10)
        >>> m.if_(price=10).then(score=0)
        >>> m.if_(size=0).then(score=0)
        >>> m.if_(size=10).then(score=10)
        >>> m.add_data('apple', price=2, size=3)
        >>> m.add_data('orange', {'price': 7, 'size': 5})
        >>> m
        |        |   size |   price | Percentage         |
        |:-------|-------:|--------:|:-------------------|
        | Weight |      4 |       8 |                    |
        | apple  |      3 |       8 | 63.33333333333333  |
        | orange |      5 |       3 | 36.666666666666664 |
        >>> m.value_score_df
           price  price_score  size  size_score
        0    0.0         10.0   NaN         NaN
        1   10.0          0.0   NaN         NaN
        2    NaN          NaN   0.0         0.0
        3    NaN          NaN  10.0        10.0

        See also
        --------
        batch_add_data : The method that is wrapped

        Todo
        -------
        Use a more descriptive name
        """
        if values_dict and not criteria_to_values:
            return self.batch_add_data({choice: values_dict})
        elif criteria_to_values:
            return self.batch_add_data({choice: criteria_to_values})

        raise TypeError(
            'Criteria values must be given either as a dictionary or as keyword args'
        )

    def batch_add_data(self, choices_and_values: dict[str, dict[str, float]]):
        """For multiple choices, add criterion data.
        If the length of either list is different, the extra items in the
        longer list is ignored.

        Parameters
        ----------
        choices_and_values : dict[str, dict[str, float]]
            The name of the choice The criterion-value pairs.

        Examples
        --------
        >>> import matrix
        >>> m = matrix.Matrix('apple', 'orange')
        >>> m.add_continuous_criterion('size', weight=4)
        >>> m.add_continuous_criterion('price', weight=8)
        >>> m.if_(price=0).then(score=10)
        >>> m.if_(price=10).then(score=0)
        >>> m.if_(size=0).then(score=0)
        >>> m.if_(size=10).then(score=10)
        >>> m.batch_add_data({
        ...     'apple': {'price': 8, 'size': 5},
        ...     'orange': {'price': 5, 'size': 3}
        ... })
        >>> m
        |        |   size |   price | Percentage         |
        |:-------|-------:|--------:|:-------------------|
        | Weight |      4 |       8 |                    |
        | apple  |      5 |       2 | 30.0               |
        | orange |      3 |       5 | 43.333333333333336 |
        """
        new = pd.DataFrame(choices_and_values).T
        for criterion in self.continuous_criteria:
            # Build interpolators
            value = self.value_score_df[criterion].dropna()
            score = self.value_score_df[criterion + '_score'].dropna()
            f = interpolate.interp1d(value, score, fill_value='extrapolate')
            self._interpolators[criterion] = f

            # Apply column-wise (by criteria)
            new.loc[:, criterion] = f(new.loc[:, criterion])

        self.df.update(new)
        self._calculate_percentage()

    def plot_interpolator(self, criterion_name: str, start=0, end=10):
        """Visualize the interpolator function used.
        Needs to explicitly show the plot with `plt.show()`

        Parameters
        ----------
        criterion_name : str
            The name of the criterion to view.
        start : int, optional
            The lower bound of the x-axis (criterion value) to plot. The default is 0.
        end : int, optional
            The upper bound of the x-axis (criterion value) to plot. The default is 10.

        See also
        --------
        :py:class:`scipy.interpolate.interp1d` : The interpolator used
        """
        self._reject_if_if_method_active()
        x = np.arange(start, end)
        y = self._interpolators[criterion_name](x)
        plt.plot(x, y)

    def _setup(self, *choices: str, **kwargs):
        """See docstring for __init__"""
        if choices:
            self.add_choices(*choices)

        if kwargs:
            if 'choices' in kwargs.keys():
                self.add_choices(*kwargs['choices'])

            if 'criteria' in kwargs.keys() and 'weights' in kwargs.keys():
                if np.any(kwargs['weights'] == 0):
                    raise ValueError('Weights cannot be equal to zero!')

                self.add_criteria(*kwargs['criteria'], weights=kwargs['weights'])

    def _calculate_percentage(self):
        """Calculates the Percentage column"""
        self._reject_if_if_method_active()
        max_total = self.df.iloc[0].sum() * 10
        totals = (self.df[1:] * self.df.iloc[0]).sum(axis=1)
        self.df['Percentage'] = totals / max_total * 100
        if 'Percentage' in self.df.columns:
            percentage = self.df.pop('Percentage')
            self.df.insert(len(self.df.columns), 'Percentage', percentage)

    def _reject_if_if_method_active(self):
        """Prevent method calls in-between a if-then chain.

        Raises
        ------
        SyntaxError
            If a method is called immediately after calling the if_() method.
        """
        if self._if_method_active:
            self._if_method_active = False
            raise SyntaxError('if_() method not followed by a then() method!')
