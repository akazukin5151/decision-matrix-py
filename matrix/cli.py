from itertools import product

import click
import numpy as np
from rich.console import Console
from rich.table import Table

from matrix import Matrix


def maybe_exit_insufficient(interact, *args):
    if not all(args) and not interact:
        click.get_current_context().fail(
            'Insufficient information in non-interactive mode!\n'
            'At the minimium, choices, criteria, weights, and ratings must be supplied.\n'
            'If continuous criteria is given, their weights, all value-score '
            'pairs, and all data need to be given.'
        )


@click.command()
@click.option('-c', '--choices', 'choices_tup', multiple=True)
@click.option('-C', '--criteria', 'criteria_tup', multiple=True)
@click.option('--continuous-criteria', 'continuous_criteria_tup', multiple=True)
@click.option('--interact/--no-interact', '-i/-I', default=True)
@click.option('-w', '--weights', 'weights_tup', multiple=True)
@click.option('-W', '--continuous-weights', 'c_weights_tup', multiple=True)
@click.option('-r', '--ratings', 'ratings_tup', multiple=True)
@click.option('-V', '--all-values', 'all_c_values', multiple=True)
@click.option('-S', '--all-scores', 'all_c_scores', multiple=True)
@click.option('-d', '--data', 'data_tup', multiple=True)
def main(choices_tup, criteria_tup, continuous_criteria_tup, weights_tup,
         c_weights_tup, ratings_tup, all_c_values, all_c_scores, data_tup: 'tuple[str]',
         interact: bool):
    maybe_exit_insufficient(interact, choices_tup, criteria_tup, weights_tup, ratings_tup)
    if continuous_criteria_tup:
        maybe_exit_insufficient(interact, c_weights_tup, all_c_values, all_c_scores, data_tup)

    # maybe_ask functions for choices, criteria, & ratings doesn't need interact,
    # because if it's not passed in non-interactive mode, maybe_exit_insufficient
    # will throw an error
    choices = maybe_ask_choices(choices_tup)
    criteria = maybe_ask_criteria(criteria_tup)
    continuous_criteria = maybe_ask_continuous_criteria(continuous_criteria_tup, interact)
    weights = maybe_ask_weights(weights_tup, criteria)
    cc_weights = maybe_ask_weights(c_weights_tup, continuous_criteria)
    all_ratings = maybe_ask_ratings(ratings_tup, choices, criteria)
    value_scores = maybe_ask_criterion_value_to_scores(all_c_values, all_c_scores, continuous_criteria)
    data = maybe_ask_data(data_tup, continuous_criteria, choices)

    m = Matrix(*choices, criteria=criteria, weights=weights)
    m.rate_choices(all_ratings)
    m.add_continuous_criteria(*continuous_criteria, weights=cc_weights)
    m.criteria_values_to_scores(continuous_criteria, value_scores[0], value_scores[1])
    m.batch_add_data(data)

    display_table(m.df)


def flat_split(it: 'Iterable[str]', func: 'fn[str] -> T' = lambda x: x) -> 'list[T]':
    """
    For an iterator of strings, split each string by a comma,
    then flatten each resulting lists.

    An optional function can be passed to be applied on every flattened item.

    Example
    -------
    >>> flat_split(('a,b', 'c,d'))
    ['a', 'b', 'c', 'd']

    >>> flat_split(['1,2', '3,4'], float)
    [1.0, 2.0, 3.0, 4.0]
    """
    return [func(item)
            for string in it
            for item in string.split(',')]


def maybe_ask_choices(choices_tup):
    if not choices_tup:
        return click.prompt(
            'Enter the choices, separated by commas with no spaces',
            value_proc=lambda x: x.split(',')
        )
    return flat_split(choices_tup)


def maybe_ask_criteria(criteria_tup):
    if not criteria_tup:
        return click.prompt(
            'Enter the criteria, separated by commas with no spaces:',
            value_proc=lambda x: x.split(',')
        )
    return flat_split(criteria_tup)


def maybe_ask_continuous_criteria(continuous_criteria_tup, interact: bool):
    if not continuous_criteria_tup and interact:
        ans = click.prompt(
            'Are there any more criteria that are continuous and need to be calculated?',
            default='false',
            show_default=True,
            show_choices=True,
            type=click.Choice(['true', 'false'], case_sensitive=False)
        )
        if ans == 'true':
            return click.prompt(
                'Enter the continuous criteria, separated by commas with no spaces',
                value_proc=lambda x: x.split(',')
            )
    return flat_split(continuous_criteria_tup)


def maybe_ask_weights(weights_tup, criteria):
    if not weights_tup:
        return [click.prompt(f'Enter a weight for {criterion}', type=float)
                for criterion in criteria]
    return flat_split(weights_tup, float)


def maybe_ask_ratings(ratings_tup, choices, criteria):
    all_ratings: 'dict[str, dict[str, float]]' = {}

    if not ratings_tup:
        # Dict key = choice, dict value = ratings for every criterion
        for choice in choices:
            # Dict key = criterion, dict value = score
            ratings_for_this_choice: 'dict[str, float]' = {}
            for criterion in criteria:
                ans = click.prompt(
                    f'Enter a score for {choice} on {criterion}\n',
                    type=float
                )
                ratings_for_this_choice.update({criterion: ans})

            all_ratings.update({choice: ratings_for_this_choice})
        return all_ratings

    row_major_values = flat_split(ratings_tup, float)
    rows = range(len(choices))
    cols = range(len(criteria))
    for value, (row, col) in zip(row_major_values, product(rows, cols)):
        current_choice = choices[row]
        current_criterion = criteria[col]

        if current_choice in all_ratings.keys():
            all_ratings[current_choice][current_criterion] = value
        else:
            all_ratings[current_choice] = {current_criterion: value}

    return all_ratings


def maybe_ask_criterion_value_to_scores(all_c_values, all_c_scores, continuous_criteria):
    if continuous_criteria and not (all_c_values or all_c_scores):
        return criterion_to_scores(continuous_criteria)

    all_values = []
    all_scores = []
    for criteria, values_str, scores_str in zip(continuous_criteria, all_c_values, all_c_scores):
        values = values_str.split(',')
        scores = scores_str.split(',')
        all_values.append([float(value) for value in values])
        all_scores.append([float(score) for score in scores])

    return all_values, all_scores


def criterion_to_scores(continuous_criteria) -> 'tuple[list[float], list[float]]':
    """Asks for value and score pairs for every continuous criteria"""
    all_values = []
    all_scores = []

    for criterion in continuous_criteria:
        click.echo('Enter the requirements in the form <criterion>: <score>. Exit with :wq')
        click.echo(f'<Given this {criterion}...>: <What score should this value receive?>')
        click.echo(
            'For example, if criterion is cost, then '
            '`10: 4` means if cost is $10, the item should get a 4'
        )
        click.echo(f'\n{criterion}')

        values, scores = ask_pairs()
        all_values.append(values)
        all_scores.append(scores)

    return all_values, all_scores


def ask_pairs() -> 'tuple[list[float], list[float]]':
    criterion_values: 'list[float]' = []
    scores: 'list[float]' = []
    while True:
        ans = input('> ')
        if ans == ':wq':
            break

        if ': ' not in ans:
            click.echo('Invalid syntax!')
            continue

        value, score = ans.split(': ')

        if not value.isdigit() or not score.isdigit():
            click.echo('Both value and score must be a float!')
            continue

        criterion_values.append(float(value))
        scores.append(float(score))

    return criterion_values, scores


def maybe_ask_data(data_tup, continuous_criteria, choices):
    # Dict key = choice, dict value = criterion values for every criterion
    data = {}
    if continuous_criteria and not data_tup:
        for choice in choices:
            # Dict key = criterion, dict value = criterion value
            values_for_this_choice: 'dict[str, float]' = {}
            for criterion in continuous_criteria:
                ans = click.prompt(
                    f'Enter the {criterion} for {choice}',
                    type=float
                )
                values_for_this_choice.update({criterion: ans})

            data.update({choice: values_for_this_choice})
        return data

    result = {}
    for string in data_tup:
        choice, criterion, value = string.split(',')
        value = float(value)

        if choice in result.keys():
            result[choice].update({criterion: value})
        else:
            result[choice] = {criterion: value}

    return result


def display_table(df):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")

    table.add_column('Choices')
    for col in df.columns:
        table.add_column(col)

    for index, row in df.fillna('').iterrows():
        table.add_row(index, *[str(item) for item in row])

    console.print(table)


if __name__ == '__main__':
    main()
