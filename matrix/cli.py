from itertools import product

import click
import numpy as np
from rich.console import Console
from rich.table import Table

from matrix import Matrix


@click.command()
@click.option('-c', '--choices', 'choices_tup', multiple=True)
@click.option('-C', '--criteria', 'criteria_tup', multiple=True)
@click.option('--continous-criteria', 'continous_criteria_tup', multiple=True)
@click.option('--interact/--no-interact', ' /-I', default=True)
@click.option('-w', '--weights', 'weights_tup', multiple=True)
@click.option('-W', '--continous-weights', 'c_weights_tup', multiple=True)
@click.option('-s', '--scores', 'scores_tup', multiple=True)
@click.option('-p', '--value-score-pairs', 'value_score_pairs_tup', multiple=True)
@click.option('-d', '--data', 'data_tup', multiple=True)
def main(choices_tup, criteria_tup, continous_criteria_tup, weights_tup,
         c_weights_tup, scores_tup, value_score_pairs_tup, data_tup: 'tuple[str]',
         interact: bool):

    choices = maybe_ask_choices(choices_tup)
    criteria = maybe_ask_criteria(criteria_tup)
    continous_criteria = maybe_ask_continous_criteria(continous_criteria_tup, interact)
    all_criteria = list(criteria) + list(continous_criteria)

    weights = maybe_ask_weights(weights_tup, all_criteria)
    cc_weights = maybe_ask_continous_criteria_weights(c_weights_tup, continous_criteria, interact)
    all_scores = maybe_ask_scores(scores_tup, choices, criteria)
    value_scores = maybe_ask_criterion_value_to_scores(value_score_pairs_tup, continous_criteria)
    data = maybe_ask_data(data_tup, continous_criteria, choices)

    m = Matrix(*choices, criteria=criteria, weights=weights)
    for choice, criterion_to_scores in all_scores.items():
        m.score_choice(choice, **criterion_to_scores)

    for criterion, weight in zip(continous_criteria, cc_weights):
        m.add_criterion(criterion, weight=weight, continous=True)

    for criterion, value_and_score in value_scores.items():
        m.criterion_value_to_score(criterion, {
            value: score
            for value, score in zip(value_and_score[0], value_and_score[1])
        })

    for choice, criteria_to_values in data.items():
        for criterion, value in criteria_to_values.items():
            eval(f'm.add_data(choice, {criterion}=value)')

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


def maybe_ask_continous_criteria(continous_criteria_tup, interact: bool):
    if not continous_criteria_tup and interact:
        ans = click.prompt(
            'Are there any more criteria that are continous and need to be calculated?',
            default='false',
            show_default=True,
            show_choices=True,
            type=click.Choice(['true', 'false'], case_sensitive=False)
        )
        if ans == 'true':
            return click.prompt(
                'Enter the continous criteria, separated by commas with no spaces',
                value_proc=lambda x: x.split(',')
            )
    return flat_split(continous_criteria_tup)


def maybe_ask_weights(weights_tup, all_criteria):
    if not weights_tup:
        return [click.prompt(f'Enter a weight for {criterion}', type=float)
                for criterion in all_criteria]
    return flat_split(weights_tup, float)


def maybe_ask_continous_criteria_weights(c_weights_tup, continous_criteria, interact: bool):
    if not c_weights_tup and interact:
        return [click.prompt(f'Enter a weight for {criterion}', type=float)
                for criterion in continous_criteria]
    return flat_split(c_weights_tup, float)


def maybe_ask_scores(scores_tup, choices, criteria):
    all_scores: 'dict[str, dict[str, float]]' = {}

    if not scores_tup:
        # Dict key = choice, dict value = scores for every criterion
        for choice in choices:
            # Dict key = criterion, dict value = score
            scores_for_this_choice: 'dict[str, float]' = {}
            for criterion in criteria:
                ans = click.prompt(
                    f'Enter a score for {choice} on {criterion}\n',
                    type=float
                )
                scores_for_this_choice.update({criterion: ans})

            all_scores.update({choice: scores_for_this_choice})
        return all_scores

    row_major_values = flat_split(scores_tup, float)
    rows = range(len(choices))
    cols = range(len(criteria))
    for value, (row, col) in zip(row_major_values, product(rows, cols)):
        current_choice = choices[row]
        current_criterion = criteria[col]

        if current_choice in all_scores.keys():
            all_scores[current_choice][current_criterion] = value
        else:
            all_scores[current_choice] = {current_criterion: value}

    return all_scores


def maybe_ask_criterion_value_to_scores(value_score_pairs_tup, continous_criteria):
    if continous_criteria and not value_score_pairs_tup:
        return criterion_to_scores(continous_criteria)

    result = {}
    for string in value_score_pairs_tup:
        criterion, criterion_value, score = string.split(',')
        criterion_value = float(criterion_value)
        score = float(score)

        if criterion in result.keys():
            old_values, old_scores = result[criterion]
            result[criterion] = (old_values + [criterion_value], old_scores + [score])

        else:
            result[criterion] = ([criterion_value], [score])
    return result


def criterion_to_scores(continous_criteria) -> 'dict[str, tuple[np.array[int], np.array[int]]]':
    """Asks for value and score pairs for every continous criteria"""

    # criteria: str, tuple[value: np.array[int], score: np.array[int]]
    dfs: 'dict[str, tuple[np.array[int], np.array[int]]]' = {}

    for criterion in continous_criteria:
        click.echo('Enter the requirements in the form <criterion>: <score>. Exit with :wq')
        click.echo(f'<Given this {criterion}...>: <What score should this value receive?>')
        click.echo(
            'For example, if criterion is cost, then '
            '`10: 4` means if cost is $10, the item should get a 4'
        )
        click.echo(f'\n{criterion}')

        dfs[criterion] = ask_pairs()

    return dfs


def ask_pairs() -> 'tuple[np.array[int], np.array[int]]':
    criterion_values: 'list[int]' = []
    scores: 'list[int]' = []
    while True:
        ans = input('> ')
        if ans == ':wq':
            break

        if ': ' not in ans:
            click.echo('Invalid syntax!')
            continue

        value, score = ans.split(': ')

        if not value.isdigit() or not score.isdigit():
            click.echo('Both value and score must be an int!')
            continue

        criterion_values.append(int(value))
        scores.append(int(score))

    return np.array(criterion_values), np.array(scores)


def maybe_ask_data(data_tup, continous_criteria, choices):
    # Dict key = choice, dict value = criterion values for every criterion
    data = {}
    if continous_criteria and not data_tup:
        for choice in choices:
            # Dict key = criterion, dict value = criterion value
            values_for_this_choice: 'dict[str, float]' = {}
            for criterion in continous_criteria:
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
