import pytest
from click.testing import CliRunner
from matrix.cli import main


def test_short_no_prompt_in_non_interactive_mode():
    runner = CliRunner()
    args = (
        '-c a,b '
        '-C c,d '
        '-w 1,2 '
        '-s 1,2,3,4 '
        '-I'
    )
    result = runner.invoke(main, args)
    assert result.output == (
        '┏━━━━━━━━━┳━━━━━┳━━━━━┳━━━━━━━━━━━━━━━━━━━━┓\n'
        '┃ Choices ┃ c   ┃ d   ┃ Percentage         ┃\n'
        '┡━━━━━━━━━╇━━━━━╇━━━━━╇━━━━━━━━━━━━━━━━━━━━┩\n'
        '│ Weight  │ 1.0 │ 2.0 │                    │\n'
        '│ a       │ 1.0 │ 2.0 │ 16.666666666666664 │\n'
        '│ b       │ 3.0 │ 4.0 │ 36.666666666666664 │\n'
        '└─────────┴─────┴─────┴────────────────────┘\n'
    )

@pytest.mark.parametrize('interactive', (True, False))
def test_ugly(interactive):
    runner = CliRunner()
    args = (
        '-c a,b '
        '-C c,d '
        '-w 1,2 '
        '-s 1,2,3,4 '
        '--continuous-criteria e,f '
        '-W 5,6 '
        '-V 0,10,20 '
        '-S 10,5,0 '
        '-V 0,5,10 '
        '-S 0,3,10 '
        '-d a,e,1 '
        '-d a,f,2 '
        '-d b,e,3 '
        '-d b,f,4 '
    )

    # Make sure that there's no prompts in non-interactive mode
    if not interactive:
        args += '-I '

    result = runner.invoke(main, args)
    assert result.output == (
        '┏━━━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━━━━━━━━━━━━━━━━┓\n'
        '┃ Choices ┃ c   ┃ d   ┃ e   ┃ f   ┃ Percentage         ┃\n'
        '┡━━━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━━━━━━━━━━━━━━━━┩\n'
        '│ Weight  │ 1.0 │ 2.0 │ 5.0 │ 6.0 │                    │\n'
        '│ a       │ 1.0 │ 2.0 │ 9.5 │ 1.2 │ 42.642857142857146 │\n'
        '│ b       │ 3.0 │ 4.0 │ 8.5 │ 2.4 │ 48.50000000000001  │\n'
        '└─────────┴─────┴─────┴─────┴─────┴────────────────────┘\n'
    )


def test_slightly_better():
    runner = CliRunner()
    args = (
        '-c a,b '
        '-C c,d '
        '-w 1,2 '
        '-s 1,2,3,4 '
        '--continuous-criteria e '
        '-W 5 '
        '-V 0,10,20 '
        '-S 10,5,0 '
        '--continuous-criteria f '
        '-W 6 '
        '-V 0,5,10 '
        '-S 0,3,10 '
        '-d a,e,1 '
        '-d a,f,2 '
        '-d b,e,3 '
        '-d b,f,4 '
    )
    result = runner.invoke(main, args)
    assert result.output == (
        '┏━━━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━━━━━━━━━━━━━━━━┓\n'
        '┃ Choices ┃ c   ┃ d   ┃ e   ┃ f   ┃ Percentage         ┃\n'
        '┡━━━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━━━━━━━━━━━━━━━━┩\n'
        '│ Weight  │ 1.0 │ 2.0 │ 5.0 │ 6.0 │                    │\n'
        '│ a       │ 1.0 │ 2.0 │ 9.5 │ 1.2 │ 42.642857142857146 │\n'
        '│ b       │ 3.0 │ 4.0 │ 8.5 │ 2.4 │ 48.50000000000001  │\n'
        '└─────────┴─────┴─────┴─────┴─────┴────────────────────┘\n'
    )


def test_longer_options():
    runner = CliRunner()
    args = (
        '--choices a,b '
        '--criteria c,d '
        '--weights 1,2 '
        '--scores 1,2,3,4 '
        '--continuous-criteria e '
        '--continuous-weights 5 '
        '--all-values 0,10,20 '
        '--all-scores 10,5,0 '
        '--continuous-criteria f '
        '--continuous-weights 6 '
        '--all-values 0,5,10 '
        '--all-scores 0,3,10 '
        '--data a,e,1 '
        '--data a,f,2 '
        '--data b,e,3 '
        '--data b,f,4 '
    )
    result = runner.invoke(main, args)
    assert result.output == (
        '┏━━━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━━━━━━━━━━━━━━━━┓\n'
        '┃ Choices ┃ c   ┃ d   ┃ e   ┃ f   ┃ Percentage         ┃\n'
        '┡━━━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━━━━━━━━━━━━━━━━┩\n'
        '│ Weight  │ 1.0 │ 2.0 │ 5.0 │ 6.0 │                    │\n'
        '│ a       │ 1.0 │ 2.0 │ 9.5 │ 1.2 │ 42.642857142857146 │\n'
        '│ b       │ 3.0 │ 4.0 │ 8.5 │ 2.4 │ 48.50000000000001  │\n'
        '└─────────┴─────┴─────┴─────┴─────┴────────────────────┘\n'
    )


def test_continuous_input_no():
    runner = CliRunner()
    args = (
        '--choices a,b '
        '--criteria c,d '
        '--weights 1,2 '
        '--scores 1,2,3,4 '
    )
    result = runner.invoke(main, args, input='false\n')
    assert result.output == (
        'Are there any more criteria that are continuous and need to be calculated? (true, false) [false]: false\n'
        '┏━━━━━━━━━┳━━━━━┳━━━━━┳━━━━━━━━━━━━━━━━━━━━┓\n'
        '┃ Choices ┃ c   ┃ d   ┃ Percentage         ┃\n'
        '┡━━━━━━━━━╇━━━━━╇━━━━━╇━━━━━━━━━━━━━━━━━━━━┩\n'
        '│ Weight  │ 1.0 │ 2.0 │                    │\n'
        '│ a       │ 1.0 │ 2.0 │ 16.666666666666664 │\n'
        '│ b       │ 3.0 │ 4.0 │ 36.666666666666664 │\n'
        '└─────────┴─────┴─────┴────────────────────┘\n'
    )


def test_input_continuous_weight_and_data():
    runner = CliRunner()
    args = (
        '--choices a,b '
        '--criteria c,d '
        '--weights 1,2 '
        '--scores 1,2,3,4 '
    )
    ans = (
        # Continous criteria
        'true\n'
        'e,f\n'
        # Weights
        '5\n'
        '6\n'
        # Value-score pairs
        '0: 10\n'
        '10: 5\n'
        '20: 10\n'
        ':wq\n'
        '0: 0\n'
        '5: 3\n'
        '10: 10\n'
        ':wq\n'
        # Data
        '1\n'
        '2\n'
        '3\n'
        '4\n'
    )

    result = runner.invoke(main, args, input=ans)
    assert result.output == (
        'Are there any more criteria that are continuous and need to be calculated? (true, false) [false]: true\n'
        'Enter the continuous criteria, separated by commas with no spaces: e,f\n'
        'Enter a weight for e: 5\n'
        'Enter a weight for f: 6\n'
        'Enter the requirements in the form <criterion>: <score>. Exit with :wq\n'
        '<Given this e...>: <What score should this value receive?>\n'
        'For example, if criterion is cost, then `10: 4` means if cost is $10, the item should get a 4\n\n'
        'e\n'
        '> > > > Enter the requirements in the form <criterion>: <score>. Exit with :wq\n'
        '<Given this f...>: <What score should this value receive?>\n'
        'For example, if criterion is cost, then `10: 4` means if cost is $10, the item should get a 4\n\n'
        'f\n'
        '> > > > Enter the e for a: 1\n'
        'Enter the f for a: 2\n'
        'Enter the e for b: 3\n'
        'Enter the f for b: 4\n'
        '┏━━━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━━━━━━━━━━━━━━━━┓\n'
        '┃ Choices ┃ c   ┃ d   ┃ e   ┃ f   ┃ Percentage         ┃\n'
        '┡━━━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━━━━━━━━━━━━━━━━┩\n'
        '│ Weight  │ 1.0 │ 2.0 │ 5.0 │ 6.0 │                    │\n'
        '│ a       │ 1.0 │ 2.0 │ 9.5 │ 1.2 │ 42.642857142857146 │\n'
        '│ b       │ 3.0 │ 4.0 │ 8.5 │ 2.4 │ 48.50000000000001  │\n'
        '└─────────┴─────┴─────┴─────┴─────┴────────────────────┘\n'
    )


def test_input_choice_criteria_weight_scores():
    runner = CliRunner()
    ans = (
        # Choices
        'a,b\n'
        # Criteria
        'c,d\n'
        # No continuous criteria
        'false\n'
        # Weights
        '1\n'
        '2\n'
        # Scores
        '1\n'
        '2\n'
        '3\n'
        '4\n'
    )

    result = runner.invoke(main, input=ans)
    assert result.output == (
        'Enter the choices, separated by commas with no spaces: a,b\n'
        'Enter the criteria, separated by commas with no spaces:: c,d\n'
        'Are there any more criteria that are continuous and need to be calculated? (true, false) [false]: false\n'
        'Enter a weight for c: 1\n'
        'Enter a weight for d: 2\n'
        'Enter a score for a on c\n'
        ': 1\n'
        'Enter a score for a on d\n'
        ': 2\n'
        'Enter a score for b on c\n'
        ': 3\n'
        'Enter a score for b on d\n'
        ': 4\n'
        '┏━━━━━━━━━┳━━━━━┳━━━━━┳━━━━━━━━━━━━━━━━━━━━┓\n'
        '┃ Choices ┃ c   ┃ d   ┃ Percentage         ┃\n'
        '┡━━━━━━━━━╇━━━━━╇━━━━━╇━━━━━━━━━━━━━━━━━━━━┩\n'
        '│ Weight  │ 1.0 │ 2.0 │                    │\n'
        '│ a       │ 1.0 │ 2.0 │ 16.666666666666664 │\n'
        '│ b       │ 3.0 │ 4.0 │ 36.666666666666664 │\n'
        '└─────────┴─────┴─────┴────────────────────┘\n'
    )


def test_input_all():
    runner = CliRunner()
    ans = (
        # Choices
        'a,b\n'
        # Criteria
        'c,d\n'
        # Continuous criteria
        'true\n'
        'e,f\n'
        # Weights
        '1\n'
        '2\n'
        # Continuous weights
        '5\n'
        '6\n'
        # Scores
        '1\n'
        '2\n'
        '3\n'
        '4\n'
        # Value-score pairs
        '0: 10\n'
        '10: 5\n'
        '20: 10\n'
        ':wq\n'
        '0: 0\n'
        '5: 3\n'
        '10: 10\n'
        ':wq\n'
        # Data
        '1\n'
        '2\n'
        '3\n'
        '4\n'
    )

    result = runner.invoke(main, input=ans)
    assert result.output == (
        'Enter the choices, separated by commas with no spaces: a,b\n'
        'Enter the criteria, separated by commas with no spaces:: c,d\n'
        'Are there any more criteria that are continuous and need to be calculated? (true, false) [false]: true\n'
        'Enter the continuous criteria, separated by commas with no spaces: e,f\n'
        'Enter a weight for c: 1\n'
        'Enter a weight for d: 2\n'
        'Enter a weight for e: 5\n'
        'Enter a weight for f: 6\n'
        'Enter a score for a on c\n'
        ': 1\n'
        'Enter a score for a on d\n'
        ': 2\n'
        'Enter a score for b on c\n'
        ': 3\n'
        'Enter a score for b on d\n'
        ': 4\n'
        'Enter the requirements in the form <criterion>: <score>. Exit with :wq\n'
        '<Given this e...>: <What score should this value receive?>\n'
        'For example, if criterion is cost, then `10: 4` means if cost is $10, the item should get a 4\n\n'
        'e\n'
        '> > > > Enter the requirements in the form <criterion>: <score>. Exit with :wq\n'
        '<Given this f...>: <What score should this value receive?>\n'
        'For example, if criterion is cost, then `10: 4` means if cost is $10, the item should get a 4\n\n'
        'f\n'
        '> > > > Enter the e for a: 1\n'
        'Enter the f for a: 2\n'
        'Enter the e for b: 3\n'
        'Enter the f for b: 4\n'
        '┏━━━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━━━━━━━━━━━━━━━━┓\n'
        '┃ Choices ┃ c   ┃ d   ┃ e   ┃ f   ┃ Percentage         ┃\n'
        '┡━━━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━━━━━━━━━━━━━━━━┩\n'
        '│ Weight  │ 1.0 │ 2.0 │ 5.0 │ 6.0 │                    │\n'
        '│ a       │ 1.0 │ 2.0 │ 9.5 │ 1.2 │ 42.642857142857146 │\n'
        '│ b       │ 3.0 │ 4.0 │ 8.5 │ 2.4 │ 48.50000000000001  │\n'
        '└─────────┴─────┴─────┴─────┴─────┴────────────────────┘\n'
    )


@pytest.mark.parametrize('num', list(range(13)))  # len of args + 1
def test_insufficient_info_no_interact_fails(num):
    runner = CliRunner()
    args = (
        '-c a,b ',
        '-C c,d ',
        '-w 1,2 ',
        # Scores not here because it would be valid
        '--continuous-criteria e,f ',
        '-W 5,6 ',
        '-V 0,10,20 ',
        '-S 10,5,0 ',
        '-V 0,5,10 ',
        '-S 0,3,10 ',
        '-d a,e,1 ',
        '-d a,f,2 ',
        '-d b,e,3 ',
        '-d b,f,4 ',
    )
    result = runner.invoke(main, '-I ' + ''.join(args[:num]))
    assert result.exit_code == 2
    assert result.output == (
        'Usage: main [OPTIONS]\n'
        'Try "main --help" for help.\n\n'
        'Error: Insufficient information in non-interactive mode!\n'
        'At the minimium, choices, criteria, weights, and scores must be supplied.\n'
        'If continuous criteria is given, their weights, all value-score '
        'pairs, and all data need to be given.\n'
    )


@pytest.mark.parametrize('weights', (True, False))
@pytest.mark.parametrize('values', (True, False))
@pytest.mark.parametrize('scores', (True, False))
@pytest.mark.parametrize('data', (True, False))
def test_insufficient_args_continuous_quits(weights, values, scores, data):
    runner = CliRunner()
    args = [
        '-c a,b ',
        '-C c,d ',
        '-w 1,2 ',
        '-s 1,2,3,4 ',
        '--continuous-criteria e,f ',
    ]

    if weights:
        args += ['-W 5,6 ']

    if values:
        args += ['-V 0,10,20 ', '-V 0,5,10 ']

    if scores:
        args += ['-S 10,5,0 ','-S 0,3,10 ']

    # At least one of the four additional arguments must be false
    if data and (not weights or not values or not scores):
        args += ['-d a,e,1 ', '-d a,f,2 ', '-d b,e,3 ', '-d b,f,4 ']

    result = runner.invoke(main, '-I ' + ''.join(args))
    assert result.exit_code == 2
    assert result.output == (
        'Usage: main [OPTIONS]\n'
        'Try "main --help" for help.\n\n'
        'Error: Insufficient information in non-interactive mode!\n'
        'At the minimium, choices, criteria, weights, and scores must be supplied.\n'
        'If continuous criteria is given, their weights, all value-score '
        'pairs, and all data need to be given.\n'
    )
