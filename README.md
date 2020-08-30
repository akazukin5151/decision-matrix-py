[![docs](https://readthedocs.org/projects/decision-matrix-py/badge/?version=latest)](https://decision-matrix-py.readthedocs.io/en/latest/?badge=latest)

# Decision matrix

> Helps making decisions between multiple choices and against multiple weighted criteria (MCDM)

Struggling to make a decision? Have multiple criteria but some of them are more important than others? The number of choices are too much for you to handle in your mind? Then you should use a decision matrix!

## What does this do?

It helps you construct a decision matrix for multiple choices ('alternatives' in literature) and multiple weighted criteria. You can either manually rate each combination, or use a linear interpolating function to calculate the rating.

Examples of things you might need to pick between:

* A new phone to buy
* A new laptop
* The best location for a new house
* The best university course you should pick
* The best company offers you should accept

Examples of criteria you might be considering:
* For a new laptop:
    * Price
    * Speed \*
    * Storage
    * Battery
    * Screen
* For accommodation:
    * Rent
    * Location \*
    * Transportation
    * Food

\* Note that some criteria should be split up. See docs for more

A decision matrix is a table; each choice gets one row and each criteria gets one column. The combination of cells between them is your rating for this choice against this criteria. The percentage column calculates, for each choice, how close they are to the ideal choice. A higher percentage is better.

For example, in choosing between Phone A and Phone B, the criteria might be price and battery life. Four ratings need to be done, two for the price of Phone A and Phone B, and two for the battery life of Phone A and Phone B.

## Why use this?

* Fast -- abstractation over pandas, numpy, scipy, and matplotlib
* Allows access to the internal pandas dataframes for further end-stage processing
* Programmatic way to create decision matrices
* Code is a portable and readable text-based format, unlike Excel.
* Use the pythonic API interface or the fluent API interface, at your choice.
* CLI interface for quick usage

## Example

Simple example

```py
import matrix
m = matrix.Matrix()
m.add_choices('Laptop A', 'Laptop B')
m.add_criteria('price', 'cpu', 'ram', weights=(9, 8, 7.5))
m.rate_choice('Laptop A', price=5, cpu=9, ram=8)
m.rate_choice('Laptop B', price=7, cpu=7, ram=5)
print(m)

    |          |   price |   cpu |   ram | Percentage        |
    |:---------|--------:|------:|------:|:------------------|
    | Weight   |       9 |     8 |   7.5 |                   |
    | Laptop A |       5 |     9 |   8   | 72.24489795918367 |
    | Laptop B |       7 |     7 |   5   | 63.87755102040816 |
```

Fluent API example

```py
import matrix
m = matrix.Matrix('Tokyo', 'Hong Kong', 'London')
m.add_continuous_criterion('population', weight=5)
m.if_(population=7e6).then(score=5)
m.if_(population=2e7).then(score=10)
print(m.value_score_df)

       population  population_score
    0   7000000.0                 5
    1  20000000.0                10

m.add_data('Tokyo', population=13_929_280)
m.add_data('Hong Kong', {'population': 7_500_700})
m.add_data('London', population=8_961_989)
# Note that the population scores are automatically calculated
print(m)

    |           |   population | Percentage         |
    |:----------|-------------:|:-------------------|
    | Weight    |      5       |                    |
    | Tokyo     |      7.66511 | 76.65107692307693  |
    | Hong Kong |      5.19258 | 51.925769230769234 |
    | London    |      5.75461 | 57.546111538461545 |
```

Don't like the fluent API? No problem, just use this.

```py
import matrix
m = matrix.Matrix('Tokyo', 'Hong Kong', 'London')
m.add_continuous_criterion('population', weight=5)
m.criterion_value_to_score('population', {7e6: 5, 2e7: 10})
print(m.value_score_df)

       population  population_score
    0   7000000.0                 5
    1  20000000.0                10

m.batch_add_data({
    'Tokyo': {'population': 13_929_280},
    'Hong Kong': {'population': 7_500_700},
    'London': {'population': 8_961_989},
})
print(m)

    |           |   population | Percentage         |
    |:----------|-------------:|:-------------------|
    | Weight    |      5       |                    |
    | Tokyo     |      7.66511 | 76.65107692307693  |
    | Hong Kong |      5.19258 | 51.925769230769234 |
    | London    |      5.75461 | 57.546111538461545 |
```

### CLI usage

This package also includes a cli interface. If you intend to use it, install Click and Rich as well.

```
❯ python matrix/cli.py --choices a,b --criteria c,d --weights 1,2 --ratings 1,2,3,4 -I
┏━━━━━━━━━┳━━━━━┳━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Choices ┃ c   ┃ d   ┃ Percentage         ┃
┡━━━━━━━━━╇━━━━━╇━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ Weight  │ 1.0 │ 2.0 │                    │
│ a       │ 1.0 │ 2.0 │ 16.666666666666664 │
│ b       │ 3.0 │ 4.0 │ 36.666666666666664 │
└─────────┴─────┴─────┴────────────────────┘
```


## Installation

**Requirements**

* Python 3
* Pandas
* Numpy
* Scipy
* Matplotlib
* Click (optional, cli only)
* Rich (optional, cli only)

**From source**
* git clone
* `python setup.py install`
