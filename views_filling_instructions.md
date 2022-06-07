In this code library, views can be given for mean, volatility and correlation. Views can be absolute or relative, and they can be equalities or inequalities. All numbers are to be given in the same units as the scenarios data is in. Correlation is a dimensionless quantity. Each view is filled as its own row, and the number of rows is only limited by Excel.

*Minor mathematical note: The **strict** inequality signs in the Excel act as ≤ and ≥ in the program. Do not use <= or >= in the Excel sheet.*

## Mean values
**Example:** We want to set the mean of *Eurozone Core Inflation* to 1 % (annual). This is filled as follows,

| * View on | * Risk factor 1         | Risk factor 2       (applicable for corr) | * Operator | * Constant       (alpha) | Multiplier       (beta) | Risk factor 3 | Risk factor 4       (applicable for corr) |
|-----------|-------------------------|-------------------------------------------|------------|--------------------------|-------------------------|---------------|-------------------------------------------|
| Mean      | Eurozone Core Inflation |                                           |          = |            1           |                         |               |                                           |


By leaving blank cells (or dash **-**), the program knows that we're dealing with an absolute view.

**Example:** *Eurozone Core Inflation* is at least 1 % (annual) greater than *US Core Inflation*. This is filled as follows,

| * View on | * Risk factor 1         | Risk factor 2       (applicable for corr) | * Operator | * Constant       (alpha) | Multiplier       (beta) | Risk factor 3 | Risk factor 4       (applicable for corr) |
|-----------|-------------------------|-------------------------------------------|------------|--------------------------|-------------------------|---------------|-------------------------------------------|
| Mean      | Eurozone Core Inflation |                                           |          > |            1           |       1                 | US Core Inflation  |                                           |

If a given multiplier is a dash (**-**), or the cell is left blank, the multiplier is interpreted as 1. The multiplier acts on the *Risk factor 3* (and *Risk factor 4* in the case of a correlation view). In mathematical terms,

<img src="https://render.githubusercontent.com/render/math?math=\mu \text{(Risk factor 1)} - \beta \mu \text{(Risk factor 3)} = \alpha .">

The equality sign can be changed to < or > if needed.

## Volatility
Filling volatility views is analogous to filling mean values. The left-most column is changed from *Mean* to *Vol*. The volatilities are also filled with annual units.

## Correlation
Only with correlation, are *Risk factor 2* and *Risk factor 4* used.

**Example:** The correlation between *Eurozone Core Inflation* and *US Core Inflation* is greater than 0.8. This is accomplished below,

| * View on | * Risk factor 1         | Risk factor 2       (applicable for corr) | * Operator | * Constant       (alpha) | Multiplier       (beta) | Risk factor 3 | Risk factor 4       (applicable for corr) |
|-----------|-------------------------|-------------------------------------------|------------|--------------------------|-------------------------|---------------|-------------------------------------------|
| Corr      | Eurozone Core Inflation |  US Core Inflation         |          > |            0.8           |                         |   |                                           |

## Possible errors
The following cases must be **satisfied**

* The view rows should not lead to contradictions (e.g., rows (*Eurozone Core Inflation*) = 1 %, and (*Eurozone Core Inflation*) = 2 % would lead to a contradiction)

* Volatility is always positive, or zero.

* Correlation only gets values from −1 to 1.

* With correlation view, (*Risk factor 1*) ≠ (*Risk factor 2*) and (*Risk factor 3*) ≠ (*Risk factor 4*).

* With any relative view, (*Risk factor 1*) ≠ (*Risk factor 3*) and with relative correlation, (*Risk factor 2*) ≠ (*Risk factor 4*).