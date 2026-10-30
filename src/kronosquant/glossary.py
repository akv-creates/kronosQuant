"""Plain-language definitions of the technical terms used on the dashboard and in the docs. One place, so tooltips and the glossary never disagree."""

GLOSSARY = {
    "Research and paper trading": "Everything here is a study on historical data or a simulation with made-up fills. No real orders are placed, no real money is used, and nothing here is investment advice.",
    "Forecast skill": "Whether a model's predictions are better than a simple rule such as 'tomorrow looks like today'. A forecast can look plausible and still have no skill.",
    "Information coefficient (IC)": "How well forecasts put outcomes in the right order, on a scale from -1 to +1. 0 means no skill and +1 means a perfect ranking. Real signals in markets are usually small, "
                                    "so a value like +0.05 is meaningful only if the interval around it stays above 0.",
    "99% interval": "A range of values the data cannot rule out, found by resampling the results. If the interval includes the no-skill value (0), the test could not tell skill from luck. "
                    "A wider interval means less evidence, for example because there were few forecasts.",
    "Passed / Failed / Inconclusive": "Passed: the whole interval was on the good side of the rule declared before the run. Failed: the whole interval was reliably on the wrong side. "
                                      "Inconclusive: the interval included the no-skill value, so the test could not say either way. Inconclusive is not a hidden success.",
    "Declared rule": "The pass/fail rule written into the experiment script before any result was seen. It is not edited afterwards, so results cannot be tuned to look better.",
    "Basis point (bps)": "One hundredth of one percent. 10 bps is 0.10%.",
    "Net return": "Return after trading costs (fees and slippage). 'Gross' return ignores costs, which is not what a real account would earn.",
    "Drawdown": "How far the account has fallen from its previous highest value, as a percentage. A maximum drawdown of -30% means it once lost 30% from a peak.",
    "Sharpe ratio": "Average return divided by how much returns bounce around, scaled to a year (risk-free rate taken as 0). Higher is better, but over short or noisy samples it is unreliable.",
    "Turnover": "How much of the account is traded. High turnover means costs eat more of any edge.",
    "Random walk baseline": "The simplest forecast: the price stays where it is. Any model must beat it before its other results mean anything.",
    "Validation split": "The part of the history used to judge ideas. The later 'test' split is kept sealed and is opened once, only after a validation pass.",
    "Test split": "Data held back for one final check. It has not been used in any result shown here.",
    "Sampled paths": "Kronos generates random possible futures. Averaging several (here 5) gives a steadier forecast than a single draw.",
    "Look-ahead bias (leakage)": "Using information that would not have been known at the time. Tests in this project check that no forecast or signal can see later data.",
    "Survivorship bias": "Studying only assets that still exist today makes past results look better than what an investor would really have seen.",
    "Pretraining contamination": "Kronos may already have seen the historical period it is tested on during its own training, which would flatter its results. Only the part of history after its training cutoff is a clean test.",
    "Paper account": "A simulated account that applies position, order, daily-loss and drawdown limits bar by bar. Fills are simulated; nothing is sent to an exchange.",
}


def definition(term: str) -> str:
    """The plain-language definition for a term (case-sensitive key); a KeyError if the term is not defined, so typos fail loudly."""
    return GLOSSARY[term]
