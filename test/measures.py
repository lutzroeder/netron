#!/usr/bin/env python

""" Test Measures Script """

import logging

import pandas

pandas.set_option("display.max_rows", None)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _summarize(summary_df, measures_df, column, threshold):
    measures_df = measures_df.sort_values(column, ascending=False)
    total = measures_df[column].sum()
    column_df = measures_df[measures_df[column] > threshold]
    top = column_df[column].sum()
    percent = 100 * (top / total)
    summary_df.loc[len(summary_df)] = [ column, total, top, len(column_df), percent ]
    return column_df.to_string(index=False) + "\n"

def main():
    measures_df = pandas.read_csv("dist/test/measures.csv")
    measures_df.fillna(0, inplace=True)
    summary_df = pandas.DataFrame(columns=[ "Name", "Total", "Top", "Count", "Ratio" ])
    logger.info(_summarize(summary_df, measures_df, "load", 1))
    logger.info(_summarize(summary_df, measures_df, "validate", 1))
    logger.info(_summarize(summary_df, measures_df, "render", 1))
    logger.info(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
