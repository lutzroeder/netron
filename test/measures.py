#!/usr/bin/env python

''' Test Measures Script '''

import pandas  # pylint: disable=import-error

pandas.set_option('display.max_rows', None)

def _summarize(summary_df, measures_df, column, threshold):
    measures_df = measures_df.sort_values(column, ascending=False)
    total = measures_df[column].sum()
    column_df = measures_df[measures_df[column] > threshold]
    top = column_df[column].sum()
    summary_df.loc[len(summary_df)] = [ column, total, top, len(column_df), top / total ]
    return column_df.to_string(index=False) + '\n'

def main(): # pylint: disable=missing-function-docstring
    measures_df = pandas.read_csv('dist/test/measures.csv')
    measures_df.fillna(0, inplace=True)
    summary_df = pandas.DataFrame(columns=[ 'Name', 'Total', 'Top', 'Count', 'Percentage' ])
    print(_summarize(summary_df, measures_df, 'load', 2))
    print(_summarize(summary_df, measures_df, 'validate', 2))
    print(_summarize(summary_df, measures_df, 'render', 2))
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()
