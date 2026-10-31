"""Inspect local input coverage without downloading or changing market data.

Membership removal is not proof of delisting. This structural audit cannot
certify membership history, security identity, licensing or economic prices.
"""
import csv
import json
from collections import Counter
from pathlib import Path


def inspect(root=Path('.')):
    membership = next((p for p in (root/'sp500_ticker_start_end.csv',
                      root/'data/sp500_ticker_start_end.csv') if p.exists()), None)
    if membership is None:
        raise FileNotFoundError('Historical membership CSV required')
    with membership.open(encoding='utf-8-sig', newline='') as source:
        rows = list(csv.DictReader(source))
    tickers = sorted({r['ticker'].strip().upper() for r in rows})
    missing, malformed = [], []
    for ticker in tickers:
        price = root/'data'/f'{ticker}.csv'
        if not price.exists():
            missing.append(ticker)
            continue
        with price.open(encoding='utf-8-sig', newline='') as source:
            header = next(csv.reader(source), [])
        columns = {h.strip().lower().replace(' ', '_') for h in header}
        if not columns.intersection({'date', 'timestamp', 'time'}):
            malformed.append(ticker)
    counts = {}
    for date in ('2000-01-03', '2010-01-04', '2020-01-02', '2025-01-02'):
        active = [r['ticker'] for r in rows if r['start_date'] <= date <= (r['end_date'] or '2999-12-31')]
        counts[date] = {'unique': len(set(active)),
                        'duplicate_active_symbols': sum(n-1 for n in Counter(active).values() if n > 1)}
    return {'membership_rows': len(rows), 'unique_symbols': len(tickers),
            'missing_price_files': missing, 'malformed_price_headers': malformed,
            'sample_membership_counts': counts,
            'certified_historically_complete': False}


if __name__ == '__main__':
    print(json.dumps(inspect(), indent=2))
