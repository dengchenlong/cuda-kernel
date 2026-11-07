import csv, statistics, sys
p = sys.argv[1] if len(sys.argv) > 1 else 'bench_results.csv'
rows = list(csv.DictReader(open(p, encoding='utf-8-sig')))
s = sorted(float(x['speedup']) for x in rows)
print('n', len(s), 'min', s[0], 'p10', s[len(s)//10], 'median', round(statistics.median(s), 3), 'max', s[-1])
loses = [(x['op'], x['width'], x['height'], x['param'], x['speedup']) for x in rows if float(x['speedup']) <= 1.0]
print('loses', loses)
