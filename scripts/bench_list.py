import csv, sys
p = sys.argv[1]
rows = [r for r in csv.DictReader(open(p, encoding='utf-8-sig')) if r.get('speedup')]
for r in sorted(rows, key=lambda r: float(r['speedup'])):
    print('%22s %5sx%-5s %-6s mine=%9s cv=%9s sp=%.3f' % (
        r['op'], r['width'], r['height'], r['param'], r['mine_ms'], r['opencv_ms'], float(r['speedup'])))
