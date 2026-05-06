import json
r = json.load(open(r'C:\Users\klkehl\are_llms_biased\data\ds001\tasks\prostate\named\_my_analysis\results.json'))

# All iter5 entries (no filter)
print('=== iter5 all stratified treatment effects (sorted by treatment, |diff|) ===')
rows = r['iter5_stratified_tx_effects_by_binary']
rows.sort(key=lambda x: (x['treatment'], -abs(x['diff'])))
for row in rows:
    print(f"  {row['treatment']:25s} {row['feature']:15s}={row['feature_value']}: rate_on={row['rate_on']:.3f} (n={row['n_on']}) rate_off={row['rate_off']:.3f} (n={row['n_off']}) diff={row['diff']:+.4f} p={row['p']:.3g}")
