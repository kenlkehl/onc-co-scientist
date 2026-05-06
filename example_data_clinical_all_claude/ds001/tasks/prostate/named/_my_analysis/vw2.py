import json
r = json.load(open(r'C:\Users\klkehl\are_llms_biased\data\ds001\tasks\prostate\named\_my_analysis\results.json'))

print('=== iter5 stratified treatment effect by binary feature value (sig only) ===')
for row in r['iter5_stratified_tx_effects_by_binary']:
    if row['p'] < 0.05:
        print(f"  {row['treatment']:25s} stratum {row['feature']}={row['feature_value']}: rate_on={row['rate_on']:.3f} (n={row['n_on']}) rate_off={row['rate_off']:.3f} (n={row['n_off']}) diff={row['diff']:+.3f} p={row['p']:.3g}")

print()
print('=== iter6 treatment x continuous interactions sorted by p (top 25) ===')
ints = [row for row in r['iter6_treatment_continuous_interactions'] if 'interaction_p' in row]
ints.sort(key=lambda x: x['interaction_p'])
for row in ints[:25]:
    print(f"  {row['treatment']:25s} x {row['feature']:25s} int_coef={row['interaction_coef']:+.6f} p={row['interaction_p']:.3g}")

print()
print('=== iter8 specific biomarker subgroups ===')
for row in r['iter8_targeted_biomarker_subgroups']:
    if row is None:
        continue
    print(f"  {row['treatment']:25s} {row['subgroup']:25s} rate_on={row['rate_on']:.3f} (n={row['n_on']}) rate_off={row['rate_off']:.3f} (n={row['n_off']}) diff={row['diff']:+.3f} p={row['p']:.3g}")
