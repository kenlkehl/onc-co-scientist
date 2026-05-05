import json
r = json.load(open(r'C:\Users\klkehl\are_llms_biased\data\ds001\tasks\prostate\named\_my_analysis\results.json'))

print('=== iter2 multivar logit treatments and binary features ===')
for row in r['iter2_multivar_logit']:
    if row['var'].startswith('treatment_') or row['var'] in ['mcrpc','visceral_mets','brca2_mutation','ar_v7_positive','msi_high','psma_high'] or row['p'] < 0.001:
        print(f"  {row['var']:30s} coef={row['coef']:+.4f} OR={row['or']:.3f} p={row['p']:.3g}")

print()
print('=== iter3 binary feature main effects ===')
for row in r['iter3_feature_main_effects'][:6]:
    print(f"  {row['feature']:30s} rate_pos={row.get('rate_pos',0):.3f} rate_neg={row.get('rate_neg',0):.3f} diff={row['diff']:+.3f} p={row['p']:.3g}")

print()
print('=== iter3 continuous main effects sorted by p ===')
cont = [r2 for r2 in r['iter3_feature_main_effects'] if 'mean_resp' in r2]
cont.sort(key=lambda x: x['p'])
for row in cont:
    print(f"  {row['feature']:30s} mean_R={row['mean_resp']:.3f} mean_NR={row['mean_nonresp']:.3f} diff={row['diff']:+.3f} p={row['p']:.3g}")

print()
print('=== iter4 treatment x binary interactions sorted by p ===')
ints = [row for row in r['iter4_treatment_binary_feature_interactions'] if 'interaction_p' in row]
ints.sort(key=lambda x: x['interaction_p'])
for row in ints[:30]:
    print(f"  {row['treatment']:25s} x {row['feature']:18s} int_coef={row['interaction_coef']:+.3f} OR={row['interaction_or']:.3f} p={row['interaction_p']:.3g}")
