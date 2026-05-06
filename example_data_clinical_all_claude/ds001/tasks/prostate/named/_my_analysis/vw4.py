import json
r = json.load(open(r'C:\Users\klkehl\are_llms_biased\data\ds001\tasks\prostate\named\_my_analysis\results.json'))

print('=== iter7 top 2-feature subgroups for enzalutamide ===')
for row in r['iter7_two_binary_feature_subgroups_top10']['treatment_enzalutamide']:
    print(f"  {row['f1']}={row['v1']} & {row['f2']}={row['v2']}: rate_on={row['rate_on']:.3f} (n={row['n_on']}) rate_off={row['rate_off']:.3f} (n={row['n_off']}) diff={row['diff']:+.3f} p={row['p']:.3g}")

print()
print('=== iter7 top subgroups for olaparib (any direction of effect) ===')
for row in r['iter7_two_binary_feature_subgroups_top10']['treatment_olaparib']:
    print(f"  {row['f1']}={row['v1']} & {row['f2']}={row['v2']}: rate_on={row['rate_on']:.3f} (n={row['n_on']}) rate_off={row['rate_off']:.3f} (n={row['n_off']}) diff={row['diff']:+.3f} p={row['p']:.3g}")

print()
print('=== iter7 top subgroups for pembrolizumab ===')
for row in r['iter7_two_binary_feature_subgroups_top10']['treatment_pembrolizumab']:
    print(f"  {row['f1']}={row['v1']} & {row['f2']}={row['v2']}: rate_on={row['rate_on']:.3f} (n={row['n_on']}) rate_off={row['rate_off']:.3f} (n={row['n_off']}) diff={row['diff']:+.3f} p={row['p']:.3g}")

print()
print('=== iter7 top subgroups for lu177_psma ===')
for row in r['iter7_two_binary_feature_subgroups_top10']['treatment_lu177_psma']:
    print(f"  {row['f1']}={row['v1']} & {row['f2']}={row['v2']}: rate_on={row['rate_on']:.3f} (n={row['n_on']}) rate_off={row['rate_off']:.3f} (n={row['n_off']}) diff={row['diff']:+.3f} p={row['p']:.3g}")

print()
print('=== iter7 top subgroups for abiraterone ===')
for row in r['iter7_two_binary_feature_subgroups_top10']['treatment_abiraterone']:
    print(f"  {row['f1']}={row['v1']} & {row['f2']}={row['v2']}: rate_on={row['rate_on']:.3f} (n={row['n_on']}) rate_off={row['rate_off']:.3f} (n={row['n_off']}) diff={row['diff']:+.3f} p={row['p']:.3g}")

print()
print('=== iter7 top subgroups for docetaxel ===')
for row in r['iter7_two_binary_feature_subgroups_top10']['treatment_docetaxel']:
    print(f"  {row['f1']}={row['v1']} & {row['f2']}={row['v2']}: rate_on={row['rate_on']:.3f} (n={row['n_on']}) rate_off={row['rate_off']:.3f} (n={row['n_off']}) diff={row['diff']:+.3f} p={row['p']:.3g}")

print()
print('=== iter13 top 3-feature subgroups for enzalutamide ===')
for row in r['iter13_three_binary_subgroups_top10']['treatment_enzalutamide']:
    print(f"  {row['f1']}={row['v1']} & {row['f2']}={row['v2']} & {row['f3']}={row['v3']}: rate_on={row['rate_on']:.3f} (n={row['n_on']}) rate_off={row['rate_off']:.3f} (n={row['n_off']}) diff={row['diff']:+.3f} p={row['p']:.3g}")

print()
print('=== iter15 final interaction logit (key terms) ===')
for row in r['iter15_final_interaction_logit']:
    print(f"  {row['var']:25s} coef={row['coef']:+.4f} OR={row['or']:.3f} p={row['p']:.3g}")

print()
print('=== iter9 olaparib within BRCA2+ refinement ===')
for row in r['iter9_olaparib_within_brca2_refinement']['binary']:
    print(f"  modifier={row['modifier']}={row['modifier_value']}: olap_on rate={row['rate_on']:.3f} (n={row['n_on']}) off rate={row['rate_off']:.3f} (n={row['n_off']}) diff={row['diff']:+.3f} p={row['p']:.3g}")

print()
print('=== iter11 lu177 within psma_high refinement ===')
for row in r['iter11_lu177_within_psmahigh_refinement']:
    print(f"  modifier={row['modifier']}={row['modifier_value']}: rate_on={row['rate_on']:.3f} (n={row['n_on']}) rate_off={row['rate_off']:.3f} (n={row['n_off']}) diff={row['diff']:+.3f} p={row['p']:.3g}")

print()
print('=== iter14 continuous splits stratified (significant only, p<0.01) ===')
for row in r['iter14_continuous_split_stratified']:
    if row['p'] < 0.01 and abs(row['diff']) > 0.02:
        print(f"  {row['treatment']:25s} {row['split']:30s}: rate_on={row['rate_on']:.3f} (n={row['n_on']}) rate_off={row['rate_off']:.3f} (n={row['n_off']}) diff={row['diff']:+.3f} p={row['p']:.3g}")
