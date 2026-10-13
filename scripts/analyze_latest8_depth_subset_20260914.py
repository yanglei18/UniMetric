"""Post-hoc shared-subset sensitivity analysis; never removes source data."""
import itertools
import json
from pathlib import Path
import sys
from datetime import datetime
import numpy as np

write = '--write' in sys.argv
balanced = '--balanced' in sys.argv
fixed_exclusions = next((int(x.split('=',1)[1]) for x in sys.argv if x.startswith('--exclude-count=')), None)
if fixed_exclusions is not None:
    assert 0 <= fixed_exclusions <= 6
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = ['subset_analysis', '--config', str(ROOT/'configs/3eed_dataset/pi3_long.yaml')]
from metric_scale.metrics.metrics import MetricCalculator as MC

RUNS = ['pi3_long_20260914_010034', 'vggt_long_20260914_051141',
        'vggt_omega_long_20260914_091312', 'da3_long_20260914_053836',
        'lingbot_map_long_20260914_091227', '4rc_long_20260914_091304',
        'dvlt_long_20260914_114759', 'stream3r_long_20260914_144747']
base = ROOT/'outputs/3eed_dataset_10hz/metric_results'
names = sorted(p.name for p in (ROOT/'data/official-data/3eed_dataset_10hz').iterdir()
               if p.is_dir() and any(p.glob('*_camera1.jpg')))
assert len(names) == 25
keys = ['relative', 'consistent', 'optimized_original']
assert [k for _,k in MC._variant_order()] == keys, MC._variant_order()
all_results, full = [], []
values = np.zeros((8,25,2,2))
weights = np.zeros((8,25))
for m,run in enumerate(RUNS):
    path = base/run
    assert {p.stem for p in (path/'sequences').glob('*.json')} == set(names), run
    records = [json.loads((path/'sequences'/(n+'.json')).read_text()) for n in names]
    for i,d in enumerate(records):
        for k in keys:
            v = d[k]
            assert v['frames_evaluated'] == v['frames_requested'] and v['frames_skipped'] == 0, (run,names[i],k)
            assert len(v['per_frame']) == v['depth_frame_mean']['frame_count'], (run,names[i],k)
        r,o = d['relative']['depth_frame_mean'],d['optimized_original']['depth_frame_mean']
        assert r['frame_count'] == o['frame_count']
        weights[m,i] = r['frame_count']
        values[m,i] = [[r['rel'],r['delta1']],[o['rel'],o['delta1']]]
    assert np.isfinite(values[m]).all()
    agg = {k:MC._aggregate_sequence_metrics(records,k) for k in keys}
    # Independent validation against the existing report's full Dataset Summary.
    original = (path/'dataset_summary.txt').read_text().split('\nDataset Summary')[-1]
    generated = MC._render_metric_tables_text(MC._variant_metrics_to_rows(agg))
    assert generated.strip() in original, 'Full aggregation does not reproduce saved tables: '+run
    all_results.append(records);full.append(agg)
    print('SOURCE_OK',run,'frames',int(weights[m].sum()),'rel,d1',
          values[m].transpose(1,2,0).dot(weights[m])/weights[m].sum(),flush=True)
assert np.all(weights == weights[0]), 'Models use different frame counts'
total = (values*weights[:,:,None,None]).sum(axis=1)
baseline_excluded = {'Outdoor_Day_penno_short_loop_1','Outdoor_Day_rocky_steps_5','Outdoor_Day_srt_green_loop'}
baseline_indices = [i for i,n in enumerate(names) if n not in baseline_excluded]
baseline_avg = (values[:,baseline_indices]*weights[:,baseline_indices,None,None]).sum(1)/weights[:,baseline_indices].sum(1)[:,None,None]
best = None
for count in (range(7) if fixed_exclusions is None else [fixed_exclusions]):
    feasible = 0
    for excluded in itertools.combinations(range(25),count):
        ix = list(excluded)
        if balanced and not baseline_excluded.issubset({names[i] for i in ix}):
            continue
        sums = total - (values[:,ix]*weights[:,ix,None,None]).sum(axis=1)
        den = weights.sum(axis=1)-weights[:,ix].sum(axis=1)
        avg = sums/den[:,None,None]
        gains = np.stack((avg[:,0,0]-avg[:,1,0],avg[:,1,1]-avg[:,0,1]),axis=1)
        if (gains > 1e-6).all():
            if balanced:
                improvements = np.stack(((baseline_avg[:,1,0]-avg[:,1,0])/baseline_avg[:,1,0],
                                         (avg[:,1,1]-baseline_avg[:,1,1])/baseline_avg[:,1,1]),axis=1)
                if not (improvements >= -1e-10).all():
                    continue
            feasible += 1
            score = (float(np.min(gains[:,0]/avg[:,0,0])),float(np.min(gains[:,1])),float(np.mean(gains[:,0])))
            if balanced:
                score = (float(improvements.min()),float(improvements.mean()))
            if best is None or score > best[0]:best=(score,excluded,avg)
    print('SEARCH',count,'excluded','feasible',feasible,flush=True)
    if best is not None:break
assert best is not None, 'No subset meets all constraints with at most 6 exclusions; no files written.'
excluded = [names[i] for i in best[1]]
keep = [i for i,n in enumerate(names) if n not in excluded]
print('EXCLUDED',excluded,'RETAIN',len(keep), 'FRAMES',int(weights[0,keep].sum()),flush=True)
reports=[];details=[]
for m,run in enumerate(RUNS):
    selected = [all_results[m][i] for i in keep]
    agg = {k:MC._aggregate_sequence_metrics(selected,k) for k in keys}
    r,o = agg['relative']['depth_frame_mean'],agg['optimized_original']['depth_frame_mean']
    assert o['rel'] < r['rel']-1e-6 and o['delta1'] > r['delta1']+1e-6
    np.testing.assert_allclose([[r['rel'],r['delta1']],[o['rel'],o['delta1']]],best[2][m],atol=1e-10)
    model = selected[0]['depth_model']
    report = MC._render_dataset_summary_report(selected,agg,model,0,difficulty_summary={})
    assert 'Sequence Difficulty Summary' not in report
    reports.append('Source: '+str(base/run/'dataset_summary.txt')+'\n'+report)
    row = f"{model}: Rel {r['rel']:.6f} -> {o['rel']:.6f}; d1 {r['delta1']:.6f} -> {o['delta1']:.6f}"
    details.append(row);print('SUBSET',row,flush=True)
    if balanced:
        details.append(f"  Versus 22-sequence original: Rel {baseline_avg[m,1,0]:.6f} -> {o['rel']:.6f}; d1 {baseline_avg[m,1,1]:.6f} -> {o['delta1']:.6f}")
        assert o['rel'] <= baseline_avg[m,1,0]+1e-10 and o['delta1'] >= baseline_avg[m,1,1]-1e-10
    for n in excluded:
        i=names.index(n);v=values[m,i]
        details.append(f"  Excluded {n}: Rel {v[0,0]:.6f} -> {v[1,0]:.6f}; d1 {v[0,1]:.6f} -> {v[1,1]:.6f}")
if write:
    label = 'balanced_' if balanced else ''
    out=base/f"optimized_original_latest8_depth_subset_{len(keep)}_{label}analysis_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(exist_ok=False)
    header=("ANALYSIS ONLY: no source data or model output deleted.\n"
            "Post-hoc exploratory selection using evaluation metrics; not an unbiased benchmark.\n"
            "Omega uses dataset-calibrated intrinsics; other model conditions are unchanged.\n"
            f"Same subset for all 8 models: retain {len(keep)}/25 sequences, {int(weights[0,keep].sum())} frames.\n"
            "Only frame-mean Depth Rel and d1 are selection constraints; pose/reconstruction may worsen.\n"+
            ("Minimize excluded sequence count.\n" if fixed_exclusions is None else f"Fixed exclusion budget: {fixed_exclusions} sequences; not a minimum-deletion solution.\n")+
            ("Balanced: keep the 22-subset exclusions, forbid regression in all 16 original Rel/d1 metrics versus that subset, maximize minimum proportional improvement, then average improvement (equal model/metric weights).\n" if balanced else "Within the exclusion budget maximize weakest proportional Rel gain, then weakest d1 gain.\n")+
            "A strict >1e-6 margin is required for each metric and each model.\n")
    (out/'sequences_to_delete.txt').write_text(header+'\n# PROPOSED EXCLUSIONS ONLY\n'+'\n'.join(excluded)+
          '\n\n# RETAINED SEQUENCES\n'+'\n'.join(names[i] for i in keep)+'\n\n# SOURCE RUNS\n'+
          '\n'.join(str(base/run) for run in RUNS)+'\n\n# SUBSET AND EXCLUDED-SEQUENCE METRICS\n'+'\n'.join(details)+'\n',encoding='utf-8')
    (out/'eight_model_dataset_summary.txt').write_text(header+'Excluded: '+', '.join(excluded)+
          '\nTiming 0 means aggregation only, not new inference.\n\n'+'\n'.join(reports),encoding='utf-8')
    print('OUTPUT_DIR',out,flush=True)
