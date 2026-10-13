"""Append a fixed Scal3R run to two existing subsets; no reselection/deletion."""
import json
from pathlib import Path
import re
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = ['subset_extension', '--config', str(ROOT/'configs/3eed_dataset/pi3_long.yaml')]
from metric_scale.metrics.metrics import MetricCalculator as MC

base = ROOT/'outputs/3eed_dataset_10hz/metric_results'
scal_run = 'scal3r_long_20260914_172242'
sources = [
    'optimized_original_latest8_depth_subset_20_balanced_analysis_20260914_171209',
    'optimized_original_latest8_depth_subset_20_analysis_20260914_170406',
]
keys = ['relative','consistent','optimized_original']
assert [key for _,key in MC._variant_order()] == keys
cache = {}
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
for source in sources:
    original = (base/source/'eight_model_dataset_summary.txt').read_text()
    listing = (base/source/'sequences_to_delete.txt').read_text()
    excluded = listing.split('# PROPOSED EXCLUSIONS ONLY\n')[1].split('\n\n')[0].splitlines()
    retained = listing.split('# RETAINED SEQUENCES\n')[1].split('\n\n')[0].splitlines()
    assert len(excluded)==5 and len(retained)==20 and not set(excluded)&set(retained)
    runs = [Path(p).parent.name for p in re.findall(r'^Source: (.+)$',original,re.M)]
    assert len(runs)==8 and len(set(runs))==8 and scal_run not in runs
    runs.append(scal_run)
    reports,checks,details = [],[],[]
    expected_frames = None
    for run in runs:
        if run not in cache:
            run_dir = base/run
            records = {p.stem:json.loads(p.read_text()) for p in (run_dir/'sequences').glob('*.json')}
            assert set(records)==set(excluded+retained), run
            for name,d in records.items():
                for key in keys:
                    v=d[key]
                    assert v['frames_evaluated']==v['frames_requested'] and v['frames_skipped']==0,(run,name,key)
                    assert len(v['per_frame'])==v['depth_frame_mean']['frame_count'],(run,name,key)
            full = {key:MC._aggregate_sequence_metrics(list(records.values()),key) for key in keys}
            tables = MC._render_metric_tables_text(MC._variant_metrics_to_rows(full))
            assert tables.strip() in (run_dir/'dataset_summary.txt').read_text().split('\nDataset Summary')[-1],run
            cache[run]=records
            print('SOURCE_VALIDATED',run,flush=True)
        records = cache[run]
        selected = [records[n] for n in retained]
        agg = {key:MC._aggregate_sequence_metrics(selected,key) for key in keys}
        frames=agg['relative']['frames_requested']
        if expected_frames is None: expected_frames=frames
        assert expected_frames==frames
        model=selected[0]['depth_model']
        report=MC._render_dataset_summary_report(selected,agg,model,0,difficulty_summary={})
        assert 'Sequence Difficulty Summary' not in report
        if run!=scal_run:
            assert report.strip() in original,'Original eight-model subset changed: '+run
        reports.append('Source: '+str(base/run/'dataset_summary.txt')+'\n'+report)
        r,o=agg['relative']['depth_frame_mean'],agg['optimized_original']['depth_frame_mean']
        ok=o['rel']<r['rel']-1e-6 and o['delta1']>r['delta1']+1e-6
        checks.append(ok)
        details.append(f"{model}: Rel {r['rel']:.6f} -> {o['rel']:.6f}; d1 {r['delta1']:.6f} -> {o['delta1']:.6f}; {'PASS' if ok else 'FAIL'}")
        if run==scal_run:print('SCAL_SUBSET',source,details[-1],flush=True)
    label='balanced_' if '_balanced_' in source else ''
    out=base/f'optimized_original_latest9_depth_subset_20_{label}analysis_{timestamp}'
    header=("ANALYSIS ONLY: no data or model outputs deleted.\n"
            "Post-hoc exploratory subset; not an unbiased full-dataset benchmark.\n"
            "Omega uses dataset-calibrated intrinsics.\n"
            f"Source subset: {base/source}\n"
            "Original exclusion/retention list and eight source runs preserved exactly; Scal3R appended, no reselection.\n"
            "Balanced label (if present) describes the original eight-model selection, not a newly optimized nine-model subset.\n"
            f"Retained: 20/25 sequences, {expected_frames} frames.\n"
            f"Depth Rel lower AND d1 higher than relative with >1e-6 margin: {sum(checks)}/9 models.\n"
            "Pose/reconstruction are reported, not constrained. Timing 0 means aggregation only.\n")
    out.mkdir(exist_ok=False)
    (out/'sequences_to_delete.txt').write_text(header+'\n# PROPOSED EXCLUSIONS ONLY\n'+'\n'.join(excluded)+
        '\n\n# RETAINED SEQUENCES\n'+'\n'.join(retained)+'\n\n# SOURCE RUNS\n'+'\n'.join(str(base/r) for r in runs)+
        '\n\n# SUBSET METRIC CHECKS\n'+'\n'.join(details)+'\n',encoding='utf-8')
    (out/'nine_model_dataset_summary.txt').write_text(header+'\n'+'\n'.join(reports),encoding='utf-8')
    print('OUTPUT',out,'PASS',sum(checks),'FRAMES',expected_frames,flush=True)
