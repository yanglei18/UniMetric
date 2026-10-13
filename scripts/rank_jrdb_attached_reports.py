import json
import re
from pathlib import Path
import numpy as np

files = [Path('C:/Users/Lenovo/Downloads') / f'dataset_summary ({i}).txt' for i in (3, 4, 5)]
records = []
datasets = set()
for path in files:
    text = path.read_text(encoding='utf-8-sig')
    datasets.update(re.findall(r'(?m)^relative_scale dir: .*?/outputs/([^/]+)/', text))
    for block in re.split(r'(?m)^Sequence: ', text)[1:]:
        name = block.splitlines()[0].strip()
        model = re.search(r'(?m)^Model: (.+)', block).group(1).strip()
        # Stop at the end of this sequence, before appended dataset aggregates.
        block = block.split('Metric geometry calculation time:')[0]
        table = None
        rows = {v: {} for v in ('relative_scale', 'optimized_scale_original')}
        for line in block.splitlines():
            if not line.startswith('|'):
                continue
            cells = [c.strip().replace('*', '') for c in line.strip('|').split('|')]
            if cells[0] == 'Scale':
                table = cells
            elif cells[0] in rows:
                for key, value in zip(table[1:], cells[1:]):
                    if '↓' in key or '↑' in key:
                        rows[cells[0]][key] = float(value.split()[0])
        before, after = rows.values()
        assert len(before) == len(after) == 17, (name, model, rows)
        gains = {}
        for key, b in before.items():
            a = after[key]
            # Accuracy is converted to failure rate; avoids division by tiny d1.
            if '↑' in key:
                b, a = 1-b, 1-a
            gains[key] = (b-a)/b if b else (0 if not a else float('-inf'))
        groups = {
            'depth': [k for k in gains if k.startswith('Depth')],
            'geometry': [k for k in gains if not k.startswith(('Depth', 'ATE', 'RPE'))],
            'pose': [k for k in gains if k.startswith(('ATE', 'RPE')) and 'Rot' not in k],
        }
        task_scores = {g: float(np.mean([gains[k] for k in ks])) for g, ks in groups.items()}
        records.append(dict(sequence=name, model=model, before=before, after=after, gains=gains,
                            tasks=task_scores, score=float(np.mean(list(task_scores.values()))),
                            flat_score=float(np.mean([v for k,v in gains.items() if 'Rot' not in k])),
                            improved=sum(v>0 for v in gains.values()), worsened=sum(v<0 for v in gains.values())))
models = sorted(set(r['model'] for r in records))
common = set.intersection(*[{r['sequence'] for r in records if r['model']==m} for m in models])
ranking = []
for seq in common:
    rs = [r for r in records if r['sequence']==seq]
    ranking.append(dict(sequence=seq, score=float(np.mean([r['score'] for r in rs])),
                        flat_score=float(np.mean([r['flat_score'] for r in rs])),
                        worst_model=min(r['score'] for r in rs), improved=sum(r['improved'] for r in rs),
                        worsened=sum(r['worsened'] for r in rs)))
ranking.sort(key=lambda r:r['score'], reverse=True)
assert len(datasets) == 1, datasets
out = Path(__file__).resolve().parents[1]/('analysis_outputs/' + next(iter(datasets)).lower() + '_three_model_selection')
out.mkdir(parents=True, exist_ok=True)
(out/'ranking.json').write_text(json.dumps(dict(method='Equal models; equal depth, geometry, pose tasks; mean error reduction, d1 as failure rate; rotation checked but excluded from scale score',
    sources=[str(p) for p in files], ranking=ranking, records=records), indent=2), encoding='utf-8')
print('COUNTS', {m:sum(r['model']==m for r in records) for m in models}, 'COMMON',len(common))
print('JOINT TOP',json.dumps(ranking[:5],indent=2))
for m in models:
    best=sorted([r for r in records if r['model']==m],key=lambda r:r['score'],reverse=True)
    print('MODEL TOP',m,[(r['sequence'],r['score'],r['worsened']) for r in best[:3]])
print('FLAT TOP', sorted(ranking,key=lambda r:r['flat_score'],reverse=True)[:3])
for r in records:
    if r['sequence']==ranking[0]['sequence']:
        print('WINNER DETAIL',json.dumps(r,indent=2))
