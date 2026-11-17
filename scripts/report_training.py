"""All-seed, public-safe results; retain negative generation and baselines."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];groups=[]
specs=[('round1','rig','Rig reconstruction'),('round1','geometry','Geometry rig'),('round1_balanced','rig','Balanced EOS')]
for suite,mode,name in specs:
    runs=[]
    for path in sorted((root/'results'/suite/mode).glob('seed*/evaluation_metrics_v2.json')):
        d=json.loads(path.read_text());test=d['splits']['test'];s=[r for r in test if r['skeleton_eligible']]
        scalar=dict(seed=d['config']['seed'],best_step=d['best_step'],source=str(path.relative_to(root)),
            skin_l1_element=float(np.mean([r['skin']['l1_element'] for r in test])),skin_l1_vector=float(np.mean([r['skin']['l1_vector'] for r in test])),
            skin_l2_vector=float(np.mean([r['skin']['l2_vector'] for r in test])),skin_precision=float(np.mean([r['skin']['precision'] for r in test])),skin_recall=float(np.mean([r['skin']['recall'] for r in test])),
            **{k:float(np.mean([r['skeleton'][k] for r in s])) for k in ['j2j','j2b','b2b']},
            joint_count=float(np.mean([r['generated_joint_count'] for r in s])),capped=sum(r['capped'] for r in s),test_assets=len(test),
            coordinate_accuracy=float(np.mean([r['teacher_forced']['coordinate_accuracy'] for r in s])),parent_accuracy=float(np.mean([r['teacher_forced']['parent_accuracy'] for r in s])),
            teacher_eos_recall=float(np.mean([r['teacher_forced']['eos_at_true_end'] for r in s])),
            teacher_j2j=float(np.mean([r['teacher_forced_geometry_diagnostic']['j2j'] for r in s])),
            quantized_gt_j2j=float(np.mean([r['quantized_gt_oracle']['j2j'] for r in s])),
            metric_b2b_dense_max_difference=max(r['metric_sensitivity']['b2b_abs_difference'] for r in s))
        runs.append(scalar)
    assert len(runs)==3,(suite,mode,len(runs))
    metrics={k:dict(mean=float(np.mean([r[k] for r in runs])),std=float(np.std([r[k] for r in runs],ddof=1))) for k in runs[0] if k not in ['seed','source','best_step','capped','test_assets']}
    groups.append(dict(suite=suite,mode=mode,name=name,runs=runs,metrics=metrics,capped_runs=sum(r['capped'] for r in runs),evaluations=96))
baseline=json.loads((root/'results/round1/baselines.json').read_text());base_skin=float(np.mean([r['skin']['l1_element'] for r in baseline['test']]));base_j2j=float(np.mean([r['skeleton']['j2j'] for r in baseline['test']]))
summary=dict(scope='Real Anymate subset: 96 train/16 validation/32 test; 83/14/32 eligible single-root skeletons; 2000 steps, three seeds per condition, width64/layers2, 256 train queries and 512 eval queries',
    claim='Held-out reconstruction proxy; GT rig geometry remains a conditioning input. Rig-context mode also uses GT skin weights to form context. Not image-conditioned generation or original-paper numerical reproduction.',
    selection='Each seed checkpoint selected by validation loss; all three seeds reported, no best test seed',
    uncertainty='Mean and sample standard deviation across three training seeds, not confidence intervals over original benchmark',
    metric_version=2,groups=groups,baselines=dict(nearest_bone_temperature=baseline['selected_temperature'],skin_l1_element=base_skin,surface_retrieval_j2j=base_j2j))
(root/'results/round1/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,2,figsize=(10,6),layout='constrained');labels=['Rig context','Geometry rig','Balanced EOS'];colors=['#0072B2','#009E73','#D55E00']
for i,g in enumerate(groups):
    for panel,key,scale in [(ax[0,0],'skin_l1_element',100),(ax[0,1],'j2j',100)]:
        panel.scatter(np.full(3,i)+np.linspace(-.08,.08,3),[r[key]*scale for r in g['runs']],color=colors[i],s=30)
        panel.errorbar(i,g['metrics'][key]['mean']*scale,yerr=g['metrics'][key]['std']*scale,color=colors[i],capsize=5)
    ax[1,0].bar(i,100*g['capped_runs']/96,color=colors[i],width=.5)
    ax[1,1].scatter(np.full(3,i)-.10,[100*r['teacher_j2j'] for r in g['runs']],marker='s',color=colors[i])
    ax[1,1].scatter(np.full(3,i)+.10,[100*r['j2j'] for r in g['runs']],marker='o',color=colors[i])
ax[0,0].axhline(base_skin*100,color='gray',ls='--',label='Bone-distance control');ax[0,0].legend(fontsize=8)
ax[0,1].axhline(base_j2j*100,color='gray',ls='--',label='Train-rig retrieval');ax[0,1].legend(fontsize=8)
ax[0,0].set_ylabel('Held-out skin L1 element ×100');ax[0,1].set_ylabel('Free-generation J2J ×100')
ax[1,0].set_ylabel('Generation reaching 96-joint cap (%)');ax[1,0].set_ylim(0,105)
ax[1,1].set_ylabel('J2J ×100');ax[1,1].scatter([],[],marker='s',color='gray',label='Teacher-forced diagnostic');ax[1,1].scatter([],[],marker='o',color='gray',label='Free generation');ax[1,1].legend(fontsize=8)
for a in ax.flat:a.set_xticks(range(3),labels,rotation=12)
fig.savefig(root/'results/round1/comparison.png',dpi=170);plt.close(fig)
lines=['# Original Anymate subset: all-seed results','',summary['scope'],'',summary['claim'],'',
    'Distances and L1 below are multiplied by 100 for readability. Mean ± sample SD across three seeds. They are not comparable to image-conditioned paper tables.','',
    '| Condition | Skin element L1 ×100 | J2J ×100 | J2B ×100 | B2B ×100 | Cap hits /96 |','|---|---:|---:|---:|---:|---:|']
for g in groups:
    vals=[f"{g['metrics'][k]['mean']*100:.3f} ± {g['metrics'][k]['std']*100:.3f}" for k in ['skin_l1_element','j2j','j2b','b2b']]
    lines.append('| '+g['name']+' | '+' | '.join(vals)+f" | {g['capped_runs']}/96 |")
lines+=['',f'Bone-distance control: skin element L1 ×100 = {base_skin*100:.3f}; train-rig surface retrieval: J2J ×100 = {base_j2j*100:.3f}. These are local controls, not the paper baselines.','',
    'The unweighted-stop settings reach the cap frequently. Balanced EOS is an explicitly changed diagnostic loss; its improvement in stopping does not establish improved geometry. No unsuccessful run is replaced by a favorable seed.','',
    'Skinning evaluation uses supplied GT bone geometry and point correspondence. The rig reconstruction context also contains GT-skin-derived pooled features. No full image-to-asset generation, learned Sparse VAE, geometry decoder or paper table is reproduced.','',
    'Original raw files, pointwise predictions and pretrained weights remain local and ignored by Git. Configs, traces, hashes, exclusions and per-asset scalar metrics are retained.']
(root/'results/round1/REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary['baselines']))
