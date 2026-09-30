"""PMF-truth false positives / negatives of R2 combos from t2c_r2v3 jobs.

python t2c_r2v3_fpfn.py JOBS_DIR OUT.json
"""
import json,glob,numpy as np,sys
jobs=[json.load(open(p)) for p in glob.glob(sys.argv[1] + '/*.json')]
K=['any|fixed20|2.0|0.5','any|fixed20|2.0|0.25','any|g5|2.0|0.25','same-column|g5|1.0|0.25','same-column|g5|2.0|0.5','same-column|g5|2.0|0.25','same-column|g5|3.0|0.25']
out={}
for k in K:
  row={}
  for cell in ['indep|2000','indep|8000','re|2000','re|8000']:
    reg,n=cell.split('|'); fp=tp=mid=0; need=hit=0
    for j in jobs:
      if j['regime']!=reg: continue
      for c in j['r2'][n]['columns']:
        props=[p for p in c['combos'][k] if p['decision']=='proposed']
        for p in props:
          e=p['max_err_kT'] or 0.0
          if e<=0.2: fp+=1
          elif e>0.5: tp+=1
          else: mid+=1
        # columns whose worst heavy-interval error > 0.5 kT: was any proposal made there?
        er=np.array([np.nan if x is None else x for x in c['err_kT']]); fr=np.array(c['frac'])
        heavy=(fr>=0.02)&np.isfinite(er)
        if heavy.any() and np.nanmax(er[heavy])>0.5:
          need+=1; hit+=int(any((p['max_err_kT'] or 0)>0.5 for p in props))
    row[cell]=dict(prop_err_le_0p2=fp,prop_err_0p2_0p5=mid,prop_err_gt_0p5=tp,cols_err_gt_0p5=need,of_them_hit=hit)
  out[k]=row
  print(k); [print('  ',c,v) for c,v in row.items()]
json.dump(out,open(sys.argv[2],'w'),indent=1)
