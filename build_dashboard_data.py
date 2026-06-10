"""
build_dashboard_data.py
=======================
대시보드용 JSON 생성.
실행: python build_dashboard_data.py
출력: dashboard_data.json
"""
import pandas as pd, numpy as np, yfinance as yf
import requests, os, json, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def _load_env():
    for f in ['.env.local','.env']:
        if os.path.exists(f):
            for line in open(f):
                line=line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k,v=line.split('=',1)
                    os.environ.setdefault(k.strip(),v.strip())
            break
_load_env()
KEY  = os.getenv('FRED_API_KEY','')
STA  = '1971-01-01'

FRED = {
    'vix':'VIXCLS','credit_spread':'BAA10Y','hy_spread':'BAMLH0A0HYM2',
    'fin_stress':'STLFSI4','yield_curve':'T10Y2Y','yield_3m10y':'T10Y3M',
    'real_rate':'DFII10','ted_spread':'TEDRATE','fedfunds':'FEDFUNDS',
    'fed_assets':'WALCL','initial_claims':'ICSA',
    'consumer_sent':'UMCSENT','lei':'USALOLITONOSTSAM',
    'm2':'M2SL','cpi':'CPIAUCSL',
}

def ffred(sid):
    url='https://api.stlouisfed.org/fred/series/observations'
    p={'series_id':sid,'api_key':KEY,'file_type':'json',
       'observation_start':STA,'observation_end':datetime.today().strftime('%Y-%m-%d'),
       'sort_order':'asc'}
    try:
        r=requests.get(url,params=p,timeout=20).json()
        if 'observations' not in r: return pd.Series(dtype=float)
        s=pd.Series({o['date']:float(o['value']) if o['value'] not in ['.',''] else np.nan
                     for o in r['observations']},dtype=float)
        s.index=pd.to_datetime(s.index); return s
    except: return pd.Series(dtype=float)

def collect():
    print('📡 데이터 수집...')
    raw={}
    for k,sid in FRED.items():
        raw[k]=ffred(sid); print(f'  ✅ {k}')
    print('  ⬇️  SP500...')
    df=yf.download('^GSPC',start=STA,interval='1wk',progress=False,auto_adjust=True)
    cl=df['Close']
    if isinstance(cl,pd.DataFrame): cl=cl.iloc[:,0]
    sp=cl.squeeze(); sp.index=pd.to_datetime(sp.index)
    return raw,sp

def build(raw,sp):
    W={}
    def wr(s,lag=0):
        if s is None or s.empty: return pd.Series(dtype=float)
        ws=s.resample('W-FRI').last().ffill()
        return ws.shift(lag) if lag>0 else ws
    for k in ['vix','credit_spread','hy_spread','fin_stress',
               'yield_curve','yield_3m10y','real_rate','ted_spread','fedfunds']:
        s=raw.get(k,pd.Series())
        if s.empty: continue
        ws=wr(s); W[k]=ws
        for p in [1,2,4,8,13]: W[f'{k}_{p}w']=ws.diff(p)
    for k,lag in [('fed_assets',0),('initial_claims',0),
                  ('consumer_sent',2),('lei',4),('m2',4),('cpi',4)]:
        s=raw.get(k,pd.Series())
        if s.empty: continue
        ws=wr(s,lag); W[f'{k}_yoy']=ws.pct_change(52)*100; W[f'{k}_4w']=ws.diff(4)
    ff=wr(raw.get('fedfunds',pd.Series()))
    if not ff.empty:
        W['rate_hiking'] =(ff.diff(13)>0.25).astype(float)*100
        W['rate_cutting']=(ff.diff(13)<-0.25).astype(float)*100
    W['sp500']=sp
    for p in [1,2,4,8,13,26,52]: W[f'sp500_{p}w']=sp.pct_change(p)*100
    df=pd.concat(W.values(),axis=1,keys=W.keys())
    return df.sort_index()

def pct_roll(series,window=260):
    arr=series.values.astype(float); n=len(arr); out=np.full(n,np.nan)
    for i in range(52,n):
        hist=arr[max(0,i-window):i]; valid=hist[~np.isnan(hist)]
        if len(valid)<20: continue
        v=arr[i]
        if np.isnan(v): continue
        out[i]=float(np.mean(valid<v))*100
    return pd.Series(out,index=series.index)

def build_pct(df):
    print('📐 퍼센타일...')
    excl=['sp500']+[f'sp500_{p}w' for p in [1,2,4,8,13,26,52]]
    cols=[c for c in df.columns if c not in excl]
    pct={c:pct_roll(df[c]) for c in cols}
    print(f'  {len(pct)}개 완료'); return pd.DataFrame(pct)

SELL_W=[
    ('vix','high',0.1240),('vix_2w','high',0.0219),
    ('credit_spread','high',0.1661),('credit_spread_2w','high',0.1316),
    ('hy_spread','high',0.0706),('hy_spread_2w','high',0.1450),
    ('fin_stress','high',0.0590),('ted_spread','high',0.1160),
    ('rate_hiking','high',0.1000),('yield_curve_4w','low',0.0658),
]
BUY_W=[
    ('vix','low',0.1365),('vix_4w','low',0.0925),
    ('credit_spread','low',0.0277),('credit_spread_4w','low',0.0834),
    ('hy_spread','low',0.2146),('hy_spread_4w','low',0.0870),
    ('rate_cutting','high',0.0958),('fed_assets_4w','high',0.0806),
    ('initial_claims_4w','low',0.1160),('consumer_sent_yoy','high',0.0659),
]

def sc(row,W):
    s=0.0; ws=0.0
    for col,d,w in W:
        if col not in row.index: continue
        v=row[col]
        if pd.isna(v): continue
        s+=(v if d=='high' else 100-v)*w; ws+=w
    return round(s/ws,2) if ws>=0.3 else np.nan

def compute_scores(pct):
    print('🧮 점수 계산...')
    sell=[sc(pct.iloc[i],SELL_W) for i in range(len(pct))]
    buy =[sc(pct.iloc[i],BUY_W)  for i in range(len(pct))]
    return pd.DataFrame({'sell':sell,'buy':buy},index=pct.index)

def build_daily_nowcast(raw):
    """최근 90거래일 일별 점수 — 매일 갱신되는 지표 기반 나우캐스트"""
    print('📅 일별 나우캐스트...')
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=2200)  # 약 6년
    W={}
    def dr(s, lag_days=0):
        if s is None or s.empty: return None
        s=s[s.index>=cutoff]
        if s.empty: return None
        ds=s.resample('D').last().ffill()
        return ds.shift(lag_days) if lag_days>0 else ds
    for k in ['vix','credit_spread','hy_spread','fin_stress','ted_spread','fedfunds','yield_curve']:
        ds=dr(raw.get(k))
        if ds is None: continue
        W[k]=ds
        W[f'{k}_2w']=ds.diff(14); W[f'{k}_4w']=ds.diff(28)
    for k,lag in [('fed_assets',0),('initial_claims',0),('consumer_sent',14)]:
        ds=dr(raw.get(k),lag)
        if ds is None: continue
        W[f'{k}_yoy']=ds.pct_change(364)*100
        W[f'{k}_4w']=ds.diff(28)
    ff=W.get('fedfunds')
    if ff is not None:
        W['rate_hiking'] =(ff.diff(91)> 0.25).astype(float)*100
        W['rate_cutting']=(ff.diff(91)<-0.25).astype(float)*100
    if not W:
        print('  ⚠ 일별 데이터 없음'); return []
    df=pd.concat(list(W.values()),axis=1,keys=list(W.keys())).sort_index()

    # ffill로 만들어진 '가짜 오늘' 방지: 핵심 일간지표의 실제 마지막 관측일까지만
    real_dates=[raw[k].index.max() for k in ['vix','hy_spread','credit_spread']
                if k in raw and not raw[k].empty]
    if real_dates:
        df=df[df.index<=max(real_dates)]

    n=len(df)
    if n<400: return []
    eval_n=min(130,n)
    win=1825  # 5년(일)
    out_cols={}
    for c in df.columns:
        arr=df[c].values.astype(float)
        o=np.full(n,np.nan)
        for i in range(n-eval_n,n):
            hist=arr[max(0,i-win):i]
            valid=hist[~np.isnan(hist)]
            if len(valid)<200: continue
            v=arr[i]
            if np.isnan(v): continue
            o[i]=float(np.mean(valid<v))*100
        out_cols[c]=o
    dpct=pd.DataFrame(out_cols,index=df.index)

    # 일별 주가
    pxd={}
    try:
        dfp=yf.download('^GSPC',
            start=str((pd.Timestamp.today()-pd.Timedelta(days=220)).date()),
            interval='1d',progress=False,auto_adjust=True)
        cl=dfp['Close']
        if isinstance(cl,pd.DataFrame): cl=cl.iloc[:,0]
        for idx,v in cl.items():
            pxd[str(pd.Timestamp(idx).date())]=float(v)
    except: pass

    rows=[]
    for i in range(n-eval_n,n):
        d=df.index[i]
        if d.weekday()>=5: continue
        prow=dpct.iloc[i]
        s=sc(prow,SELL_W); b=sc(prow,BUY_W)
        if pd.isna(s) or pd.isna(b): continue
        ds=str(d.date())
        p=pxd.get(ds)
        rows.append({'date':ds,
                     'sell':round(float(s),1),'buy':round(float(b),1),
                     'price':round(p,0) if p else None})
    rows=rows[-90:]
    print(f'  {len(rows)}거래일 완료')
    return rows

def hit_rates(scores,sp,horizons=[1,2,3,4,5,6,8]):
    """각 점수구간×시점별 적중률"""
    print('📊 적중률 계산...')
    spr=sp.reindex(scores.index).ffill()
    bands=[(0,40,'낮음'),(40,55,'보통'),(55,65,'주의'),(65,75,'경고'),(75,100,'위험')]
    res={'sell':{},'buy':{}}
    for h in horizons:
        fwd=spr.pct_change(h).shift(-h)*100
        res['sell'][h]=[]; res['buy'][h]=[]
        for lo,hi,lbl in bands:
            # SELL
            m=(scores['sell']>=lo)&(scores['sell']<hi)&fwd.notna()
            if m.sum()>20:
                dp=(fwd[m]<-2).mean()*100; ar=fwd[m].mean(); n=int(m.sum())
            else: dp=ar=None; n=0
            res['sell'][h].append({'band':lbl,'lo':lo,'hi':hi,
                'down_prob':round(dp,1) if dp is not None else None,
                'avg_ret':round(ar,2) if ar is not None else None,'n':n})
            # BUY
            m=(scores['buy']>=lo)&(scores['buy']<hi)&fwd.notna()
            if m.sum()>20:
                up=(fwd[m]>2).mean()*100; ar=fwd[m].mean(); n=int(m.sum())
            else: up=ar=None; n=0
            res['buy'][h].append({'band':lbl,'lo':lo,'hi':hi,
                'up_prob':round(up,1) if up is not None else None,
                'avg_ret':round(ar,2) if ar is not None else None,'n':n})
    return res

def timeline(scores,sp,pct):
    print('📈 타임라인...')
    rec=scores.loc['2020-01-01':]
    spr=sp.reindex(rec.index).ffill()
    pr=pct.reindex(rec.index)
    out=[]
    for date,row in rec.iterrows():
        s=row['sell']; b=row['buy']
        price=float(spr.get(date,np.nan))
        contribs={}
        if not pd.isna(s):
            prow=pr.loc[date] if date in pr.index else pd.Series()
            for col,d,w in SELL_W:
                if col not in prow.index: continue
                v=prow[col]
                if pd.isna(v): continue
                contribs[col]=round(float((v if d=='high' else 100-v)*w),1)
        out.append({'date':str(date.date()),
            'sell':round(float(s),1) if not pd.isna(s) else None,
            'buy': round(float(b),1) if not pd.isna(b) else None,
            'sp500':round(price,0) if not pd.isna(price) else None,
            'sell_contributions':contribs})
    return out

def ind_stats(pct,scores,sp):
    print('🔬 지표 예측력...')
    spr=sp.reindex(scores.index).ffill()
    fwd4=spr.pct_change(4).shift(-4)*100
    out=[]
    seen=set()
    for col,d,w,stype in [(c,d,w,'sell') for c,d,w in SELL_W]+\
                          [(c,d,w,'buy')  for c,d,w in BUY_W]:
        if col not in pct.columns or col in seen: continue
        seen.add(col)
        pv=pct[col].dropna()
        q75=pv.quantile(0.75); q25=pv.quantile(0.25)
        hr=fwd4[pct[col]>=q75].mean(); lr=fwd4[pct[col]<=q25].mean()
        pp=abs(hr-lr) if not (pd.isna(hr) or pd.isna(lr)) else None
        out.append({'indicator':col,'direction':d,'weight':w,'score_type':stype,
            'high_q_ret':round(float(hr),2) if not pd.isna(hr) else None,
            'low_q_ret': round(float(lr),2) if not pd.isna(lr) else None,
            'predictive_power':round(float(pp),2) if pp is not None else None})
    return sorted(out,key=lambda x:x['predictive_power'] or 0,reverse=True)

def current_signal(scores,pct):
    last=scores.iloc[-1]
    sell=float(last['sell']) if not pd.isna(last['sell']) else None
    buy =float(last['buy'])  if not pd.isna(last['buy'])  else None
    prow=pct.iloc[-1]
    sd=[]; bd=[]
    for col,d,w in SELL_W:
        if col not in prow.index: continue
        v=prow[col]
        if pd.isna(v): continue
        danger=v if d=='high' else 100-v
        sd.append({'indicator':col,'direction':d,'weight':w,
            'percentile':round(float(v),1),'danger':round(float(danger),1),
            'contrib':round(float(danger*w),2)})
    for col,d,w in BUY_W:
        if col not in prow.index: continue
        v=prow[col]
        if pd.isna(v): continue
        safety=v if d=='low' else 100-v
        bd.append({'indicator':col,'direction':d,'weight':w,
            'percentile':round(float(v),1),'safety':round(float(safety),1),
            'contrib':round(float(safety*w),2)})
    sig=('SELL' if sell and sell>=72 else 'BUY' if buy and buy>=65 else 'HOLD')
    act={'SELL':'다음 주 월요일 전량 매도 검토','BUY':'다음 주 월요일 매수 검토',
         'HOLD':'현 포지션 유지 — 아무것도 하지 마세요'}[sig]
    return {'date':str(scores.index[-1].date()),
        'sell_score':round(sell,1) if sell else None,
        'buy_score': round(buy,1)  if buy  else None,
        'signal':sig,'action':act,
        'sell_detail':sorted(sd,key=lambda x:x['danger'],reverse=True),
        'buy_detail': sorted(bd,key=lambda x:x['contrib'],reverse=True)}

def sp_chart(scores,sp):
    rec=sp.loc['2009-01-01':]
    scr=scores.reindex(rec.index)
    out=[]
    for date,price in rec.items():
        if pd.isna(price): continue
        row=scr.loc[date] if date in scr.index else pd.Series()
        out.append({'date':str(date.date()),'price':round(float(price),0),
            'sell':round(float(row['sell']),1) if not row.empty and not pd.isna(row.get('sell',np.nan)) else None,
            'buy': round(float(row['buy']),1)  if not row.empty and not pd.isna(row.get('buy', np.nan)) else None})
    return out

def run():
    print('='*50+'\n  Dashboard Data Builder\n'+'='*50)
    raw,sp=collect()
    df=build(raw,sp)
    pct=build_pct(df)
    scores=compute_scores(pct)
    data={
        'current':      current_signal(scores,pct),
        'hit_rates':    {k:{str(k2):v2 for k2,v2 in v.items()} for k,v in hit_rates(scores,sp).items()},
        'timeline':     timeline(scores,sp,pct),
        'indicator_stats': ind_stats(pct,scores,sp),
        'sp_chart':     sp_chart(scores,sp),
        'daily_nowcast': build_daily_nowcast(raw),
        'generated_at': datetime.now().isoformat(),
    }
    with open('dashboard_data.json','w',encoding='utf-8') as f:
        json.dump(data,f,indent=2,ensure_ascii=False,default=str)
    print(f'\n✅ dashboard_data.json 완료')
    print(f'  타임라인 {len(data["timeline"])}주 / SP차트 {len(data["sp_chart"])}주')
    print(f'  현재신호: {data["current"]["signal"]} | sell={data["current"]["sell_score"]} buy={data["current"]["buy_score"]}')

if __name__=='__main__':
    run()
