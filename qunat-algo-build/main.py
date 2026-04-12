import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import yfinance as yf
from datetime import datetime, timedelta

np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
TICKER          = "^NSEI"
INITIAL_CAPITAL = 1_000_000
TC              = 0.0005
SLIPPAGE        = 0.0002
RISK_FREE       = 0.065
KELLY_CAP       = 1.0
HURST_TREND     = 0.52
HURST_MR        = 0.44
VOL_FILTER_PCT  = 93
SCORE_LONG      = 2.0
SCORE_SHORT     = 3.8
CONFIRM_DAYS    = 2
MIN_HOLD        = 5
MAX_HOLD        = 15
TRADING_DAYS    = 252

BG=      "#0D1117"
PANEL_BG="#161B22"
BORDER=  "#30363D"
TEXT=    "#E6EDF3"
MUTED=   "#8B949E"
ACCENT=  "#58A6FF"
GREEN=   "#3FB950"
RED=     "#F85149"
GOLD=    "#D29922"
PURPLE=  "#BC8CFF"
ORANGE=  "#FFA657"
CYAN=    "#39D353"


# ── Data ──────────────────────────────────────────────────────────────────────

def _synthetic(start, end):
    print("[DATA] Using calibrated synthetic Nifty 50 data ...")
    am,av,ar = 0.13,0.175,0.08
    bdays = pd.bdate_range(start,end); n=len(bdays)
    dt=1/TRADING_DAYS; sd=av*np.sqrt(dt)
    eps=np.random.normal(0,1,n); ae=np.zeros(n); ae[0]=eps[0]
    for i in range(1,n): ae[i]=ar*ae[i-1]+eps[i]*np.sqrt(1-ar**2)
    jmp=np.where(np.random.rand(n)<0.015,np.random.normal(-0.02,0.04,n),0)
    lr=am*dt-0.5*av**2*dt+sd*ae+jmp
    C=8000*np.exp(np.cumsum(lr))
    iv=av*np.sqrt(dt)*np.random.uniform(0.5,2,n)
    O=C*np.exp(np.random.normal(0,iv*0.3,n))
    H=np.maximum(O,C)*np.exp(np.abs(np.random.normal(0,iv*0.5,n)))
    L=np.minimum(O,C)*np.exp(-np.abs(np.random.normal(0,iv*0.5,n)))
    V=(200_000_000*np.random.lognormal(0,0.5,n)).astype(int)
    df=pd.DataFrame({"Open":O,"High":H,"Low":L,"Close":C,"Volume":V},index=bdays)
    df["High"]=df[["Open","High","Close"]].max(axis=1)
    df["Low"] =df[["Open","Low","Close"]].min(axis=1)
    return df.round(2)


def download_data():
    end=datetime.today(); start=end-timedelta(days=365*10+5)
    print(f"[DATA] Downloading {TICKER} ({start.date()} to {end.date()}) ...")
    df=pd.DataFrame()
    try:
        raw=yf.download(TICKER,start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"),interval="1d",progress=False)
        if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
        raw=raw[["Open","High","Low","Close","Volume"]].dropna()
        if len(raw)>500: df=raw.copy(); print(f"[DATA] Live: {len(df)} rows")
        else: raise ValueError("Too few rows")
    except Exception as e:
        print(f"[DATA] Live unavailable ({e})")
        df=_synthetic(start,end)
    df.dropna(inplace=True)
    df.index=pd.to_datetime(df.index)
    df=df[df["Volume"]>0].sort_index()
    print(f"[DATA] Shape:{df.shape} | {df.index[0].date()} to {df.index[-1].date()}")
    print(df.head().to_string()); print()
    df["log_ret"]=np.log(df["Close"]/df["Close"].shift(1))
    return df


# ── Features ──────────────────────────────────────────────────────────────────

def hurst_dfa(series):
    n=len(series)
    if n<20: return 0.5
    try:
        x=np.cumsum(series-series.mean())
        scales=np.unique(np.logspace(np.log10(4),np.log10(n//4),12).astype(int))
        flucts=[]
        for s in scales:
            ns=n//s
            if ns<2: continue
            f2=[]
            for k in range(ns):
                seg=x[k*s:(k+1)*s]; t=np.arange(s)
                trend=np.polyval(np.polyfit(t,seg,1),t)
                f2.append(np.mean((seg-trend)**2))
            flucts.append(np.sqrt(np.mean(f2)))
        if len(flucts)<4: return 0.5
        h,_=np.polyfit(np.log(scales[:len(flucts)]),np.log(flucts),1)
        return float(np.clip(h,0.05,0.95))
    except: return 0.5


def compute_features(df):
    f=df.copy()
    O,H,L,C,V=f["Open"],f["High"],f["Low"],f["Close"],f["Volume"]
    pC=C.shift(1)
    f["daily_ret"]   =(C-O)/(O+1e-9)
    f["intraday_rng"]=(H-L)/(O+1e-9)
    f["gap"]         =(O-pC)/(pC+1e-9)
    f["body_ratio"]  =np.abs(C-O)/(H-L+1e-9)
    f["upper_wick"]  =(H-np.maximum(O,C))/(H-L+1e-9)
    f["lower_wick"]  =(np.minimum(O,C)-L)/(H-L+1e-9)
    f["gap_dir"]     =np.sign(f["gap"])
    f["ret_5d"] =C/C.shift(5)-1
    f["ret_10d"]=C/C.shift(10)-1
    f["ret_20d"]=C/C.shift(20)-1
    f["ret_30d"]=C/C.shift(30)-1
    f["realized_vol"]=f["log_ret"].rolling(10).std()*np.sqrt(TRADING_DAYS)
    vm,vs=V.rolling(10).mean(),V.rolling(10).std()
    f["vol_zscore"]=(V-vm)/(vs+1e-9)
    f["parkinson_vol"]=np.sqrt((1/(4*np.log(2)))*np.log(H/L)**2)
    roll_min=C.rolling(20).min(); roll_max=C.rolling(20).max()
    f["range_z"]=(C-roll_min)/(roll_max-roll_min+1e-9)
    var1=f["log_ret"].rolling(5).var()
    vark=f["log_ret"].rolling(5).sum().rolling(5).var()
    f["vr"]=vark/(var1*5+1e-12)
    print("[FEATURE] Computing rolling Hurst (DFA, 90d) ...")
    lr_arr=f["log_ret"].values; h_arr=np.full(len(f),0.5)
    for i in range(90,len(f)): h_arr[i]=hurst_dfa(lr_arr[i-90:i])
    f["hurst"]=h_arr
    f.dropna(inplace=True)
    return f


# ── Signals ───────────────────────────────────────────────────────────────────

def generate_signals(f):
    s=f.copy()
    pvol_thresh=np.nanpercentile(s["parkinson_vol"].values,VOL_FILTER_PCT)

    m20=np.clip(s["ret_20d"]/0.025,-2,2)
    m10=np.clip(s["ret_10d"]/0.015,-2,2)
    m30=np.clip(s["ret_30d"]/0.035,-2,2)
    m5 =np.clip(s["ret_5d"] /0.010,-1,1)

    trend=pd.Series(0.0,index=s.index)
    trend+=m20*1.5; trend+=m10*1.0; trend+=m30*0.8; trend+=m5*0.5
    trend+=np.where(s["body_ratio"]>0.6, np.sign(s["daily_ret"])*0.4, 0.0)
    trend+=np.where(s["vol_zscore"]>1.5, np.sign(s["daily_ret"])*0.5, 0.0)
    trend+=np.where(s["vr"]>1.1, np.sign(s["ret_10d"])*0.4, 0.0)
    trend+=np.where(s["range_z"]>0.70, 0.4, np.where(s["range_z"]<0.30,-0.4,0.0))

    mr=pd.Series(0.0,index=s.index)
    mr+=np.where(s["range_z"]<0.12,2.5,np.where(s["range_z"]<0.22,1.5,
        np.where(s["range_z"]<0.30,0.8,0.0)))
    mr+=np.where(s["lower_wick"]>0.50,1.0,np.where(s["lower_wick"]>0.40,0.6,0.0))
    mr+=np.where(s["ret_5d"]<-0.025,1.0,np.where(s["ret_5d"]<-0.015,0.5,0.0))
    mr+=np.where((s["vol_zscore"]>1.5)&(s["daily_ret"]<0),0.5,0.0)

    raw_score=np.where(s["hurst"]>HURST_TREND,trend.values,
              np.where(s["hurst"]<HURST_MR,mr.values,0.0))
    raw_sig=np.where(raw_score>= SCORE_LONG,  1.0,
            np.where(raw_score<=-SCORE_SHORT, -1.0, 0.0))
    raw_sig=np.where(s["parkinson_vol"]>pvol_thresh,0.0,raw_sig)

    rs=pd.Series(raw_sig,index=s.index); confirmed=rs*0.0
    for i in range(CONFIRM_DAYS-1,len(rs)):
        w=rs.iloc[i-CONFIRM_DAYS+1:i+1]
        if (w==1.0).all(): confirmed.iloc[i]=1.0
        elif (w==-1.0).all(): confirmed.iloc[i]=-1.0

    s["raw_score"]=raw_score; s["signal"]=confirmed
    lr=s["log_ret"]; wm=lr>0
    aw=lr[wm].mean() if wm.sum()>0 else 0.01
    al=lr[~wm].abs().mean() if (~wm).sum()>0 else 0.01
    wr=wm.mean()
    kf=float(np.clip(wr/(al+1e-9)-(1-wr)/(aw+1e-9),0.2,KELLY_CAP))
    rv=s["realized_vol"]/(s["realized_vol"].mean()+1e-9)
    s["pos_size"]=(kf/(rv+1e-9)).clip(upper=KELLY_CAP)*np.abs(confirmed)
    return s


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(s):
    """Event-driven backtest. Costs charged once at entry, once at exit (2x total)."""
    n=len(s); opens=s["Open"].values; closes=s["Close"].values
    sigs=s["signal"].values; sizes=s["pos_size"].values; dates=s.index
    portfolio=np.full(n,np.nan); benchmark=np.full(n,np.nan)
    portfolio[0]=benchmark[0]=INITIAL_CAPITAL
    capital=INITIAL_CAPITAL; bm_capital=INITIAL_CAPITAL
    trade_log=[]; in_trade=False; pos_dir=0; entry_px=0.0
    entry_cap=0.0; entry_date=None; hold_count=0

    for i in range(1,n):
        bm_capital*=(1+(closes[i]-closes[i-1])/(closes[i-1]+1e-9))
        benchmark[i]=bm_capital
        prev_sig=sigs[i-1]

        if in_trade:
            hold_count+=1
            exit_now=False
            if hold_count>=MIN_HOLD:
                if prev_sig==-pos_dir or prev_sig==0 or hold_count>=MAX_HOLD:
                    exit_now=True
            if exit_now:
                exit_px=opens[i]
                raw_ret=pos_dir*(exit_px-entry_px)/(entry_px+1e-9)
                net_ret=raw_ret-(TC+SLIPPAGE)   # exit side only (entry already charged)
                pnl_inr=entry_cap*net_ret
                capital+=pnl_inr; capital=max(capital,1.0)
                trade_log.append({
                    "entry_date" :str(entry_date)[:10],
                    "exit_date"  :str(dates[i])[:10],
                    "signal"     :int(pos_dir),
                    "entry_price":round(float(entry_px),2),
                    "exit_price" :round(float(exit_px),2),
                    "pnl_pct"   :round(float(net_ret*100),4),
                    "pnl_inr"   :round(float(pnl_inr),2),
                    "holding_days":hold_count,
                })
                in_trade=False; pos_dir=0; hold_count=0

        if not in_trade and prev_sig!=0:
            sz=float(sizes[i-1])
            entry_cap=min(capital*sz,capital)
            capital-=entry_cap*(TC+SLIPPAGE)   # entry side cost
            capital=max(capital,1.0)
            entry_px=opens[i]; entry_date=dates[i]
            pos_dir=int(prev_sig); in_trade=True; hold_count=0

        portfolio[i]=capital

    portfolio=pd.Series(portfolio,index=s.index).ffill().fillna(INITIAL_CAPITAL)
    benchmark=pd.Series(benchmark,index=s.index).ffill().fillna(INITIAL_CAPITAL)
    r=s.copy()
    r["portfolio_value"]=portfolio.values; r["benchmark_value"]=benchmark.values
    r["strategy_returns"]=portfolio.pct_change().fillna(0)
    r["benchmark_returns"]=benchmark.pct_change().fillna(0)
    return r, pd.DataFrame(trade_log)


# ── Metrics ───────────────────────────────────────────────────────────────────

def dd_series(eq):
    pk=np.maximum.accumulate(eq); return (eq-pk)/(pk+1e-9)

def max_dd_dur(dd):
    m=c=0
    for v in dd: c=c+1 if v<0 else 0; m=max(m,c)
    return m

def omega_ratio(ret,thr=0.0):
    e=ret-thr; g=e[e>0].sum(); l=-e[e<0].sum(); return g/(l+1e-9)

def sortino_r(ret):
    rf=RISK_FREE/TRADING_DAYS; down=ret[ret<rf]
    s=down.std(ddof=1)*np.sqrt(TRADING_DAYS) if len(down)>1 else 1e-9
    return float(np.clip((ret.mean()-rf)*TRADING_DAYS/(s+1e-9),-10,10))

def max_consec(wins):
    mw=ml=cw=cl=0
    for w in wins:
        if w: cw+=1; cl=0; mw=max(mw,cw)
        else: cl+=1; cw=0; ml=max(ml,cl)
    return mw,ml

def safe_sharpe(ret):
    v=ret.std(ddof=1)*np.sqrt(TRADING_DAYS)
    if v<1e-6: return 0.0
    return float(np.clip((ret.mean()-RISK_FREE/TRADING_DAYS)*TRADING_DAYS/v,-5,5))


def compute_metrics(r,trades):
    sr=r["strategy_returns"].values; br=r["benchmark_returns"].values
    pv=r["portfolio_value"].values;  bv=r["benchmark_value"].values
    ny=len(sr)/TRADING_DAYS

    def cagr(f,i,y): return (f/i)**(1/y)-1 if y>0 else 0
    def av(ret): return ret.std(ddof=1)*np.sqrt(TRADING_DAYS)
    def var95(ret): return np.percentile(ret,5)
    def cvar95(ret):
        v=var95(ret); sub=ret[ret<=v]; return sub.mean() if len(sub)>0 else v

    dd_s=dd_series(pv); dd_b=dd_series(bv)
    r2=r.copy(); r2.index=pd.to_datetime(r2.index)
    ann_s=r2["strategy_returns"].resample("YE").apply(lambda x:(1+x).prod()-1)
    ann_b=r2["benchmark_returns"].resample("YE").apply(lambda x:(1+x).prod()-1)

    m={}
    for tag,ret,eq,dd,ann in [
        ("strategy",sr,pv,dd_s,ann_s),("benchmark",br,bv,dd_b,ann_b)]:
        cg=cagr(eq[-1],eq[0],ny)
        m[tag]={
            "total_return":(eq[-1]/eq[0]-1)*100,"cagr":cg*100,
            "best_year":ann.max()*100,"worst_year":ann.min()*100,
            "avg_ann_ret":ann.mean()*100,"ann_vol":av(ret)*100,
            "max_dd":dd.min()*100,"max_dd_dur":max_dd_dur(dd),
            "avg_dd":dd[dd<0].mean()*100 if (dd<0).sum()>0 else 0,
            "var95":var95(ret)*100,"cvar95":cvar95(ret)*100,
            "downside_dev":ret[ret<0].std(ddof=1)*np.sqrt(TRADING_DAYS)*100
                           if (ret<0).sum()>1 else 0,
            "sharpe":safe_sharpe(ret),"sortino":sortino_r(ret),
            "calmar":cg/(abs(dd.min())+1e-9),"omega":omega_ratio(ret),
            "skew":stats.skew(ret),"kurt":stats.kurtosis(ret),
            "hurst":hurst_dfa(ret[~np.isnan(ret)]),"autocorr":pd.Series(ret).autocorr(lag=1),
        }
    active=sr-br; te=active.std(ddof=1)*np.sqrt(TRADING_DAYS)
    m["strategy"]["info_ratio"]=float(np.clip(active.mean()*TRADING_DAYS/(te+1e-9),-10,10))
    m["benchmark"]["info_ratio"]=0.0

    if len(trades)>0:
        w=trades["pnl_pct"]>0; mw,ml=max_consec(w.values)
        m["trades"]={
            "n_trades":len(trades),
            "n_long":int((trades["signal"]==1).sum()),
            "n_short":int((trades["signal"]==-1).sum()),
            "win_rate":w.mean()*100,
            "avg_win":trades.loc[w,"pnl_pct"].mean() if w.sum()>0 else 0,
            "avg_loss":trades.loc[~w,"pnl_pct"].mean() if (~w).sum()>0 else 0,
            "profit_factor":trades.loc[w,"pnl_pct"].sum()/(-trades.loc[~w,"pnl_pct"].sum()+1e-9),
            "expectancy":trades["pnl_pct"].mean(),"avg_hold":trades["holding_days"].mean(),
            "max_w":mw,"max_l":ml,
        }
    else:
        m["trades"]={k:0 for k in ["n_trades","n_long","n_short","win_rate","avg_win",
                                    "avg_loss","profit_factor","expectancy","avg_hold","max_w","max_l"]}
    return m


# ── Visualisations ────────────────────────────────────────────────────────────

def _ax(ax,title="",xl="",yl=""):
    ax.set_facecolor(PANEL_BG); ax.tick_params(colors=TEXT,labelsize=8)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER); sp.set_linewidth(0.8)
    ax.grid(True,color=BORDER,alpha=0.45,linewidth=0.5,zorder=0)
    if title: ax.set_title(title,color=GOLD,fontsize=10,fontweight="bold",pad=8)
    if xl: ax.set_xlabel(xl,color=MUTED,fontsize=8)
    if yl: ax.set_ylabel(yl,color=MUTED,fontsize=8)


def plot_metrics_dashboard(m):
    """Full metrics rendered as a dark visual figure — no console text."""
    s,b,t=m["strategy"],m["benchmark"],m["trades"]
    fig=plt.figure(figsize=(20,12),facecolor=BG)
    fig.suptitle("NIFTY 50 QUANT SYSTEM v3.0  |  PERFORMANCE DASHBOARD",
                 color=TEXT,fontsize=15,fontweight="bold",y=0.98,fontfamily="monospace")

    ax_hdr=fig.add_axes([0.01,0.920,0.98,0.040])
    ax_hdr.set_facecolor("#1C2333"); ax_hdr.set_xticks([]); ax_hdr.set_yticks([])
    for sp in ax_hdr.spines.values(): sp.set_edgecolor(BORDER)
    for x,lbl,col in [(0.17,"METRIC",MUTED),(0.52,"STRATEGY",ACCENT),(0.78,"BENCHMARK",ORANGE)]:
        ax_hdr.text(x,0.5,lbl,transform=ax_hdr.transAxes,color=col,
                    fontsize=10,fontweight="bold",fontfamily="monospace",va="center",ha="center")

    sections=[
        ("RETURN METRICS",[
            ("Total Return (%)",   f"{s['total_return']:+.2f}", f"{b['total_return']:+.2f}"),
            ("CAGR (%)",           f"{s['cagr']:+.2f}",         f"{b['cagr']:+.2f}"),
            ("Best Year (%)",      f"{s['best_year']:+.2f}",    f"{b['best_year']:+.2f}"),
            ("Worst Year (%)",     f"{s['worst_year']:+.2f}",   f"{b['worst_year']:+.2f}"),
            ("Avg Annual (%)",     f"{s['avg_ann_ret']:+.2f}",  f"{b['avg_ann_ret']:+.2f}"),
        ]),
        ("RISK METRICS",[
            ("Ann. Volatility (%)", f"{s['ann_vol']:.2f}",       f"{b['ann_vol']:.2f}"),
            ("Max Drawdown (%)",    f"{s['max_dd']:.2f}",        f"{b['max_dd']:.2f}"),
            ("Max DD Dur. (days)",  f"{s['max_dd_dur']}",        f"{b['max_dd_dur']}"),
            ("Avg Drawdown (%)",    f"{s['avg_dd']:.2f}",        f"{b['avg_dd']:.2f}"),
            ("VaR 95% (%)",         f"{s['var95']:.4f}",         f"{b['var95']:.4f}"),
            ("CVaR 95% (%)",        f"{s['cvar95']:.4f}",        f"{b['cvar95']:.4f}"),
            ("Downside Dev (%)",    f"{s['downside_dev']:.4f}",  f"{b['downside_dev']:.4f}"),
        ]),
        ("RISK-ADJUSTED & STATS",[
            ("Sharpe Ratio",       f"{s['sharpe']:.4f}",        f"{b['sharpe']:.4f}"),
            ("Sortino Ratio",      f"{s['sortino']:.4f}",       f"{b['sortino']:.4f}"),
            ("Calmar Ratio",       f"{s['calmar']:.4f}",        f"{b['calmar']:.4f}"),
            ("Omega Ratio",        f"{s['omega']:.4f}",         f"{b['omega']:.4f}"),
            ("Info. Ratio",        f"{s['info_ratio']:.4f}",    "    —"),
            ("Skewness",           f"{s['skew']:.4f}",          f"{b['skew']:.4f}"),
            ("Kurtosis",           f"{s['kurt']:.4f}",          f"{b['kurt']:.4f}"),
            ("Hurst Exponent",     f"{s['hurst']:.4f}",         f"{b['hurst']:.4f}"),
            ("Autocorr (lag-1)",   f"{s['autocorr']:.4f}",      f"{b['autocorr']:.4f}"),
        ]),
        ("TRADE EXECUTION",[
            ("Total Trades",       f"{t['n_trades']}",          "    —"),
            ("  Long Trades",      f"{t['n_long']}",            "    —"),
            ("  Short Trades",     f"{t['n_short']}",           "    —"),
            ("Win Rate (%)",       f"{t['win_rate']:.2f}",      "    —"),
            ("Avg Win (%)",        f"{t['avg_win']:.4f}",       "    —"),
            ("Avg Loss (%)",       f"{t['avg_loss']:.4f}",      "    —"),
            ("Profit Factor",      f"{t['profit_factor']:.4f}", "    —"),
            ("Expectancy (%/tr)",  f"{t['expectancy']:.4f}",    "    —"),
            ("Avg Hold (days)",    f"{t['avg_hold']:.2f}",      "    —"),
            ("Max Consec. Wins",   f"{t['max_w']}",             "    —"),
            ("Max Consec. Loss",   f"{t['max_l']}",             "    —"),
        ]),
    ]

    n_cols=len(sections); gap=0.010
    col_w=(0.98-gap*(n_cols-1))/n_cols; top=0.910; bot=0.02

    for ci,(title,rows) in enumerate(sections):
        x0=0.01+ci*(col_w+gap)
        ax=fig.add_axes([x0,bot,col_w,top-bot])
        ax.set_facecolor(PANEL_BG); ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER); sp.set_linewidth(0.9)
        ax.add_patch(plt.Rectangle((0,0.958),1,0.042,fc="#1C2333",ec=BORDER,lw=0.7,transform=ax.transAxes))
        ax.text(0.5,0.979,title,transform=ax.transAxes,color=GOLD,fontsize=9,
                fontweight="bold",fontfamily="monospace",va="center",ha="center")
        ax.text(0.04,0.942,"Metric",  color=MUTED, fontsize=7.5,fontfamily="monospace",transform=ax.transAxes,va="top")
        ax.text(0.64,0.942,"Strat",   color=ACCENT,fontsize=7.5,fontfamily="monospace",transform=ax.transAxes,va="top",ha="center")
        ax.text(0.88,0.942,"Bench",   color=ORANGE,fontsize=7.5,fontfamily="monospace",transform=ax.transAxes,va="top",ha="center")
        ax.axhline(0.928, color=BORDER, lw=0.7, xmin=0.02, xmax=0.98)
        n_r=len(rows); row_h=0.895/(n_r+0.5)
        for ri,(lbl,sv,bv) in enumerate(rows):
            y=0.920-ri*row_h
            if ri%2==0:
                ax.add_patch(plt.Rectangle((0.005,y-row_h*0.55),0.99,row_h*0.92,
                                           fc="#1A2233",ec="none",transform=ax.transAxes,clip_on=True))
            lbl_col=MUTED if lbl.startswith("  ") else TEXT
            ax.text(0.04,y-row_h*0.05,lbl,color=lbl_col,fontsize=7.8,fontfamily="monospace",
                    transform=ax.transAxes,va="center")
            try:
                n_sv=float(sv.replace("+","").replace(",",""))
                sv_col=GREEN if n_sv>0 else (RED if n_sv<0 else TEXT)
            except: sv_col=TEXT
            ax.text(0.64,y-row_h*0.05,sv,color=sv_col,fontsize=7.8,fontweight="bold",
                    fontfamily="monospace",transform=ax.transAxes,va="center",ha="center")
            try:
                n_bv=float(bv.strip().replace("+","").replace(",","").replace("—","0"))
                bv_col=GREEN if n_bv>0 else (RED if n_bv<0 else MUTED)
            except: bv_col=MUTED
            ax.text(0.88,y-row_h*0.05,bv,color=bv_col,fontsize=7.8,fontfamily="monospace",
                    transform=ax.transAxes,va="center",ha="center")
        ax.add_patch(plt.Rectangle((0,0),1,0.003,fc=GOLD,ec="none",transform=ax.transAxes))

    plt.show(); print("[PLOT 0] Metrics Dashboard ✓")


def plot1_equity_curve(r):
    try:
        fig=plt.figure(figsize=(15,9),facecolor=BG)
        gs=gridspec.GridSpec(3,1,figure=fig,hspace=0.06,height_ratios=[3,1,1])
        ax1=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1],sharex=ax1)
        ax3=fig.add_subplot(gs[2],sharex=ax1)
        for ax in [ax1,ax2,ax3]: ax.set_facecolor(PANEL_BG)
        fig.suptitle("EQUITY CURVE  |  Strategy vs Nifty 50 Benchmark",
                     color=TEXT,fontsize=13,fontweight="bold",y=0.99)
        d=r.index
        ns=r["portfolio_value"]/r["portfolio_value"].iloc[0]*100
        nb=r["benchmark_value"]/r["benchmark_value"].iloc[0]*100
        dd=dd_series(r["portfolio_value"].values)
        ax1.plot(d,ns,color=ACCENT,lw=1.8,label="Strategy",zorder=3)
        ax1.plot(d,nb,color=ORANGE,lw=1.4,label="Benchmark",zorder=2,alpha=0.85)
        in_dd=False; ds=None
        for dt,v in zip(d,dd):
            if v<-0.05 and not in_dd: in_dd=True; ds=dt
            elif v>=-0.01 and in_dd: ax1.axvspan(ds,dt,color=RED,alpha=0.10,zorder=1); in_dd=False
        if in_dd: ax1.axvspan(ds,d[-1],color=RED,alpha=0.10)
        ax1.axhline(100,color=BORDER,lw=0.7,ls="--",alpha=0.6)
        _ax(ax1,yl="Indexed (base=100)")
        ax1.legend(loc="upper left",framealpha=0.15,facecolor=PANEL_BG,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
        ax2.fill_between(d,dd*100,0,color=RED,alpha=0.45)
        ax2.plot(d,dd*100,color=RED,lw=0.6); ax2.axhline(0,color=BORDER,lw=0.6)
        _ax(ax2,yl="Drawdown %")
        rs=r["strategy_returns"].rolling(TRADING_DAYS).apply(
            lambda x: 0.0 if x.std()<1e-6 else
            float(np.clip((x.mean()-RISK_FREE/TRADING_DAYS)/x.std()*np.sqrt(TRADING_DAYS),-4,4)),raw=True)
        ax3.plot(d,rs,color=CYAN,lw=1.2)
        ax3.axhline(1.0,color=GOLD,ls="--",lw=0.8,alpha=0.7)
        ax3.axhline(0.0,color=MUTED,ls="-",lw=0.5,alpha=0.4)
        ax3.fill_between(d,rs,0,where=rs>0,color=CYAN,alpha=0.15)
        ax3.fill_between(d,rs,0,where=rs<0,color=RED,alpha=0.15)
        _ax(ax3,yl="252d Sharpe",xl="Date")
        plt.setp(ax1.get_xticklabels(),visible=False)
        plt.setp(ax2.get_xticklabels(),visible=False)
        plt.tight_layout(rect=[0,0,1,0.97]); plt.show()
        print("[PLOT 1] Equity Curve ✓")
    except Exception as e: print(f"[WARNING] Plot 1: {e}")


def plot2_drawdown(r):
    try:
        fig,ax=plt.subplots(figsize=(15,6),facecolor=BG); ax.set_facecolor(PANEL_BG)
        d=r.index; dd=dd_series(r["portfolio_value"].values)*100
        ddb=dd_series(r["benchmark_value"].values)*100
        dds=pd.Series(dd,index=d)
        ax.fill_between(d,dds,0,color=RED,alpha=0.35,label="Strategy DD")
        ax.plot(d,dds,color=RED,lw=0.8)
        ax.plot(d,ddb,color=ORANGE,lw=1.0,alpha=0.6,ls="--",label="Benchmark DD")
        ax.axhline(0,color=BORDER,lw=0.7)
        troughs=[]; in_d=False; lm=0; li=0
        for i,v in enumerate(dd/100):
            if v<0:
                if not in_d: in_d=True; lm=v; li=i
                elif v<lm: lm=v; li=i
            else:
                if in_d: troughs.append((lm,li)); in_d=False
        if in_d: troughs.append((lm,li))
        for dp,idx in sorted(troughs,key=lambda x:x[0])[:5]:
            ax.annotate(f"{dp*100:.1f}%\n{str(d[idx])[:10]}",
                        xy=(d[idx],dp*100),xytext=(15,15),textcoords="offset points",
                        color=GOLD,fontsize=7.5,fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>",color=GOLD,lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.3",fc=PANEL_BG,ec=BORDER,alpha=0.85))
        _ax(ax,"DRAWDOWN ANALYSIS  |  Rolling Underwater Equity","Date","Drawdown (%)")
        ax.legend(framealpha=0.15,facecolor=PANEL_BG,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
        plt.tight_layout(); plt.show(); print("[PLOT 2] Drawdown ✓")
    except Exception as e: print(f"[WARNING] Plot 2: {e}")


def plot3_dist(r):
    try:
        fig,axes=plt.subplots(1,2,figsize=(15,6),facecolor=BG)
        fig.suptitle("RETURNS DISTRIBUTION  |  Strategy vs Benchmark",
                     color=TEXT,fontsize=13,fontweight="bold")
        for ax,ck,col,lbl in [(axes[0],"strategy_returns",ACCENT,"Strategy"),
                               (axes[1],"benchmark_returns",ORANGE,"Benchmark")]:
            ret=r[ck].dropna().values; ret=ret[np.abs(ret)<0.12]
            _ax(ax,lbl,"Daily Return (%)","Density")
            ax.hist(ret*100,bins=80,density=True,color=col,alpha=0.3,edgecolor="none")
            x=np.linspace(ret.min()*100,ret.max()*100,400)
            mu,sg=ret.mean()*100,ret.std()*100
            ax.plot(x,stats.norm.pdf(x,mu,sg),color=GREEN,lw=2,label="Normal Fit")
            try:
                df_t,loc_t,sc_t=stats.t.fit(ret*100)
                ax.plot(x,stats.t.pdf(x,df_t,loc_t,sc_t),color=PURPLE,lw=2,ls="--",
                        label=f"t-dist (df={df_t:.1f})")
            except: pass
            try: ax.plot(x,gaussian_kde(ret*100)(x),color=GOLD,lw=1.5,ls=":",label="KDE")
            except: pass
            v95=np.percentile(ret,5)*100
            cv95=ret[ret<=np.percentile(ret,5)].mean()*100
            ax.axvline(v95,color=RED,lw=1.5,ls="--",label=f"VaR 95%={v95:.2f}%")
            ax.axvline(cv95,color=PURPLE,lw=1.5,ls=":",label=f"CVaR 95%={cv95:.2f}%")
            sk=stats.skew(ret); ku=stats.kurtosis(ret)
            ax.text(0.97,0.97,f"Skew: {sk:.3f}\nKurt: {ku:.3f}",
                    transform=ax.transAxes,color=GOLD,fontsize=8.5,ha="right",va="top",
                    bbox=dict(boxstyle="round,pad=0.4",fc=PANEL_BG,ec=BORDER,alpha=0.85))
            ax.legend(framealpha=0.15,facecolor=PANEL_BG,edgecolor=BORDER,labelcolor=TEXT,fontsize=7.5)
        plt.tight_layout(); plt.show(); print("[PLOT 3] Returns Distribution ✓")
    except Exception as e: print(f"[WARNING] Plot 3: {e}")


def plot4_trades(trades):
    try:
        if len(trades)==0: print("[WARNING] Plot 4 skipped."); return
        fig,ax1=plt.subplots(figsize=(15,6),facecolor=BG)
        ax1.set_facecolor(PANEL_BG); ax2=ax1.twinx(); ax2.set_facecolor(PANEL_BG)
        t2=trades.copy(); t2["entry_date"]=pd.to_datetime(t2["entry_date"])
        w=t2["pnl_pct"]>0; sz=np.clip(t2["holding_days"]*25,20,300)
        ax1.scatter(t2.loc[w,"entry_date"],t2.loc[w,"pnl_pct"],
                    c=GREEN,s=sz[w],alpha=0.75,zorder=3,label="Win",edgecolors="none")
        ax1.scatter(t2.loc[~w,"entry_date"],t2.loc[~w,"pnl_pct"],
                    c=RED,s=sz[~w],alpha=0.75,zorder=3,label="Loss",edgecolors="none")
        ax1.axhline(0,color=BORDER,lw=0.8)
        _ax(ax1,"TRADE-BY-TRADE P&L  |  Execution Quality","Entry Date","Trade P&L (%)")
        ax1.legend(loc="upper left",framealpha=0.15,facecolor=PANEL_BG,edgecolor=BORDER,labelcolor=TEXT,fontsize=9)
        cum=t2["pnl_pct"].cumsum()
        ax2.plot(t2["entry_date"],cum,color=GOLD,lw=1.8)
        ax2.fill_between(t2["entry_date"],cum,0,color=GOLD,alpha=0.08)
        ax2.set_ylabel("Cumulative P&L (%)",color=GOLD,fontsize=8)
        ax2.tick_params(colors=GOLD,labelsize=8)
        plt.tight_layout(); plt.show(); print("[PLOT 4] Trade P&L ✓")
    except Exception as e: print(f"[WARNING] Plot 4: {e}")


def plot5_rolling(r,trades):
    """Rolling metrics — all Sharpe/Sortino clipped to [-5,5], win rate on trade days only."""
    try:
        fig=plt.figure(figsize=(15,12),facecolor=BG)
        fig.suptitle("ROLLING PERFORMANCE METRICS  |  63-Day Window",
                     color=TEXT,fontsize=13,fontweight="bold",y=0.99)
        axes=[fig.add_subplot(4,1,i+1) for i in range(4)]
        for ax in axes: ax.set_facecolor(PANEL_BG)
        W=63; d=r.index; ret=r["strategy_returns"]

        def roll_sh(x):
            s=x.std()
            return 0.0 if s<1e-6 else float(np.clip((x.mean()-RISK_FREE/TRADING_DAYS)/s*np.sqrt(TRADING_DAYS),-5,5))
        def roll_so(x):
            down=x[x<RISK_FREE/TRADING_DAYS]
            s=down.std()*np.sqrt(TRADING_DAYS) if len(down)>1 else 1e-9
            return 0.0 if s<1e-6 else float(np.clip((x.mean()-RISK_FREE/TRADING_DAYS)*TRADING_DAYS/s,-5,5))
        def trade_wr(x):
            nz=x[x!=0]; return (nz>0).mean()*100 if len(nz)>0 else np.nan

        rsh=ret.rolling(W).apply(roll_sh,raw=False)
        rsort=ret.rolling(W).apply(roll_so,raw=False)
        rvol=ret.rolling(W).std()*np.sqrt(TRADING_DAYS)*100
        rwr=ret.rolling(W).apply(trade_wr,raw=False)

        for ax,series,col,yl,ref in [
            (axes[0],rsh,ACCENT,"Sharpe",1.0),(axes[1],rsort,GREEN,"Sortino",1.0),
            (axes[2],rvol,ORANGE,"Ann. Vol (%)",rvol.mean()),(axes[3],rwr,PURPLE,"Win Rate (%)",50.0)]:
            ax.plot(d,series,color=col,lw=1.2)
            ax.axhline(ref,color=GOLD,ls="--",lw=0.9,alpha=0.75)
            ax.axhline(0,color=MUTED,ls="-",lw=0.4,alpha=0.4)
            if ref==50.0:
                ax.fill_between(d,series,ref,where=series>ref,color=GREEN,alpha=0.12)
                ax.fill_between(d,series,ref,where=series<ref,color=RED,alpha=0.12)
            else:
                ax.fill_between(d,series,0,where=series>0,color=col,alpha=0.12)
                ax.fill_between(d,series,0,where=series<0,color=RED,alpha=0.12)
            _ax(ax,yl=yl)
        _ax(axes[3],xl="Date")
        for ax in axes[:-1]: plt.setp(ax.get_xticklabels(),visible=False)
        plt.tight_layout(rect=[0,0,1,0.97],h_pad=0.3); plt.show()
        print("[PLOT 5] Rolling Metrics ✓")
    except Exception as e: print(f"[WARNING] Plot 5: {e}")


def plot6_heatmaps(r):
    try:
        fig=plt.figure(figsize=(16,8),facecolor=BG)
        fig.suptitle("SEASONALITY  |  Monthly Returns & Day-of-Week Effect",
                     color=TEXT,fontsize=13,fontweight="bold",y=0.99)
        gs=gridspec.GridSpec(1,2,figure=fig,width_ratios=[3,1.2],wspace=0.3)
        ax1=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1])
        ax1.set_facecolor(PANEL_BG); ax2.set_facecolor(PANEL_BG)
        ret=r["strategy_returns"].copy(); ret.index=pd.to_datetime(ret.index)
        mo=ret.resample("ME").apply(lambda x:(1+x).prod()-1)*100
        piv=pd.DataFrame({"y":mo.index.year,"m":mo.index.month,"v":mo.values})\
              .pivot(index="y",columns="m",values="v")
        mn=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        piv.columns=[mn[c-1] for c in piv.columns]
        cmap=sns.diverging_palette(10,133,as_cmap=True)
        sns.heatmap(piv,ax=ax1,cmap=cmap,center=0,annot=True,fmt=".1f",
                    linewidths=0.4,linecolor=BORDER,annot_kws={"size":7,"color":TEXT},
                    cbar_kws={"shrink":0.8},robust=True)
        ax1.set_title("Monthly Returns (%)",color=GOLD,fontsize=10,fontweight="bold",pad=8)
        ax1.tick_params(colors=TEXT,labelsize=8)
        ax1.set_xlabel("Month",color=MUTED,fontsize=8); ax1.set_ylabel("Year",color=MUTED,fontsize=8)
        ax1.collections[0].colorbar.ax.tick_params(colors=TEXT)
        dorder=["Monday","Tuesday","Wednesday","Thursday","Friday"]
        rd=ret.to_frame("r"); rd["dow"]=rd.index.day_name()
        avg=rd.groupby("dow")["r"].mean()*100; avg=avg.reindex(dorder).fillna(0)
        bars=ax2.barh(avg.index,avg.values,color=[GREEN if v>=0 else RED for v in avg.values],alpha=0.85,height=0.6)
        ax2.axvline(0,color=MUTED,lw=0.8)
        for bar,val in zip(bars,avg.values):
            ax2.text(val+(0.0003 if val>=0 else -0.0003),bar.get_y()+bar.get_height()/2,
                     f"{val:.3f}%",va="center",ha="left" if val>=0 else "right",color=TEXT,fontsize=8)
        ax2.set_title("Avg Return by\nDay of Week",color=GOLD,fontsize=10,fontweight="bold",pad=8)
        _ax(ax2,xl="Avg Return (%)"); ax2.invert_yaxis(); ax2.tick_params(colors=TEXT,labelsize=8)
        plt.tight_layout(rect=[0,0,1,0.96]); plt.show(); print("[PLOT 6] Heatmaps ✓")
    except Exception as e: print(f"[WARNING] Plot 6: {e}")


def plot7_risk_return(r):
    try:
        fig,ax=plt.subplots(figsize=(12,8),facecolor=BG); ax.set_facecolor(PANEL_BG)
        W=TRADING_DAYS
        for col,lbl,marker,cm in [("strategy_returns","Strategy","o","plasma"),
                                    ("benchmark_returns","Benchmark","s","viridis")]:
            rv=r[col].rolling(W).std()*np.sqrt(W)*100
            rr=r[col].rolling(W).mean()*W*100
            mask=rv.notna()&rr.notna()
            vols=rv[mask].values; rets=rr[mask].values; n=len(vols)
            if n==0: continue
            ax.scatter(vols,rets,c=np.arange(n),cmap=cm,s=18,marker=marker,alpha=0.5,zorder=3)
            ax.scatter(vols.mean(),rets.mean(),
                       color=GOLD if "strategy" in col else ORANGE,
                       s=220,marker="D" if "strategy" in col else "*",zorder=5,
                       label=f"{lbl} (mean)",edgecolors="white",linewidths=0.8)
        all_v=r["strategy_returns"].rolling(W).std().dropna()*np.sqrt(W)*100
        vr=np.linspace(max(all_v.min()-2,1),all_v.max()+5,100)
        for sh,ls,al in [(0.5,":",0.55),(1.0,"--",0.65),(1.5,"-.",0.55)]:
            ax.plot(vr,RISK_FREE*100+sh*vr,color=CYAN,lw=0.9,ls=ls,alpha=al,label=f"Sharpe={sh}")
        ax.axhline(RISK_FREE*100,color=MUTED,ls="--",lw=0.7,alpha=0.5,
                   label=f"Risk-free ({RISK_FREE*100:.1f}%)")
        _ax(ax,"RISK-RETURN LANDSCAPE  |  Rolling 252-Day Windows",
            "Annualized Volatility (%)","Annualized Return (%)")
        ax.legend(framealpha=0.15,facecolor=PANEL_BG,edgecolor=BORDER,labelcolor=TEXT,fontsize=8)
        plt.tight_layout(); plt.show(); print("[PLOT 7] Risk-Return ✓")
    except Exception as e: print(f"[WARNING] Plot 7: {e}")


def plot8_monte_carlo(r):
    try:
        N=1000; ret=r["strategy_returns"].dropna().values; n=len(ret)
        sims=np.zeros((N,n))
        for i in range(N): sims[i]=INITIAL_CAPITAL*np.cumprod(1+np.random.choice(ret,n,replace=True))
        fig=plt.figure(figsize=(15,8),facecolor=BG)
        fig.suptitle("MONTE CARLO SIMULATION  |  1000 Bootstrap Paths",
                     color=TEXT,fontsize=13,fontweight="bold",y=0.99)
        gs=gridspec.GridSpec(1,2,figure=fig,width_ratios=[3,1.5],wspace=0.25)
        ax1=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1])
        ax1.set_facecolor(PANEL_BG); ax2.set_facecolor(PANEL_BG)
        days=np.arange(n)
        for i in range(N): ax1.plot(days,sims[i]/1e5,color=ACCENT,alpha=0.025,lw=0.3)
        pcts=[5,25,50,75,95]; pcols=[RED,ORANGE,"white",GREEN,CYAN]
        for pct,col in zip(pcts,pcols):
            path=np.percentile(sims,pct,axis=0)
            ax1.plot(days,path/1e5,color=col,lw=2,label=f"P{pct}: {path[-1]/1e5:.1f}L")
        actual=r["portfolio_value"].values
        ax1.plot(days,actual/1e5,color=GOLD,lw=2.5,ls="--",zorder=5,
                 label=f"Actual: {actual[-1]/1e5:.1f}L")
        _ax(ax1,yl="Portfolio (Rs Lakhs)",xl="Trading Days")
        ax1.legend(framealpha=0.15,facecolor=PANEL_BG,edgecolor=BORDER,labelcolor=TEXT,fontsize=8)
        finals=sims[:,-1]/1e5
        ax2.hist(finals,bins=50,color=ACCENT,alpha=0.6,orientation="horizontal",edgecolor="none")
        for pct,col in zip(pcts,pcols):
            val=np.percentile(finals,pct)
            ax2.axhline(val,color=col,lw=1.5,ls="--",label=f"P{pct}: {val:.1f}L")
        ax2.axhline(actual[-1]/1e5,color=GOLD,lw=2.0,label=f"Actual: {actual[-1]/1e5:.1f}L")
        _ax(ax2,yl="Final Value (Rs Lakhs)",xl="Count")
        ax2.legend(framealpha=0.15,facecolor=PANEL_BG,edgecolor=BORDER,labelcolor=TEXT,fontsize=7.5)
        print(f"\n[MC] P5 : Rs{np.percentile(sims[:,-1],5):>12,.0f}")
        print(f"[MC] P50: Rs{np.percentile(sims[:,-1],50):>12,.0f}")
        print(f"[MC] P95: Rs{np.percentile(sims[:,-1],95):>12,.0f}")
        plt.tight_layout(rect=[0,0,1,0.97]); plt.show(); print("[PLOT 8] Monte Carlo ✓")
    except Exception as e: print(f"[WARNING] Plot 8: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n"+"="*66)
    print("  NIFTY 50 QUANT SYSTEM v3.0  |  UNiverse Capital")
    print("="*66+"\n")

    df      = download_data()
    feat    = compute_features(df)
    print(f"[PIPELINE] Features: {feat.shape}")

    sig     = generate_signals(feat)
    sc      = sig["signal"].value_counts()
    print(f"[PIPELINE] Signals -> Long:{sc.get(1.0,0)}  Short:{sc.get(-1.0,0)}  Flat:{sc.get(0.0,0)}\n")

    result,trades = run_backtest(sig)
    nl = (trades["signal"]==1).sum()  if len(trades)>0 else 0
    ns = (trades["signal"]==-1).sum() if len(trades)>0 else 0
    
    print(f"[PIPELINE] Trades: {len(trades)} ({nl} long, {ns} short)")
    if len(trades)>0: print(f"[PIPELINE] Avg hold: {trades['holding_days'].mean():.1f} days")
    
    metrics = compute_metrics(result,trades)

    # --- PRINT CONSOLE METRICS TEXT SUMMARY ---
    init_cap = INITIAL_CAPITAL
    fin_cap = result['portfolio_value'].iloc[-1]
    ben_cap = result['benchmark_value'].iloc[-1]
    
    print("\n" + "="*66)
    print("  PERFORMANCE METRICS SUMMARY")
    print("="*66)
    print(f"Initial Capital      : Rs {init_cap:>12,.0f}")
    print(f"Final Capital (Strat): Rs {fin_cap:>12,.0f}")
    print(f"Final Capital (Bench): Rs {ben_cap:>12,.0f}")
    print("-" * 66)
    print(f"{'METRIC':<22} | {'STRATEGY':>15} | {'BENCHMARK':>15}")
    print("-" * 66)
    
    s = metrics["strategy"]
    b = metrics["benchmark"]
    t = metrics["trades"]
    
    def pr(name, s_val, b_val, is_pct=False, is_int=False):
        if is_int:
            sv = f"{int(s_val)}"
            bv = f"{int(b_val)}" if b_val is not None else "-"
        else:
            sv = f"{s_val:+.2f}%" if is_pct else f"{s_val:.4f}"
            if b_val is not None:
                bv = f"{b_val:+.2f}%" if is_pct else f"{b_val:.4f}"
            else:
                bv = "-"
        print(f"{name:<22} | {sv:>15} | {bv:>15}")

    pr("Total Return", s["total_return"], b["total_return"], is_pct=True)
    pr("CAGR", s["cagr"], b["cagr"], is_pct=True)
    pr("Ann. Volatility", s["ann_vol"], b["ann_vol"], is_pct=True)
    pr("Max Drawdown", s["max_dd"], b["max_dd"], is_pct=True)
    pr("Sharpe Ratio", s["sharpe"], b["sharpe"])
    pr("Sortino Ratio", s["sortino"], b["sortino"])
    pr("Calmar Ratio", s["calmar"], b["calmar"])
    pr("Omega Ratio", s["omega"], b["omega"])
    pr("Info Ratio", s["info_ratio"], None)
    print("-" * 66)
    pr("Total Trades", t["n_trades"], None, is_int=True)
    pr("Win Rate", t["win_rate"], None, is_pct=True)
    pr("Profit Factor", t["profit_factor"], None)
    pr("Expectancy/Trade", t["expectancy"], None, is_pct=True)
    pr("Avg Hold (Days)", t["avg_hold"], None)
    print("="*66 + "\n")
    # -------------------------------------------

    print("[PIPELINE] Rendering all plots ...")
    plot_metrics_dashboard(metrics)
    plot1_equity_curve(result)
    plot2_drawdown(result)
    plot3_dist(result)
    plot4_trades(trades)
    plot5_rolling(result,trades)
    plot6_heatmaps(result)
    plot7_risk_return(result)
    plot8_monte_carlo(result)

    print("\n"+"="*66)
    print("  PIPELINE COMPLETE")
    print("="*66+"\n")


if __name__=="__main__":
    main()

improvement  = """

    tarine 8 yers of data then back test on 2 yeard unseen data 

"""