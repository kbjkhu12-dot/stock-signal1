import os, json, math, time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

try:
    from pykrx import stock as krx
except Exception:
    krx = None

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
REPORTS = BASE / "reports"
DATA.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

STATE = DATA / "state.json"
FUND = DATA / "fundamentals.csv"

TOP_N = int(os.getenv("TOP_N", "250"))
MIN_AVG_AMOUNT = float(os.getenv("MIN_AVG_AMOUNT", "1000000000"))
PRICE_DAYS = int(os.getenv("PRICE_DAYS", "150"))
REL_DAYS = int(os.getenv("REL_DAYS", "60"))
FUND_BATCH = int(os.getenv("FUND_BATCH", "30"))
MIN_SIGNALS = int(os.getenv("MIN_SIGNALS", "2"))

def pct(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0:
            return np.nan
        return (float(a) / float(b) - 1) * 100
    except Exception:
        return np.nan

def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"fund_cursor": 0, "signals": {}, "scores": {}}

def save_state(x):
    STATE.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")

def listing():
    # 시총순 리스트 우선, 실패하면 KRX 전체
    for sym in ["KRX-MARCAP", "KRX"]:
        try:
            df = fdr.StockListing(sym).copy()
            if len(df):
                break
        except Exception:
            df = pd.DataFrame()
    if df.empty:
        raise RuntimeError("종목 리스트를 가져오지 못했습니다.")
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("code","symbol"): ren[c] = "Code"
        if lc == "name": ren[c] = "Name"
        if lc == "market": ren[c] = "Market"
        if lc in ("amount","marcap"): ren[c] = c
    df = df.rename(columns=ren)
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    if "Market" in df:
        df = df[df["Market"].isin(["KOSPI","KOSDAQ"])]
    return df.head(max(TOP_N*2, TOP_N))

def price_df(code, start):
    try:
        return fdr.DataReader(code, start)
    except Exception:
        return pd.DataFrame()

def index_df(market, start):
    sym = "KQ11" if str(market).upper().startswith("KOSDAQ") else "KS11"
    try:
        return fdr.DataReader(sym, start)
    except Exception:
        return pd.DataFrame()

def liquidity_universe(df):
    start = (datetime.now() - timedelta(days=PRICE_DAYS*2)).strftime("%Y-%m-%d")
    idx_cache = {}
    rows = []
    for n, (_, r) in enumerate(df.iterrows(), 1):
        code = r["Code"]
        market = r.get("Market", "KOSPI")
        px = price_df(code, start)
        if len(px) < REL_DAYS + 5 or "Close" not in px or "Volume" not in px:
            continue
        px = px.tail(PRICE_DAYS)
        if "Amount" in px:
            avg_amt = pd.to_numeric(px["Amount"], errors="coerce").tail(20).mean()
        else:
            avg_amt = (pd.to_numeric(px["Close"], errors="coerce") *
                       pd.to_numeric(px["Volume"], errors="coerce")).tail(20).mean()
        if pd.isna(avg_amt) or avg_amt < MIN_AVG_AMOUNT:
            continue
        idx_key = "KOSDAQ" if str(market).upper().startswith("KOSDAQ") else "KOSPI"
        if idx_key not in idx_cache:
            idx_cache[idx_key] = index_df(idx_key, start)
        ix = idx_cache[idx_key]
        if len(ix) < REL_DAYS + 5:
            continue
        sret = pct(px["Close"].iloc[-1], px["Close"].iloc[-REL_DAYS])
        iret = pct(ix["Close"].iloc[-1], ix["Close"].iloc[-REL_DAYS])
        high120 = px["Close"].tail(120).max()
        dd = pct(px["Close"].iloc[-1], high120)
        rows.append({
            "Code": code, "Name": r.get("Name", code), "Market": market,
            "Close": float(px["Close"].iloc[-1]),
            "AvgAmount20": float(avg_amt), "Ret60": sret,
            "IndexRet60": iret, "Rel60": sret-iret, "DD120": dd
        })
        time.sleep(0.03)
    u = pd.DataFrame(rows)
    if u.empty:
        return u
    return u.sort_values("AvgAmount20", ascending=False).head(TOP_N)

def naver_finstate(code):
    try:
        fs = fdr.SnapDataReader(f"NAVER/FINSTATE-Q/{code}")
        if fs is None or len(fs) == 0:
            return {}
        fs = fs.copy()
        idx = [str(x) for x in fs.index]
        def pick(keys):
            for k in keys:
                for i, nm in enumerate(idx):
                    if k in nm:
                        return fs.iloc[i]
            return None
        def two(s):
            if s is None:
                return np.nan, np.nan
            z = pd.to_numeric(s.astype(str).str.replace(",",""), errors="coerce").dropna()
            if len(z) < 2: return np.nan, np.nan
            return float(z.iloc[-1]), float(z.iloc[-2])
        op0, op1 = two(pick(["영업이익"]))
        oc0, oc1 = two(pick(["영업활동현금흐름","영업활동으로인한현금흐름"]))
        ca0, ca1 = two(pick(["유형자산의취득","유형자산 취득"]))
        f0 = oc0 - abs(ca0) if not pd.isna(oc0) and not pd.isna(ca0) else np.nan
        f1 = oc1 - abs(ca1) if not pd.isna(oc1) and not pd.isna(ca1) else np.nan
        return {"Code":code,"op_now":op0,"op_prev":op1,"ocf_now":oc0,"ocf_prev":oc1,
                "fcf_now":f0,"fcf_prev":f1,"fund_updated":datetime.now().isoformat(timespec="seconds")}
    except Exception:
        return {}

def refresh_fund(u, st):
    cols=["Code","op_now","op_prev","ocf_now","ocf_prev","fcf_now","fcf_prev","fund_updated"]
    old = pd.read_csv(FUND, dtype={"Code":str}) if FUND.exists() else pd.DataFrame(columns=cols)
    codes = u["Code"].tolist()
    if not codes: return old, st
    cur=int(st.get("fund_cursor",0))
    batch=[]
    for i in range(min(FUND_BATCH, len(codes))):
        row=naver_finstate(codes[(cur+i)%len(codes)])
        if row: batch.append(row)
        time.sleep(0.08)
    if batch:
        old=pd.concat([old,pd.DataFrame(batch)],ignore_index=True)
        old["Code"]=old["Code"].astype(str).str.zfill(6)
        old=old.sort_values("fund_updated").drop_duplicates("Code",keep="last")
        old.to_csv(FUND,index=False)
    st["fund_cursor"]=(cur+FUND_BATCH)%len(codes)
    return old, st

def krx_snapshot(codes):
    # pykrx는 보조 소스. 실패해도 전체 실행은 계속됨.
    if krx is None:
        return pd.DataFrame()
    d = datetime.now()
    # 휴일/당일 데이터 공백을 고려해 7일 역탐색
    for off in range(0, 8):
        day=(d-timedelta(days=off)).strftime("%Y%m%d")
        try:
            x=krx.get_market_fundamental_by_ticker(day, market="ALL")
            if x is not None and len(x):
                x=x.reset_index().rename(columns={"티커":"Code"})
                x["Code"]=x["Code"].astype(str).str.zfill(6)
                return x[x["Code"].isin(codes)]
        except Exception:
            pass
    return pd.DataFrame()

def signals(r):
    s=[]
    op0,op1=r.get("op_now"),r.get("op_prev")
    if pd.notna(op0) and pd.notna(op1):
        if op1 <= 0 < op0: s.append("영업이익 흑자전환")
        elif pct(op0,op1) >= 15: s.append("영업이익 +15% 이상")
    oc0,oc1=r.get("ocf_now"),r.get("ocf_prev")
    if pd.notna(oc0) and pd.notna(oc1) and oc0 > oc1: s.append("영업현금흐름 개선")
    f0,f1=r.get("fcf_now"),r.get("fcf_prev")
    if pd.notna(f0) and pd.notna(f1) and f1 <= 0 < f0: s.append("FCF 흑자전환")
    if pd.notna(r.get("Rel60")) and r["Rel60"] <= -10: s.append("60일 시장 대비 -10%p 소외")
    if pd.notna(r.get("DD120")) and r["DD120"] <= -15: s.append("120일 고점 대비 -15%")
    per=r.get("PER"); pbr=r.get("PBR")
    if pd.notna(per) and 0 < per <= 10: s.append("PER 10배 이하")
    if pd.notna(pbr) and 0 < pbr <= 1: s.append("PBR 1배 이하")
    return s

def score_from_signals(ss):
    w={
        "영업이익 흑자전환":25, "영업이익 +15% 이상":20,
        "영업현금흐름 개선":18, "FCF 흑자전환":22,
        "60일 시장 대비 -10%p 소외":12, "120일 고점 대비 -15%":6,
        "PER 10배 이하":8, "PBR 1배 이하":6
    }
    return sum(w.get(x,0) for x in ss)

def write_report(df, prev_scores):
    today=datetime.now().strftime("%Y-%m-%d")
    lines=[f"# Daily Stock Signal — {today}","",
           "> 매수 추천이 아니라 정량 조건 변화 감지용 리서치 목록입니다.",""]
    if df.empty:
        lines += ["오늘 조건 충족 종목이 없습니다."]
    else:
        new=df[df["Status"]=="신규"]
        up=df[df["Status"]=="강화"]
        keep=df[df["Status"]=="유지"]
        lines += [f"**신규 {len(new)} / 강화 {len(up)} / 유지 {len(keep)}**",""]
        for status, part in [("신규",new),("강화",up),("유지",keep)]:
            if part.empty: continue
            lines += [f"## {status}",""]
            for _,r in part.head(15).iterrows():
                lines += [
                    f"### {r['Name']} ({r['Code']}) — {int(r['Score'])}점",
                    f"- 전일 점수: {int(r['PrevScore']) if pd.notna(r['PrevScore']) else 0} / 변화: {int(r['Delta']):+d}",
                    f"- 60일 시장 대비: {r['Rel60']:.1f}%p / 120일 고점 대비: {r['DD120']:.1f}%",
                    f"- 신호: {r['Signals']}",
                    ""
                ]
    text="\n".join(lines)
    (REPORTS/"latest.md").write_text(text,encoding="utf-8")
    (REPORTS/f"{datetime.now():%Y%m%d}.md").write_text(text,encoding="utf-8")
    print(text)

def main():
    st=load_state()
    base=listing()
    u=liquidity_universe(base)
    if u.empty:
        raise RuntimeError("가격/유동성 데이터를 확보하지 못했습니다.")
    fund,st=refresh_fund(u,st)
    m=u.merge(fund,on="Code",how="left")
    k=krx_snapshot(m["Code"].tolist())
    if not k.empty:
        keep=[c for c in ["Code","PER","PBR","EPS","BPS","DIV","DPS"] if c in k.columns]
        m=m.merge(k[keep],on="Code",how="left")
    else:
        m["PER"]=np.nan; m["PBR"]=np.nan

    prev_sig=st.get("signals",{})
    prev_scores=st.get("scores",{})
    out=[]; now_sig={}; now_scores={}
    for _,r in m.iterrows():
        ss=signals(r)
        if len(ss)<MIN_SIGNALS: continue
        sc=score_from_signals(ss)
        code=r["Code"]
        old=prev_sig.get(code,[])
        ps=int(prev_scores.get(code,0))
        delta=sc-ps
        if not old: status="신규"
        elif delta>=8 or len(set(ss)-set(old))>=1: status="강화"
        else: status="유지"
        now_sig[code]=ss; now_scores[code]=sc
        out.append({
            "Code":code,"Name":r["Name"],"Market":r["Market"],
            "Score":sc,"PrevScore":ps,"Delta":delta,"Status":status,
            "Signals":" / ".join(ss),"Rel60":r["Rel60"],"DD120":r["DD120"],
            "AvgAmount20":r["AvgAmount20"],"PER":r.get("PER",np.nan),"PBR":r.get("PBR",np.nan)
        })

    result=pd.DataFrame(out)
    if not result.empty:
        order={"신규":0,"강화":1,"유지":2}
        result["ord"]=result["Status"].map(order)
        result=result.sort_values(["ord","Delta","Score","AvgAmount20"],ascending=[True,False,False,False]).drop(columns="ord")
        result.to_csv(REPORTS/"latest.csv",index=False,encoding="utf-8-sig")
    write_report(result,prev_scores)
    st["signals"]=now_sig; st["scores"]=now_scores; st["last_run"]=datetime.now().isoformat(timespec="seconds")
    save_state(st)

if __name__=="__main__":
    main()
