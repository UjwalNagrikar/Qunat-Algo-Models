
====================================================================================
  SHEET 1  |  IN-SAMPLE  |  NSE FUTSTK ENGINE v5.0  |  Universe Capital
====================================================================================

  A. SETUP & DATA
------------------------------------------------------------------------------------
  Data Source     : Yahoo Finance (live)
  Universe        : 12 stocks  |  RELIANCE  TCS  HDFCBANK  INFY  ICICIBANK  HINDUNILVR  ITC  SBIN  AXISBANK  BHARTIARTL  LT  KOTAKBANK
  Initial Capital : INR 5,000,000
  Margin Rate     : 20%  |  Risk/Trade: 1.5%  |  Max Positions: 6
  Signal          : Composite z-score = avg(z_1day + z_3day)
  Vol Filter      : 10-day rolling std cap
  Trend Filter    : OFF (default for live NSE — avoids signal depletion)
  Stop-Loss       : Disabled — time exit only
  Futures PnL     : (Exit − Entry) × Lot × Num Lots × Direction
  Cost Model      : Brokerage + STT(0.01% sell) + Exchange + SEBI + GST + Slippage

  IN-SAMPLE WINDOWS:
    Optimisation  : 2010-01-01  to  2017-12-31  [8 years]
    Validation    : 2018-01-01  to  2021-12-31  [4 years]
  OOS WINDOW      : 2023-01-01  to  2024-12-31  [2 years  NEVER touched in IS]

  Grid size    : 96 valid combinations (min 80 trades filter applied)
  Param keys   : ['holding_period', 'z_threshold', 'vol_threshold', 'cooldown_days']

  B. OPTIMISATION GRID  (2010-2017)  |  Top 15 by Sharpe
------------------------------------------------------------------------------------
  Rk  Hold  Z-Thr  Vol%  CD   Sharpe   Calmar   CAGR%   MaxDD%    WR%      PF  Trades  Cost%  FinalEq(L)
  ................................................................................
   1    10   0.50   2.5   3   1.0915   0.7191    4.99    -6.93   52.5  1.2964    1150   11.9      73.73L
   2    10   0.50   2.5   5   0.9084   0.6582    4.23    -6.43   52.8  1.2613    1137   13.1      69.63L
   3    10   0.50   3.5   3   0.8752   0.4549    4.07    -8.95   51.7  1.2273    1170   14.0      68.76L
   4    10   0.75   3.5   3   0.8383   0.3028    3.96   -13.06   50.8  1.2309    1148   14.1      68.15L
   5    10   1.50   4.5   3   0.7751   0.4266    3.15    -7.38   50.8  1.2864     794   12.5      64.04L
   6    10   1.50   3.5   3   0.7557   0.3960    3.02    -7.63   50.6  1.2758     782   12.8      63.42L
   7     5   0.50   2.5   3   0.7497   0.3735    3.62    -9.69   52.4  1.1631    2100   24.2      66.40L
   8     7   0.50   2.5   5   0.7145   0.3099    3.36   -10.83   52.9  1.1798    1522   20.1      65.08L
   9    10   0.50   4.5   3   0.6860   0.3858    3.36    -8.71   49.9  1.1840    1169   16.6      65.10L
  10    10   1.00   2.5   3   0.6736   0.3653    3.15    -8.62   51.8  1.2027    1054   16.1      64.04L
  11     5   1.00   4.5   3   0.6527   0.3393    3.05    -8.99   52.2  1.1497    1823   25.2      63.55L
  12     5   0.75   2.5   3   0.6483   0.3654    2.95    -8.08   53.1  1.1476    1952   26.5      63.08L
  13    10   1.50   3.5   5   0.6312   0.2939    2.43    -8.27   50.7  1.2217     728   14.7      60.57L
  14    10   1.50   4.5   5   0.6290   0.2955    2.46    -8.33   50.5  1.2237     736   14.6      60.71L
  15     5   0.50   2.5   5   0.6258   0.2899    2.92   -10.09   51.8  1.1441    1883   26.5      62.94L

  Criterion : Sharpe (primary), Calmar (secondary)
  Min trades: 80 — combos below this rejected (prevents noise-fit on tiny samples)
  Top 10 forwarded to Validation scoring

  C. VALIDATION SCORING  (2018-2021)  |  Top-N re-tested, winner by combined score
------------------------------------------------------------------------------------
  Combined Score = sqrt(IS_Sharpe × Val_Sharpe)  [geometric mean, penalises IS overfitting]

  Rk  Hold  Z-Thr  Vol%  CD    IS_Sh     V_Sh     Comb   VCAGR%    VDD%   VWR%     VPF  VTrades  VFinalEq(L)
  ................................................................................
   1    10   0.50   4.5   3   0.6860   0.9520   0.8081     6.27   -6.75   52.8  1.2772      581       63.77L  <- WINNER
   2    10   0.50   2.5   3   1.0915   0.3420   0.6110     2.03   -9.80   50.0  1.0840      558       54.19L
   3    10   1.00   2.5   3   0.6736   0.5118   0.5872     2.69   -8.92   48.5  1.1283      505       55.60L
   4    10   0.50   2.5   5   0.9084   0.2717   0.4968     1.54   -8.04   46.8  1.0656      549       53.15L
   5     7   0.50   2.5   5   0.7145   0.2882   0.4538     1.73   -7.18   50.7  1.0644      740       53.54L
   6     5   0.50   2.5   3   0.7497   0.2599   0.4414     1.50   -8.90   49.7  1.0482     1018       53.07L
   7    10   0.50   3.5   3   0.8752  -0.8458   0.0296    -6.54  -27.67   48.0  0.7862      575       38.16L
   8    10   0.75   3.5   3   0.8383  -0.4873   0.0290    -3.98  -18.67   47.9  0.8639      564       42.51L
   9    10   1.50   4.5   3   0.7751  -0.8366   0.0278    -5.79  -24.95   45.9  0.7440      381       39.39L
  10    10   1.50   3.5   3   0.7557  -0.8655   0.0275    -5.74  -23.34   45.3  0.7342      369       39.47L

  D. LOCKED OPTIMAL PARAMETERS  (frozen — never changed after this point)
------------------------------------------------------------------------------------
  Holding Period    = 10 days
  Z-Score Threshold = 0.50σ  (|composite z| > threshold to enter)
  Vol Filter        = 4.5%  (skip stocks with 10d vol above this)
  Cooldown          = 3 days  (no re-entry on same stock)
  Trend Filter      = OFF
  Stop-Loss         = Disabled  (time-exit only)
  Max Positions     = 6

  E. IN-SAMPLE PERFORMANCE  (locked params applied per sub-window)
------------------------------------------------------------------------------------
  Metric                                       Opt 2010-17           Val 2018-21
  ------------------------------------------------------------------------------
  Initial Capital                            INR 5,000,000         INR 5,000,000
  Final Equity                               INR 6,509,593         INR 6,376,501
  Trades Executed                                    1,169                   581
  ------------------------------------------------------------------------------
  Total Return (%)                                 +30.19%               +27.53%
  CAGR (%)                                          +3.36%                +6.27%
  Sharpe Ratio                                      0.6860                0.9520
  Sortino Ratio                                     1.1173                1.7192
  Calmar Ratio                                      0.3858                0.9293
  Max Drawdown (%)                                  -8.71%                -6.75%
  ------------------------------------------------------------------------------
  Win Rate (%)                                     +49.87%               +52.84%
  Profit Factor                                     1.1840                1.2772
  Expectancy / Trade                            INR +1,292            INR +2,366
  Avg Winner                                   INR +16,674           INR +20,627
  Avg Loser                                    INR -14,011           INR -18,095
    Long  Trades                                       560                   261
    Short Trades                                       609                   320
  Avg Hold (days)                                     9.97                  9.95
  Max Consec. Wins                                      12                    11
  Max Consec. Losses                                     9                     7
  ------------------------------------------------------------------------------
  Gross PnL                                 INR +1,812,237        INR +1,548,583
  Total Costs                                 INR +301,736          INR +174,153
  Net PnL                                   INR +1,509,593        INR +1,376,501
  Cost / Gross (%)                                   16.6%                 11.2%
  Long Side Net PnL                         INR +2,298,403        INR +1,180,180
  Short Side Net PnL                          INR -787,901          INR +194,249
  Avg Lots / Trade                                    3.83                  1.40
  Avg Ret on Margin(%)                                2.0%                  2.3%

====================================================================================

====================================================================================
  SHEET 2  |  OUT-OF-SAMPLE  |  NSE FUTSTK ENGINE v5.0  |  Universe Capital
  PURE FORWARD TEST  |  Parameters frozen from IS — zero contact with OOS data
====================================================================================

  A. OOS SETUP
------------------------------------------------------------------------------------
  OOS Window      : 2023-01-01  to  2024-12-31  [2 years]
  Data Status     : UNSEEN — no contact with IS optimisation or validation
  Parameters      : LOCKED from IS validation scoring, not modified
  Data Source     : Yahoo Finance (live)

  LOCKED PARAMS:
    Holding Period    = 10 days
    Z-Score Threshold = 0.50σ
    Vol Filter        = 4.5%
    Cooldown          = 3 days
    Trend Filter      = OFF
    Stop-Loss         = Disabled

  B. OUT-OF-SAMPLE PERFORMANCE  (2023-2024)
------------------------------------------------------------------------------------
  Metric                                                                 Value
  ----------------------------------------------------------------------------
  Initial Capital                                                INR 5,000,000
  Final Equity                                                   INR 5,902,100
  Trades Executed                                                          294
  ----------------------------------------------------------------------------
  Total Return (%)                                                     +18.04%
  CAGR (%)                                                              +8.68%
  Sharpe Ratio                                                          1.0071
  Sortino Ratio                                                         1.5323
  Calmar Ratio                                                          0.7692
  Max Drawdown (%)                                                     -11.28%
  ----------------------------------------------------------------------------
  Win Rate (%)                                                         +54.76%
  Profit Factor                                                         1.3190
  Expectancy / Trade                                                INR +3,045
  Avg Winner                                                       INR +22,992
  Avg Loser                                                        INR -21,101
    Long  Trades                                                           140
    Short Trades                                                           154
    Stop-Loss Exits                                                          0
  Avg Hold (days)                                                         9.85
  Max Consec. Wins                                                          10
  Max Consec. Losses                                                         8
  ----------------------------------------------------------------------------
  Gross PnL  (futures formula)                                  INR +1,037,920
  Total Transaction Costs                                         INR +142,729
  Net PnL                                                         INR +902,100
  Cost / Gross (%)                                                       13.8%
  Long Side Net PnL                                             INR +1,194,435
  Short Side Net PnL                                              INR -299,245
  Avg Lots / Trade                                                        1.09
  Avg Return on Margin (%)                                                2.4%

  C. IS vs OOS COMPARISON
------------------------------------------------------------------------------------
  Metric                    IS-Opt 10-17  IS-Val 18-21     OOS 23-24
  ------------------------------------------------------------------
  Initial Capital                 50.00L        50.00L        50.00L
  Final Equity                    65.10L        63.77L        59.02L
  Trades Executed                  1,169           581           294
  ------------------------------------------------------------------
  Total Return (%)               +30.19%       +27.53%       +18.04%
  CAGR (%)                        +3.36%        +6.27%        +8.68%
  Sharpe                          0.6860        0.9520        1.0071
  Sortino                         1.1173        1.7192        1.5323
  Max DD (%)                      -8.71%        -6.75%       -11.28%
  Win Rate (%)                   +49.87%       +52.84%       +54.76%
  Profit Factor                   1.1840        1.2772        1.3190

  D. CONCLUSION
------------------------------------------------------------------------------------
  Sharpe path : IS-Opt 0.6860  →  IS-Val 0.9520  →  OOS 1.0071
  IS-Opt → Val: +38.8%  (improvement — Val generalised better than Opt)
  IS-Val → OOS: +5.8%  (stable)

  Verdict    : STRONG    — Robust on completely unseen 2-year forward window.
  Fit check  : Low — IS-Val Sharpe did not inflate vs IS-Opt
  Cost check : Cost/Gross = 13.8%  →  healthy (<50%)
  Sample     : OOS trades = 294  →  adequate sample

  Lot Sizes  : RELIANCE=250  TCS=150  HDFCBANK=550  INFY=300  ICICIBANK=1375  HINDUNILVR=300  ITC=3200  SBIN=1500  AXISBANK=1200  BHARTIARTL=950  LT=150  KOTAKBANK=400
  Runtime    : 238.3s  |  No lookahead  |  Params locked before OOS
====================================================================================

