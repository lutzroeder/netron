import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, cols: set):
    if not cols.issubset(df.columns):
        missing = cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")


# 1) ADX (Wilder)
def ADX_Wilder(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate ADX (Wilder) and append columns: 'DX', '+DI', '-DI', 'ADX'.
    Requires: 'high', 'low', 'close'
    """
    _require_columns(df, {'high', 'low', 'close'})
    if period <= 0:
        raise ValueError("period must be > 0")

    res = df.copy()
    high = res['high']
    low = res['low']
    close = res['close']

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder smoothing (EMA-like but Wilder's)
    tr_smooth = tr.rolling(window=period, min_periods=period).sum()
    pos_smooth = pd.Series(pos_dm, index=res.index).rolling(window=period, min_periods=period).sum()
    neg_smooth = pd.Series(neg_dm, index=res.index).rolling(window=period, min_periods=period).sum()

    # For subsequent values use Wilder's smoothing (recursive). We'll implement full Wilder smoothing:
    def wilder_smooth(arr, n):
        out = pd.Series(index=arr.index, dtype=float)
        out.iloc[n-1] = arr.iloc[:n].sum()
        for i in range(n, len(arr)):
            out.iloc[i] = out.iloc[i-1] - (out.iloc[i-1] / n) + arr.iloc[i]
        return out

    tr_w = wilder = wilder_smooth(tr.fillna(0), period)
    pos_w = wilder_smooth(pd.Series(pos_dm, index=res.index).fillna(0), period)
    neg_w = wilder_smooth(pd.Series(neg_dm, index=res.index).fillna(0), period)

    plus_di = 100 * pos_w / tr_w
    minus_di = 100 * neg_w / tr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

    # ADX is Wilder-smooth of DX
    adx = wilder_smooth(dx.fillna(0), period)

    res['+DI'] = plus_di
    res['-DI'] = minus_di
    res['DX'] = dx
    res['ADX'] = adx
    return res


# 2) FRAMA (Fractal Adaptive Moving Average) - practical approximation
def FRAMA(df: pd.DataFrame, period: int = 16, fc_min: float = 0.01, fc_max: float = 0.99) -> pd.DataFrame:
    """
    Approximate FRAMA (Fractal Adaptive Moving Average).
    Implementation follows prevalent algorithm: estimate fractal dimension using log ratio of variances
    on halves vs whole and convert to an adaptive smoothing factor.
    Requires: 'close'
    Returns column 'FRAMA'.
    """
    _require_columns(df, {'close'})
    if period <= 2:
        raise ValueError("period must be > 2")

    res = df.copy()
    price = res['close']

    def fractal_dim(window):
        n = len(window)
        if n < 3:
            return np.nan
        half = n // 2
        v1 = np.var(window[:half], ddof=0)
        v2 = np.var(window[half:], ddof=0)
        v = np.var(window, ddof=0)
        # avoid division by zero
        if v == 0 or (v1 + v2) == 0:
            return 0.0
        num = np.log(v1 + v2 + 1e-10) - np.log(v + 1e-10)
        den = np.log(2)
        return num / den

    fd = price.rolling(window=period).apply(fractal_dim, raw=True)
    # map fractal dimension to an alpha between fc_min..fc_max
    # typical mapping: alpha = exp(-k * (fd - 1)) but we use min-max:
    fd_norm = (fd - fd.min()) / (fd.max() - fd.min() + 1e-12)
    alpha = fc_min + (fc_max - fc_min) * (1 - fd_norm)  # more fractal -> smaller alpha
    frama = price.copy() * np.nan
    for i in range(len(price)):
        if i == 0:
            frama.iloc[i] = price.iloc[i]
            continue
        a = alpha.iloc[i] if not np.isnan(alpha.iloc[i]) else fc_min
        frama.iloc[i] = a * price.iloc[i] + (1 - a) * frama.iloc[i-1]
    res['FRAMA'] = frama
    return res


# 3) Parabolic SAR (basic implementation)
def Parabolic_SAR(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
    """
    Compute Parabolic SAR (basic implementation).
    Requires: 'high', 'low', 'close'
    Adds column 'SAR'.
    Note: This is the common discrete algorithm (ep, af updates).
    """
    _require_columns(df, {'high', 'low', 'close'})
    if af_step <= 0 or af_max <= 0:
        raise ValueError("af_step and af_max must be positive")
    res = df.copy()
    high = res['high'].values
    low = res['low'].values
    n = len(res)
    sar = np.zeros(n)
    # initial trend: use first two closes
    trend = 1  # 1 = up, -1 = down; default up
    ep = high[0]  # extreme point
    af = af_step
    sar[0] = low[0]  # start below price
    for i in range(1, n):
        prev_sar = sar[i-1]
        # compute next SAR
        sar[i] = prev_sar + af * (ep - prev_sar)
        if trend == 1:
            # can't be above prior two lows
            sar[i] = min(sar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
            if low[i] < sar[i]:
                # flip to downtrend
                trend = -1
                sar[i] = ep
                ep = low[i]
                af = af_step
        else:
            sar[i] = max(sar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
            if high[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = af_step
        # update EP and AF if trend continues
        if trend == 1:
            if high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)
        else:
            if low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)
    res['SAR'] = sar
    return res


# 4) VIDYA (Variable Dynamic Index Moving Average) - practical variant
def VIDYA(df: pd.DataFrame, period: int = 10, er_period: int = 10) -> pd.DataFrame:
    """
    Practical VIDYA approximation using an Efficiency Ratio (ER) like KAMA but mapping to VIDYA smoothing.
    Requires: 'close'
    Adds 'VIDYA' column.
    """
    _require_columns(df, {'close'})
    if period <= 0 or er_period <= 0:
        raise ValueError("period & er_period must be > 0")
    res = df.copy()
    price = res['close']
    # Efficiency Ratio (ER)
    change = price.diff(er_period).abs()
    volatility = price.diff().abs().rolling(window=er_period).sum()
    er = change / (volatility.replace(0, np.nan))
    er = er.fillna(0.0).clip(0, 1)
    # map ER to smoothing constant similar to VIDYA (scale to alpha)
    fast = 2 / (2 + 1)  # typical fast
    slow = 2 / (30 + 1)  # typical slow
    sc = (er * (fast - slow) + slow)  # smoothing constant
    vidya = price.copy() * np.nan
    for i in range(len(price)):
        if i == 0:
            vidya.iloc[i] = price.iloc[i]
            continue
        a = sc.iloc[i] if not np.isnan(sc.iloc[i]) else slow
        vidya.iloc[i] = a * price.iloc[i] + (1 - a) * vidya.iloc[i-1]
    res['VIDYA'] = vidya
    return res


# 5) MACD
def MACD(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """
    MACD: difference of two EMAs and signal line.
    Requires: 'close'
    Adds columns: 'MACD', 'MACD_signal', 'MACD_hist'
    """
    _require_columns(df, {'close'})
    if not all(p > 0 for p in [fast_period, slow_period, signal_period]):
        raise ValueError("periods must be > 0")
    res = df.copy()
    close = res['close']
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    res['MACD'] = macd
    res['MACD_signal'] = signal
    res['MACD_hist'] = hist
    return res


# 6) TRIX (Triple Exponential Average)
def TRIX(df: pd.DataFrame, period: int = 15, signal: int = 9) -> pd.DataFrame:
    """
    TRIX: percent rate-of-change of a triple-smoothed EMA.
    Requires: 'close'
    Adds 'TRIX' and 'TRIX_signal'.
    """
    _require_columns(df, {'close'})
    if period <= 0 or signal <= 0:
        raise ValueError("period and signal must be > 0")
    res = df.copy()
    price = res['close']
    e1 = price.ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    e3 = e2.ewm(span=period, adjust=False).mean()
    trix = e3.pct_change() * 100.0
    trix_signal = trix.ewm(span=signal, adjust=False).mean()
    res['TRIX'] = trix
    res['TRIX_signal'] = trix_signal
    return res


# 7) Bill Williams Alligator (Jaw, Teeth, Lips)
def Alligator(df: pd.DataFrame, jaw_period: int = 13, teeth_period: int = 8, lips_period: int = 5,
              jaw_shift: int = 8, teeth_shift: int = 5, lips_shift: int = 3) -> pd.DataFrame:
    """
    Bill Williams Alligator: smoothed moving averages (often SMAs of median price) shifted forward.
    Adds 'Alligator_Jaw', 'Alligator_Teeth', 'Alligator_Lips'
    Requires: 'high', 'low'
    """
    _require_columns(df, {'high', 'low'})
    res = df.copy()
    med = (res['high'] + res['low']) / 2.0
    jaw = med.rolling(window=jaw_period).mean().shift(jaw_shift)
    teeth = med.rolling(window=teeth_period).mean().shift(teeth_shift)
    lips = med.rolling(window=lips_period).mean().shift(lips_shift)
    res['Alligator_Jaw'] = jaw
    res['Alligator_Teeth'] = teeth
    res['Alligator_Lips'] = lips
    return res


# 8) Bill Williams Gator Oscillator
def Gator_Oscillator(df: pd.DataFrame, jaw_period: int = 13, teeth_period: int = 8, lips_period: int = 5,
                     jaw_shift: int = 8, teeth_shift: int = 5, lips_shift: int = 3) -> pd.DataFrame:
    """
    Gator Oscillator: two histogram series: abs(Jaw - Teeth) and abs(Teeth - Lips).
    Adds 'Gator_A' and 'Gator_B' (positive values representing magnitude)
    Requires: 'high', 'low'
    """
    _require_columns(df, {'high', 'low'})
    res = Alligator(df, jaw_period, teeth_period, lips_period, jaw_shift, teeth_shift, lips_shift)
    jaw = res['Alligator_Jaw']
    teeth = res['Alligator_Teeth']
    lips = res['Alligator_Lips']
    res['Gator_A'] = (jaw - teeth).abs()
    res['Gator_B'] = (teeth - lips).abs()
    return res


# 9) Awesome Oscillator (example you provided adapted)
def Awesome_Oscillator(df: pd.DataFrame, short_period: int = 5, long_period: int = 34) -> pd.DataFrame:
    """
    Calculate the Bill Williams Awesome Oscillator (AO) and append it to the input DataFrame.

    AO = SMA(Median Price, short_period) - SMA(Median Price, long_period)

    Args:
        df (pd.DataFrame): DataFrame with 'high' and 'low' columns.
        short_period (int): Short period for SMA (default 5).
        long_period (int): Long period for SMA (default 34).

    Returns:
        pd.DataFrame: Input DataFrame with 'AO' column added.
    """
    _require_columns(df, {'high', 'low'})
    if not all(p > 0 for p in [short_period, long_period]):
        raise ValueError("Period values must be positive integers")

    result_df = df.copy()
    median_price = (result_df['high'] + result_df['low']) / 2
    short_sma = median_price.rolling(window=short_period).mean()
    long_sma = median_price.rolling(window=long_period).mean()
    result_df['AO'] = short_sma - long_sma
    return result_df


# 10) Bill Williams MFI (Market Facilitation Index, Bill Williams)
def Bill_Williams_MFI(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bill Williams' Market Facilitation Index (MFI) = (high - low) / volume
    Requires: 'high', 'low', 'volume'
    Adds 'MFI_BW' column
    Note: Volume must be > 0 to have meaningful values.
    """
    _require_columns(df, {'high', 'low', 'volume'})
    res = df.copy()
    vol = res['volume'].replace(0, np.nan)
    res['MFI_BW'] = (res['high'] - res['low']) / vol
    return res


# 11) ATR (Average True Range)
def ATR(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average True Range.
    Requires 'high','low','close'
    Adds 'TR' and 'ATR'
    """
    _require_columns(df, {'high', 'low', 'close'})
    if period <= 0:
        raise ValueError("period must be > 0")
    res = df.copy()
    high = res['high']
    low = res['low']
    close = res['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    res['TR'] = tr
    res['ATR'] = tr.rolling(window=period).mean()  # simple moving average ATR; Wilder can be used if required
    return res


# 12) Standard Deviation (volatility)
def StdDev(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.DataFrame:
    """
    Rolling standard deviation of a column (default 'close').
    Adds 'StdDev'
    """
    _require_columns(df, {column})
    if period <= 0:
        raise ValueError("period must be > 0")
    res = df.copy()
    res['StdDev'] = res[column].rolling(window=period).std(ddof=0)
    return res


# 13) CCI (Commodity Channel Index)
def CCI(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Commodity Channel Index.
    Requires: 'high','low','close'
    Adds 'CCI'
    """
    _require_columns(df, {'high', 'low', 'close'})
    if period <= 0:
        raise ValueError("period must be > 0")
    res = df.copy()
    tp = (res['high'] + res['low'] + res['close']) / 3.0
    ma = tp.rolling(window=period).mean()
    md = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    res['CCI'] = (tp - ma) / (0.015 * md.replace(0, np.nan))
    return res


# 14) RSI (Relative Strength Index)
def RSI(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.DataFrame:
    """
    Relative Strength Index (classic).
    Adds 'RSI'
    """
    _require_columns(df, {column})
    if period <= 0:
        raise ValueError("period must be > 0")
    res = df.copy()
    delta = res[column].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder smoothing
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    res['RSI'] = 100 - (100 / (1 + rs))
    return res


# 15) Bill Williams Accelerator Oscillator (AC)
def Accelerator_Oscillator(df: pd.DataFrame, ao_short: int = 5, ao_long: int = 34, ac_signal: int = 5) -> pd.DataFrame:
    """
    Accelerator Oscillator (AC) = AO - SMA(AO, ac_signal)
    AO = SMA(median, ao_short) - SMA(median, ao_long)
    Requires: 'high','low'
    Adds 'AO' and 'AC'
    """
    _require_columns(df, {'high', 'low'})
    res = df.copy()
    median = (res['high'] + res['low']) / 2.0
    ao = median.rolling(window=ao_short).mean() - median.rolling(window=ao_long).mean()
    ac = ao - ao.rolling(window=ac_signal).mean()
    res['AO'] = ao
    res['AC'] = ac
    return res


# 16) Bollinger Bands
def Bollinger_Bands(df: pd.DataFrame, period: int = 20, n_std: float = 2.0, column: str = 'close') -> pd.DataFrame:
    """
    Bollinger Bands: mid = SMA(period), upper = mid + n_std*std, lower = mid - n_std*std
    Adds 'BB_mid','BB_upper','BB_lower','BB_width'
    """
    _require_columns(df, {column})
    if period <= 0:
        raise ValueError("period must be > 0")
    res = df.copy()
    mid = res[column].rolling(window=period).mean()
    std = res[column].rolling(window=period).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    res['BB_mid'] = mid
    res['BB_upper'] = upper
    res['BB_lower'] = lower
    res['BB_width'] = (upper - lower) / mid.replace(0, np.nan)
    return res


# 17) Bill Williams Fractals (bull/bear fractal boolean)
def Bill_Williams_Fractals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify Bill Williams fractals:
    - bullish fractal: low with two higher lows on each side (pattern: low[i-2] > low[i-1] > low[i] < low[i+1] < low[i+2])
    - bearish fractal: high with two lower highs on each side
    Adds boolean columns 'fract_bull' and 'fract_bear'
    Requires: 'high','low'
    """
    _require_columns(df, {'high', 'low'})
    res = df.copy()
    low = res['low']
    high = res['high']
    n = len(res)
    bull = pd.Series(False, index=res.index)
    bear = pd.Series(False, index=res.index)
    for i in range(2, n-2):
        # bullish fractal: low in middle is lower than two on each side
        if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i-2] and low.iloc[i] < low.iloc[i+1] and low.iloc[i] < low.iloc[i+2]:
            bull.iloc[i] = True
        # bearish fractal
        if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i-2] and high.iloc[i] > high.iloc[i+1] and high.iloc[i] > high.iloc[i+2]:
            bear.iloc[i] = True
    res['fract_bull'] = bull
    res['fract_bear'] = bear
    return res


# 18) Williams %R
def Williams_R(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Williams %R = (HighestHigh - Close) / (HighestHigh - LowestLow) * -100
    Requires: 'high','low','close'
    Adds 'WilliamsR'
    """
    _require_columns(df, {'high', 'low', 'close'})
    if period <= 0:
        raise ValueError("period must be > 0")
    res = df.copy()
    highest = res['high'].rolling(window=period).max()
    lowest = res['low'].rolling(window=period).min()
    res['WilliamsR'] = -100 * (highest - res['close']) / (highest - lowest).replace(0, np.nan)
    return res


# 19) Ichimoku (Kijun-sen + Senkou Spans basic)
def Ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52, shift: int = 26) -> pd.DataFrame:
    """
    Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B, and Chikou span.
    Requires: 'high','low','close'
    Adds: 'tenkan','kijun','senkou_a','senkou_b','chikou'
    Note: senkou spans are shifted forward by `shift`
    """
    _require_columns(df, {'high', 'low', 'close'})
    if not (tenkan > 0 and kijun > 0 and senkou_b > 0 and shift >= 0):
        raise ValueError("periods must be positive")
    res = df.copy()
    high = res['high']
    low = res['low']
    close = res['close']
    tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
    kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(shift)
    senkou_b = ((high.rolling(window=senkou_b).max() + low.rolling(window=senkou_b).min()) / 2).shift(shift)
    chikou = close.shift(-shift)
    res['tenkan'] = tenkan_sen
    res['kijun'] = kijun_sen
    res['senkou_a'] = senkou_a
    res['senkou_b'] = senkou_b
    res['chikou'] = chikou
    return res


# 20) Envelopes (simple SMA-based bands)
def Envelopes(df: pd.DataFrame, period: int = 20, pct: float = 0.02, column: str = 'close') -> pd.DataFrame:
    """
    Envelopes: upper = SMA*(1 + pct), lower = SMA*(1 - pct)
    Adds 'ENV_mid','ENV_upper','ENV_lower'
    """
    _require_columns(df, {column})
    if period <= 0 or pct < 0:
        raise ValueError("period must be > 0 and pct >= 0")
    res = df.copy()
    mid = res[column].rolling(window=period).mean()
    res['ENV_mid'] = mid
    res['ENV_upper'] = mid * (1 + pct)
    res['ENV_lower'] = mid * (1 - pct)
    return res
