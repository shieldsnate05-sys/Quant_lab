"""
Quant-Lab Alpaca Downloader.

Concrete :class:`~data.loader.DataLoader` backed by the Alpaca Market
Data API (via ``alpaca-py``).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.models.bars import BarSet
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame
from alpaca.data.timeframe import TimeFrameUnit as AlpacaTimeFrameUnit

from config.logging_config import get_logger
from config.settings import AlpacaSettings, settings
from core.enums import TimeFrame
from core.exceptions import ConfigurationError, DataError
from core.types import OHLCVFrame, Symbol
from data.loader import DataLoader
from data.schema import SCHEMA

logger = get_logger(__name__)

#: Maps Quant-Lab's platform-neutral timeframe to Alpaca's SDK timeframe.
_TIMEFRAME_MAP: dict[TimeFrame, AlpacaTimeFrame] = {
    TimeFrame.ONE_MINUTE: AlpacaTimeFrame(1, AlpacaTimeFrameUnit.Minute),
    TimeFrame.FIVE_MINUTE: AlpacaTimeFrame(5, AlpacaTimeFrameUnit.Minute),
    TimeFrame.FIFTEEN_MINUTE: AlpacaTimeFrame(15, AlpacaTimeFrameUnit.Minute),
    TimeFrame.THIRTY_MINUTE: AlpacaTimeFrame(30, AlpacaTimeFrameUnit.Minute),
    TimeFrame.ONE_HOUR: AlpacaTimeFrame(1, AlpacaTimeFrameUnit.Hour),
    TimeFrame.DAILY: AlpacaTimeFrame(1, AlpacaTimeFrameUnit.Day),
}

_FEED_MAP: dict[str, DataFeed] = {
    "iex": DataFeed.IEX,
    "sip": DataFeed.SIP,
    "delayed_sip": DataFeed.DELAYED_SIP,
    "otc": DataFeed.OTC,
}


class AlpacaDownloader(DataLoader):
    """
    Fetches OHLCV bars from the Alpaca Market Data API.

    Parameters
    ----------
    alpaca_settings : config.settings.AlpacaSettings, optional
        API credentials and feed configuration. Defaults to
        ``config.settings.settings.alpaca``.

    Raises
    ------
    core.exceptions.ConfigurationError
        If no API key/secret is configured.
    """

    def __init__(self, alpaca_settings: AlpacaSettings | None = None) -> None:
        self._settings = alpaca_settings or settings.alpaca

        if not self._settings.api_key or not self._settings.secret_key:
            raise ConfigurationError(
                "Alpaca API credentials are not configured. Set "
                "config.settings.settings.alpaca.api_key and secret_key "
                "(e.g. via environment variables)."
            )

        self._client = StockHistoricalDataClient(
            api_key=self._settings.api_key,
            secret_key=self._settings.secret_key,
        )

    def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> OHLCVFrame:
        """
        Fetch OHLCV bars for ``symbol`` between ``start`` and ``end``.

        Parameters
        ----------
        symbol : core.types.Symbol
            Ticker symbol to fetch.
        timeframe : core.enums.TimeFrame
            Bar timeframe to fetch.
        start : datetime.datetime
            Inclusive start of the date range.
        end : datetime.datetime
            Inclusive end of the date range.

        Returns
        -------
        core.types.OHLCVFrame
            OHLCV bars indexed by a timezone-aware ``DatetimeIndex``.

        Raises
        ------
        core.exceptions.DataError
            If the Alpaca API request fails or returns no data.
        """
        logger.info(
            "Fetching Alpaca bars: symbol=%s timeframe=%s start=%s end=%s",
            symbol,
            timeframe.value,
            start,
            end,
        )

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_TIMEFRAME_MAP[timeframe],
            start=start,
            end=end,
            adjustment=(
                Adjustment.ALL if self._settings.paper_trading else Adjustment.RAW
            ),
            feed=_FEED_MAP.get(self._settings.data_feed, DataFeed.IEX),
        )

        try:
            bar_set = self._client.get_stock_bars(request)
        except Exception as exc:
            raise DataError(f"Alpaca request failed for {symbol}: {exc}") from exc

        if not isinstance(bar_set, BarSet):
            raise DataError(
                f"Alpaca returned raw data instead of a BarSet for {symbol}."
            )

        frame = bar_set.df

        if frame.empty:
            raise DataError(
                f"Alpaca returned no bars for {symbol} in range [{start}, {end}]."
            )

        if isinstance(frame.index, pd.MultiIndex):
            frame = pd.DataFrame(frame.xs(symbol, level="symbol"))

        frame = frame.rename(
            columns={
                "open": SCHEMA.open,
                "high": SCHEMA.high,
                "low": SCHEMA.low,
                "close": SCHEMA.close,
                "volume": SCHEMA.volume,
            }
        )[list(SCHEMA.columns)]
        frame.index.name = SCHEMA.timestamp

        logger.info("Fetched %d bars for %s.", len(frame), symbol)

        return frame
