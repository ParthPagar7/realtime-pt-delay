import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

RNG = np.random.RandomState(42)
routes = [f"R{r}" for r in range(1, 6)]
stops = [f"S{s}" for s in range(1, 11)]

def make_row(base_dt, route, stop):
    scheduled_min = base_dt.hour * 60 + base_dt.minute
    day_of_week = base_dt.weekday()
    time_of_day = base_dt.hour + base_dt.minute / 60.0
    weather_temp = 15 + 10 * np.sin((base_dt.timetuple().tm_yday/365.0)*2*np.pi) + RNG.normal(0,2)
    weather_precip = max(0, RNG.exponential(0.1) - (0.1 if day_of_week<5 else 0))
    traffic_index = 1 + 0.5 * (8 <= base_dt.hour <= 10 or 17 <= base_dt.hour <= 19)
    route_effect = routes.index(route) * 0.5
    stop_effect = stops.index(stop) * 0.2
    base_delay = RNG.normal(0 + route_effect + stop_effect + 2*traffic_index, 3.0)
    if RNG.rand() < 0.02:
        base_delay += RNG.uniform(10, 40)
    delay = float(round(base_delay, 2))
    return {
        "timestamp": base_dt.isoformat(),
        "route_id": route,
        "stop_id": stop,
        "day_of_week": day_of_week,
        "time_of_day": time_of_day,
        "weather_temp": round(weather_temp,2),
        "weather_precip": round(weather_precip,3),
        "traffic_index": traffic_index,
        "scheduled_minute_of_day": scheduled_min,
        "delay_min": delay
    }

def generate_history(start_date, days=30, per_hour=10):
    rows = []
    start_dt = datetime.fromisoformat(start_date)
    for d in range(days):
        day = start_dt + timedelta(days=d)
        for hour in range(0,24):
            for i in range(per_hour):
                minute = int((i / per_hour) * 60)
                dt = day + timedelta(hours=hour, minutes=minute)
                route = random.choice(routes)
                stop = random.choice(stops)
                rows.append(make_row(dt, route, stop))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_history("2025-09-01T00:00:00", days=60, per_hour=6)
    df.to_csv("historical_delays.csv", index=False)
    print("Wrote historical_delays.csv ({} rows)".format(len(df)))

    last_dt = datetime.fromisoformat(df['timestamp'].max()) + timedelta(minutes=5)
    live_rows = []
    for i in range(50):
        rt = last_dt + timedelta(minutes=5*i)
        live_rows.append(make_row(rt, random.choice(routes), random.choice(stops)))
    pd.DataFrame(live_rows).to_csv("live_feed.csv", index=False)
    print("Wrote live_feed.csv ({} rows)".format(len(live_rows)))
