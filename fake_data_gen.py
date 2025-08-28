import csv, random, datetime as dt, argparse, sys

# ---------------------------------------------------------------------------
# Dynamic week generation utilities
# ---------------------------------------------------------------------------

def ensure_monday(d: dt.date) -> dt.date:
    """Return the Monday of the week containing date d (if d already Monday returns d)."""
    return d - dt.timedelta(days=d.weekday())

def generate_mondays(start: dt.date, weeks: int) -> list[dt.date]:
    start_monday = ensure_monday(start)
    return [start_monday + dt.timedelta(weeks=w) for w in range(weeks)]

def base_ms(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day).timestamp() * 1000)

# ---------------------------------------------------------------------------
# Session generation
# ---------------------------------------------------------------------------

PRODUCTS = [
    ("sku123", "Running Shoe Model A", 74.99),
    ("sku456", "Trail Shoe Model B", 84.99),
    ("sku789", "Performance Tee", 19.99),
    ("sku900", "Sports Cap", 24.99),
    ("sku1000", "Yoga Mat", 29.99),
    ("sku1001", "Foam Roller", 29.49),
]

CATEGORIES = [
    ("shoes", "Shoes Category"),
    ("apparel", "Apparel Category"),
    ("equipment", "Gym Equipment"),
    ("accessories", "Accessories Category"),
]

def choose_product(week_idx: int) -> tuple[str,str,float]:
    # Slight deterministic preference per week while keeping variety
    random.seed(week_idx + 1337)
    return random.choice(PRODUCTS)

def build_event_flow(user_id: int, week_idx: int) -> list[str]:
    """Create an event flow; occasional abandonment for variation."""
    purchase = (user_id + week_idx) % 4 != 0
    base = ["session_start","page_view","view_item","add_to_cart","begin_checkout"]
    base.append("purchase" if purchase else "abandon_checkout")
    return base

def make_sessions_for_user(user_id: int, mondays: list[dt.date]) -> list[list]:
    rows: list[list] = []
    for week_idx, monday in enumerate(mondays, start=1):
        sess_id = f"s_{user_id:03d}_{week_idx:02d}"
        start_ts = base_ms(monday) + user_id * 1000  # offset per user
        events = build_event_flow(user_id, week_idx)
        unique = set()
        chosen_product = choose_product(week_idx)
        for step, ev in enumerate(events):
            ts = start_ts + step * 6000
            rev = 0.0
            unique_items = len(unique)
            qty = 0
            page_loc = "https://example.com/"
            page_title = "Home"
            if ev == "page_view":
                cat_slug, cat_title = random.choice(CATEGORIES)
                page_loc = f"https://example.com/category/{cat_slug}"
                page_title = cat_title
            if ev in ("view_item","add_to_cart"):
                sku, title, price = chosen_product
                page_loc = f"https://example.com/product/{sku}"
                page_title = title
                unique.add(sku)
                unique_items = len(unique)
                qty = 1 if ev == "add_to_cart" else 1
            if ev == "begin_checkout":
                page_loc = "https://example.com/cart"
                page_title = "Cart"
                qty = len(unique)
            if ev in ("purchase","abandon_checkout"):
                page_loc = "https://example.com/checkout" + ("/complete" if ev=="purchase" else "")
                page_title = "Order Confirmation" if ev=="purchase" else "Checkout"
                if ev == "purchase":
                    # sum price for all unique items (here 1) but allow extension
                    rev = sum(price for _ in unique for __,__,price in [chosen_product])
                    qty = len(unique)
            rows.append([
                f"u_{user_id:03d}", f"{sess_id}_w{week_idx}",
                monday.isoformat(), ts, ev,
                f"{rev:.2f}", unique_items, qty, page_loc, page_title
            ])
    return rows

# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

from typing import Optional

def main(out: Optional[str] = None, users: int = 10, weeks: int = 2, start_date: str = "2025-08-25"):
    try:
        start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    except ValueError:
        print(f"Invalid start_date {start_date}; expected YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)
    if weeks < 1:
        print("weeks must be >= 1", file=sys.stderr)
        sys.exit(1)
    mondays = generate_mondays(start, weeks)
    if out is None:
        out = f"fake-{weeks}-weeks.csv"

    header = ["user_pseudo_id","session_id","date_formatted","event_timestamp","event_name",
              "rev_usd","unique_items","qty","page_location","page_title"]
    all_rows = [header]
    for u in range(1, users + 1):
        all_rows.extend(make_sessions_for_user(u, mondays))
    with open(out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(all_rows)
    print(f"Wrote {out} weeks={weeks} users={users} rows={len(all_rows)-1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fake weekly session event data.")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: fake-<weeks>-weeks.csv)")
    parser.add_argument("--users", type=int, default=10, help="Number of users")
    parser.add_argument("--weeks", type=int, default=2, help="Number of weeks to generate")
    parser.add_argument("--start-date", type=str, default="2025-08-25", help="Start date (any day in week 1) YYYY-MM-DD")
    args = parser.parse_args()
    main(out=args.out, users=args.users, weeks=args.weeks, start_date=args.start_date)