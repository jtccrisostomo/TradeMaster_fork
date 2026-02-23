#!/usr/bin/env python3
"""
Test script to verify board lot implementation
"""
import numpy as np

# Simulate board lot implementation
def _get_pse_board_lot(price: float) -> tuple:
    """Get PSE board lot size and tick size based on price."""
    if price < 0.0099:
        return 1000000, 0.0001
    elif price < 0.2490:
        return 10000, 0.0010
    elif price < 0.4950:
        return 10000, 0.0050
    elif price < 4.9900:
        return 1000, 0.0100
    elif price < 9.9900:
        return 100, 0.0100
    elif price < 19.9800:
        return 100, 0.0200
    elif price < 49.9500:
        return 100, 0.0500
    elif price < 99.9500:
        return 100, 0.0500
    elif price < 199.9000:
        return 10, 0.1000
    elif price < 499.8000:
        return 10, 0.2000
    elif price < 999.5000:
        return 10, 0.5000
    elif price < 1999.0000:
        return 5, 1.0000
    elif price < 4998.0000:
        return 5, 2.0000
    else:
        return 5, 5.0000

def _apply_board_lot_rounding(weights, prices, portfolio_value):
    """Apply PSE board-lot rounding to convert weights to executable shares."""
    cash_weight = weights[0]
    asset_weights = weights[1:]

    # Convert weights to target values and shares
    target_values = asset_weights * portfolio_value
    target_shares = np.floor(target_values / (prices + 1e-12))

    # Apply board-lot rounding per ticker
    rounded_shares = np.zeros_like(target_shares)
    lot_sizes = np.zeros_like(target_shares)
    tick_sizes = np.zeros_like(prices)

    for i, (price, shares) in enumerate(zip(prices, target_shares)):
        lot_size, tick_size = _get_pse_board_lot(price)
        lot_sizes[i] = lot_size
        tick_sizes[i] = tick_size

        # Round down to nearest board lot
        if shares >= lot_size:
            rounded_shares[i] = int(shares // lot_size) * lot_size
        else:
            rounded_shares[i] = 0.0

    # Convert rounded shares back to executed values
    executed_values = rounded_shares * prices
    executed_asset_value = float(np.sum(executed_values))

    # Cash is residual
    executed_cash = portfolio_value - executed_asset_value

    # Normalize to weights
    if portfolio_value > 0:
        executed_cash_weight = executed_cash / portfolio_value
        executed_asset_weights = executed_values / portfolio_value
    else:
        executed_cash_weight = 1.0
        executed_asset_weights = np.zeros_like(asset_weights)

    executed_weights = np.concatenate(([executed_cash_weight], executed_asset_weights))

    return executed_weights, {
        "target_shares": target_shares.tolist(),
        "rounded_shares": rounded_shares.tolist(),
        "lot_sizes": lot_sizes.tolist(),
        "tick_sizes": tick_sizes.tolist(),
        "target_values": target_values.tolist(),
        "executed_values": executed_values.tolist(),
        "shares_truncated": (target_shares - rounded_shares).tolist(),
        "value_lost": float(np.sum((target_shares - rounded_shares) * prices)),
        "tickers_below_lot": int(np.sum(target_shares > 0) - np.sum(rounded_shares > 0)),
    }

# Test scenario: 100k initial, 3 tickers
print("=" * 60)
print("Test: Initial ₱100,000 with 3 tickers")
print("=" * 60)

# Input weights: 10% cash, 30% each for 3 assets
weights = np.array([0.10, 0.30, 0.40, 0.20])
portfolio_value = 100000.0

# Simulated prices
prices = np.array([50.0, 150.0, 5.0])

print(f"\nInput weights: {weights}")
print(f"Cash weight: {weights[0]} = ₱{weights[0] * portfolio_value:,.0f}")
print(f"Asset weights: {weights[1:]}")
print(f"Portfolio value: ₱{portfolio_value:,.0f}")
print(f"Prices: ₱{prices}")
print()

# Apply board lot rounding
executed_weights, diagnostics = _apply_board_lot_rounding(weights, prices, portfolio_value)

print(f"\n--- After Board Lot Rounding ---")
print(f"\nTarget shares (before rounding):")
print(f"  Ticker 1 (₱50): {diagnostics['target_shares'][0]:.0f}")
print(f"  Ticker 2 (₱150): {diagnostics['target_shares'][1]:.0f}")
print(f"  Ticker 3 (₱5): {diagnostics['target_shares'][2]:.0f}")

print(f"\nRounded shares (after rounding):")
print(f"  Ticker 1: {diagnostics['rounded_shares'][0]:.0f}")
print(f"  Ticker 2: {diagnostics['rounded_shares'][1]:.0f}")
print(f"  Ticker 3: {diagnostics['rounded_shares'][2]:.0f}")

print(f"\nExecuted values:")
print(f"  Ticker 1: ₱{diagnostics['executed_values'][0]:,.0f}")
print(f"  Ticker 2: ₱{diagnostics['executed_values'][1]:,.0f}")
print(f"  Ticker 3: ₱{diagnostics['executed_values'][2]:,.0f}")

print(f"\nExecuted weights:")
print(f"  Cash: {executed_weights[0]:.2%}")
print(f"  Ticker 1: {executed_weights[1]:.2%}")
print(f"  Ticker 2: {executed_weights[2]:.2%}")
print(f"  Ticker 3: {executed_weights[3]:.2%}")

print(f"\nDiagnostics:")
print(f"  Shares truncated: {diagnostics['shares_truncated']}")
print(f"  Value lost: ₱{diagnostics['value_lost']:,.0f}")
print(f"  Tickers below lot: {diagnostics['tickers_below_lot']}")

# Calculate actual cash
executed_asset_value = float(np.sum(diagnostics['executed_values']))
executed_cash = portfolio_value - executed_asset_value
print(f"\nActual cash: ₱{executed_cash:,.0f} (should be ₱10,000)")
print(f"Expected cash: ₱10,000 (10% of ₱100,000)")

# Check if cash is split across assets
print(f"\n--- Verification ---")
if executed_cash == 10000.0:
    print("✅ Cash is ₱10,000 (NOT split across assets)")
else:
    print(f"❌ Cash is ₱{executed_cash:,.0f} (SPLIT across assets)")
    print(f"   If split equally: ₱{executed_cash/3:,.0f} per ticker")
    print(f"   Cash weight per ticker: {executed_weights[0]/3:.2%}")
