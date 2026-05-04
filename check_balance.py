from pybit.unified_trading import HTTP
import os
from dotenv import load_dotenv
load_dotenv()

s = HTTP(testnet=True, api_key=os.getenv('BYBIT_API_KEY'), api_secret=os.getenv('BYBIT_API_SECRET'))

print("=== UNIFIED ===")
print(s.get_wallet_balance(accountType='UNIFIED'))

print("=== FUND ===")
print(s.get_wallet_balance(accountType='FUND'))