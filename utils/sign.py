import bittensor as bt
import time

def signature_hex(wallet: bt.Wallet, message: str) -> str:
    signed = wallet.hotkey.sign(message.encode("utf-8"))
    if isinstance(signed, bytes):
        return signed.hex()
    if hasattr(signed, "hex"):
        return signed.hex()
    return str(signed)

def signed_subnet_headers(wallet: bt.Wallet, message: str = "") -> dict[str, str]:
    hotkey = wallet.hotkey.ss58_address
    timestamp = str(int(time.time()))
    signature = signature_hex(wallet, f"<Bytes>{message}:{timestamp}:{hotkey}</Bytes>")
    return {
        "X-Hotkey": hotkey,
        "X-Timestamp": timestamp,
        "X-Signature": signature,
    }
