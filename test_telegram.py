"""Telegram bağlantısını tek başına test eder. Kullanım: python test_telegram.py"""
import os, httpx
token = os.getenv("TELEGRAM_TOKEN") or input("Bot token: ").strip()
chat = os.getenv("TELEGRAM_CHAT_ID") or input("Chat ID: ").strip()
r = httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
               json={"chat_id": chat, "text": "XNOR Guard test mesajı"}, timeout=10)
j = r.json()
print("HTTP:", r.status_code, "| ok:", j.get("ok"), "|", j.get("description", "mesaj gönderildi"))
