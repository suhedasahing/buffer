"""Kullanım: önce botuna Telegram'dan /start yaz, sonra: python get_chat_id.py"""
import os, httpx
token = os.getenv("TELEGRAM_TOKEN") or input("Bot token: ").strip()
r = httpx.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10).json()
chats = {(m["message"]["chat"]["id"], m["message"]["chat"].get("first_name") or m["message"]["chat"].get("title"))
         for m in r.get("result", []) if "message" in m}
print(chats or "Boş. Önce botuna Telegram'dan /start yaz, sonra tekrar çalıştır.")
