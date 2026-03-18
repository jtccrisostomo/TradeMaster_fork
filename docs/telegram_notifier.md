# Telegram Notifier for Slurm Logs

This helper streams your active Slurm job status and tail output to a Telegram chat.

## Prerequisites

- Create a Telegram bot and obtain `BOT_TOKEN`.
- Get your `CHAT_ID` (start a chat with the bot and read `chat.id` from `getUpdates`).
- Install dependencies: `pip install requests` (already in this repo's requirements).

## Usage

```bash
cd /home/laperia/TradeMaster
export TM_TELEGRAM_BOT_TOKEN="<bot token>"
export TM_TELEGRAM_CHAT_ID="<chat id>"
# optional overrides
export TM_SLURM_LOG_DIR="/path/to/slurm_logs"          # default: repo/slurm_logs
export TM_SLURM_USER="$USER"                           # default: $USER
export TM_NOTIFY_POLL_SECS=60
export TM_NOTIFY_PREFIXES="tm-eiie-batch,tm-eiie-sweep" # default prefixes

python scripts/telegram_notifier.py
# or dry run (prints messages instead of sending)
python scripts/telegram_notifier.py --dry-run
```

### Flags

- `--bot-token`, `--chat-id`: override env vars for credentials.
- `--log-dir`: point to a different Slurm logs folder.
- `--user`: query another user in `squeue`.
- `--prefix`: add/replace log prefixes to monitor; repeatable.
- `--poll-secs`: adjust polling interval.
- `--dry-run`: print messages only.

## What it sends

- A snapshot of `squeue` whenever it changes.
- New lines appended to matching `*.out`/`*.err` files for active jobs (prefixed with a computer or warning icon).

## Notes

- Messages are trimmed to stay under Telegram's 4096-character limit.
- Keep your bot token **out of version control**; use environment variables or a secrets store.
