import os
from pathlib import Path

import dotenv
import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.utils import send_email


@pytest.mark.slow
def test_live_send_email_from_dotenv():
    if os.getenv("RUN_SMTP_LIVE_TEST") != "1":
        pytest.skip("Set RUN_SMTP_LIVE_TEST=1 to run the live SMTP smoke test.")

    dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    required = ["SENDER", "RECEIVER", "SENDER_PASSWORD"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        pytest.skip(f"Missing required env vars for live SMTP test: {', '.join(missing)}")

    config = OmegaConf.create(
        {
            "email": {
                "sender": os.getenv("SENDER"),
                "receiver": os.getenv("RECEIVER"),
                "smtp_server": os.getenv("SMTP_SERVER", "smtp.qq.com"),
                "smtp_port": int(os.getenv("SMTP_PORT", "465")),
                "sender_password": os.getenv("SENDER_PASSWORD"),
            }
        }
    )

    send_email(config, "<html><body>SMTP live smoke test.</body></html>")
