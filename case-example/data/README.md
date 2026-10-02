# Data Guide - SMS Scam Detection (case-example)

You are practicing carrier-style **smishing risk scoring from soft signals**: message text style + sender behaviour. No number blocklist. No link-reputation lookups.

---

## Bundled file (in this folder)

| File | Rows | What it is |
|---|---|---|
| `sms_scam_joined.csv` | 3,000 messages | **Start here.** One row per message: text + sender behaviour + `label_scam` |

### Columns

- `message_id` - row id (not a feature)
- `text` - message text (fictional templates modelled on the kinds of messages in the public UCI SMS Spam Collection)
- `sender_account_age_days` - how old the sending number is
- `msgs_sent_last_hour` - velocity: messages from this sender in the last hour
- `unique_recipients_24h` - distinct recipients in the last 24h
- `pct_recipients_not_in_contacts` - share of recipients who don't have the sender saved
- `reply_rate_7d` - 7-day reply rate for this sender (slow, batch-computed feature)
- `sent_hour` - hour of day the message was sent
- `label_scam` - 1 = scam, else 0 (the answer key; never train on it as a feature)
- `message_family` - scam type (`prize`, `parcel_fee`, `bank_phish`, `tax_refund`, `hi_mum`, `job_offer`, `wrong_number`) or legit type (`personal`, `business_notification`, `marketing_legit`)

### Labels

- `label_scam` = 1 for scams (~13% of rows), else 0.
- The behaviour columns are **synthetic practice data**. The text is fictional templates modelled on the kinds of messages in the public UCI SMS Spam Collection (2011, UK/Singapore SMS).

---

## Where the data comes from

- Text style: fictional templates modelled on Almeida, Gomez Hidalgo & Yamakami (2011), *Contributions to the Study of SMS Spam Filtering* - UCI ML Repository: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Behaviour columns: synthetic, generated for this hackathon lab (MIT with the rest of the repo). Do not present them as carrier production telemetry.

---

## Loading example

```python
import pandas as pd

df = pd.read_csv("data/sms_scam_joined.csv")
print(df["label_scam"].value_counts(normalize=True))
print(df[["text", "msgs_sent_last_hour", "label_scam"]].head())
```
