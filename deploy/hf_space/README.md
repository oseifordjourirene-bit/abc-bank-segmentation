---
title: ABC Bank Drift API
emoji: 📊
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Customer drift scoring API (BAN6800 capstone, proxy data)
---

# ABC Bank customer drift API

POST `/predict` scores an active customer's risk of drifting within three months and returns
the decision, the contact tier and the top factors behind the score. GET `/explanations/{request_id}`
returns a stored decision for audit. Open `/docs` to try it.

Trained on public Santander data (Kaggle, 2016) as a stand-in for ABC Bank: a prototype, not
approved for live decisions. The API refuses age, sex, nationality and province (HTTP 422).
Code: https://github.com/oseifordjourirene-bit/abc-bank-segmentation
