# Data Anonymisation Plan

Implemented in `src/anonymise.py`; tested in `tests/test_anonymise.py`.

## PII inventory and treatment

| Column | Type of data | Treatment |
| --- | --- | --- |
| `customer_id` | Direct identifier | Replaced by `customer_key`: salted HMAC-SHA256 hash, stable across months, irreversible without the secret |
| Names, email, phone, address, Ghana Card, account number | Direct identifiers (not in the Santander data; expected in ABC Bank data) | Dropped automatically if present (tested with Faker-generated PII) |
| `age` | Quasi-identifier, protected | Exact age removed; only the age band kept, in the restricted file |
| `sex`, `foreign_born_flag`, `country_of_residence` | Protected attributes | Moved to the restricted audit file; never model inputs |
| `province_name`, `province_code` | Quasi-identifier | Province code dropped; provinces with fewer than 500 customers merged into OTHER |
| `join_date` | Quasi-identifier | Reduced to join year |
| `last_primary_date`, `spouse_of_employee` | Near-empty | Dropped in cleaning |

## Secret handling

The hashing secret (`ANON_SALT`) is set as an environment variable at run time. It is never written to code, GitHub or the Docker image. Without it, keys cannot be linked back to customer IDs.

## Re-identification check (k-anonymity)

Measured on the latest month of the restricted audit file, over age band, sex and province.

| Step | Smallest group (k) |
| --- | --- |
| Before generalisation | 1 (some customers unique) |
| After merging small provinces and suppressing 60 at-risk customers (0.1%) | **5** |

Target: k ≥ 5 on every run, reported in the audit log.
