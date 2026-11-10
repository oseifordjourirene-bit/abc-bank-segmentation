# Data Dictionary (preliminary)

Source: Santander Product Recommendation, Kaggle (2016), file `train_ver2.csv`. Original Spanish column names are renamed to English in the cleaning step (`src/`). One row = one customer in one month.

**Roles:** *Feature* = clustering or model input; *Audit only* = protected or proxy attribute used only for fairness checks; *Filter* = used to remove records; *Exclude* = dropped.

## Customer attributes

| Original | English name | Type | Description | Role |
| --- | --- | --- | --- | --- |
| `fecha_dato` | `snapshot_date` | date | Month of the snapshot (2015-01-28 to 2016-05-28) | Key |
| `ncodpers` | `customer_id` | integer | Customer code | Key |
| `ind_empleado` | `employee_status` | category | A active, B ex-employee, F filial, N not employee, P passive | Filter (remove bank staff) |
| `pais_residencia` | `country_of_residence` | category | Customer's country of residence | Audit only |
| `sexo` | `sex` | category | H / V | Audit only |
| `age` | `age` | integer | Age in years (loads as text; clean) | Audit only |
| `fecha_alta` | `join_date` | date | Date the customer first held a contract | Feature (derive) |
| `ind_nuevo` | `new_customer_flag` | binary | 1 if joined in the last 6 months | Feature |
| `antiguedad` | `tenure_months` | integer | Customer seniority in months (loads as text; clean) | Feature |
| `indrel` | `primary_customer_flag` | category | 1 primary; 99 primary at start but not end of month | Feature |
| `ult_fec_cli_1t` | `last_primary_date` | date | Last date as primary customer, if lapsed | Feature (lapse signal) |
| `indrel_1mes` | `customer_type_start_month` | category | 1 primary, 2 co-owner, P potential, 3 former primary, 4 former co-owner | Feature |
| `tiprel_1mes` | `relation_type_start_month` | category | A active, I inactive, P former, R potential | Feature |
| `indresi` | `resident_flag` | binary | Resides in bank's country (S/N) | Exclude |
| `indext` | `foreign_born_flag` | binary | Born outside bank's country (S/N) | Audit only |
| `conyuemp` | `spouse_of_employee` | binary | Spouse of an employee (mostly empty) | Exclude |
| `canal_entrada` | `joining_channel` | category | Channel used to join | Feature (proxy check) |
| `indfall` | `deceased_flag` | binary | Deceased (S/N) | Filter |
| `tipodom` | `address_type` | category | Primary address flag (constant) | Exclude |
| `cod_prov` | `province_code` | category | Province code | Audit only |
| `nomprov` | `province_name` | category | Province name | Audit only |
| `ind_actividad_cliente` | `active_customer_flag` | binary | 1 active, 0 inactive | Feature |
| `renta` | `household_income` | float | Gross household income (many missing) | Feature (proxy check) |
| `segmento` | `bank_segment` | category | 01 VIP, 02 individuals, 03 college graduates | Baseline comparison |

## Product holdings (1 = held that month, 0 = not held)

| Original | English name | Type | Role |
| --- | --- | --- | --- |
| `ind_ahor_fin_ult1` | `savings_account` | binary | Feature |
| `ind_aval_fin_ult1` | `guarantees` | binary | Feature |
| `ind_cco_fin_ult1` | `current_account` | binary | Feature |
| `ind_cder_fin_ult1` | `derivatives_account` | binary | Feature |
| `ind_cno_fin_ult1` | `payroll_account` | binary | Feature |
| `ind_ctju_fin_ult1` | `junior_account` | binary | Feature |
| `ind_ctma_fin_ult1` | `current_account_mas_particular` | binary | Feature |
| `ind_ctop_fin_ult1` | `current_account_particular` | binary | Feature |
| `ind_ctpp_fin_ult1` | `current_account_particular_plus` | binary | Feature |
| `ind_deco_fin_ult1` | `short_term_deposit` | binary | Feature |
| `ind_deme_fin_ult1` | `medium_term_deposit` | binary | Feature |
| `ind_dela_fin_ult1` | `long_term_deposit` | binary | Feature |
| `ind_ecue_fin_ult1` | `e_account` | binary | Feature |
| `ind_fond_fin_ult1` | `funds` | binary | Feature |
| `ind_hip_fin_ult1` | `mortgage` | binary | Feature |
| `ind_plan_fin_ult1` | `pension_plan` | binary | Feature |
| `ind_pres_fin_ult1` | `loan` | binary | Feature |
| `ind_reca_fin_ult1` | `taxes` | binary | Feature |
| `ind_tjcr_fin_ult1` | `credit_card` | binary | Feature |
| `ind_valo_fin_ult1` | `securities` | binary | Feature |
| `ind_viv_fin_ult1` | `home_account` | binary | Feature |
| `ind_nomina_ult1` | `salary_credited` | binary | Feature |
| `ind_nom_pens_ult1` | `pension_credited` | binary | Feature |
| `ind_recibo_ult1` | `direct_debit` | binary | Feature |

The three `current_account_*` variants are grouped with `current_account` as one current-account indicator.

## Derived features (Module 3)

| Feature | Definition |
| --- | --- |
| `product_count` | Number of products held in the month |
| `products_added_3m` / `products_dropped_3m` | Products gained or lost over the last three months |
| `months_active_6m` | Months flagged active in the last six |
| `has_salary`, `has_term_deposit`, `has_card` | Grouped product indicators used to profile segments |
