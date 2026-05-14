 

import pandas as pd
import numpy as np
import os, re
from datetime import datetime
from word2number import w2n

def load_data(file_path):

    df = pd.read_excel(file_path, header=[0,1])

    client_name = os.path.basename(os.path.dirname(file_path))
    venue_name = os.path.splitext(os.path.basename(file_path))[0]

    venue_name = re.sub(r'v\d+(\.\d+)?', '', venue_name, flags=re.IGNORECASE)
    venue_name = re.sub(r'survey', '', venue_name, flags=re.IGNORECASE)
    venue_name = re.sub(r'iGUEST', '', venue_name, flags=re.IGNORECASE)
    venue_name = re.sub(r'[_\-]+', ' ', venue_name)
    venue_name = re.sub(r'\s+', ' ', venue_name).strip()

    df.columns = [' '.join([str(c) for c in col if 'Unnamed' not in str(c)]).strip() for col in df.columns]

    df.insert(0, 'client_name', client_name)
    df.insert(1, 'venue_name', venue_name)

    return df

def fix_datatypes(df):

    for col in ['Email Address', 'First Name', 'Last Name']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df

def drop_footer_last_column(df):

    last_col = df.columns[-1]
    last_col_clean = str(last_col).strip().lower()

    keywords = ["important", "submit", "review", "typos"]

    if any(k in last_col_clean for k in keywords):
        df = df.iloc[:, :-1]

    return df

def map_recommendation_scores(df):
    cols = [
        'How likely would you be to recommend this establishment? Response',
        'Based on this experience, how likely is it that you would return? Response'
    ]

    def map_likert(val):
        if isinstance(val, str):
            val = val.lower().strip()
            if 'very likely' in val:
                return 2
            elif 'quite likely' in val:
                return 1
            elif 'not very likely' in val:
                return 0
        return np.nan

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(map_likert)

    return df

import numpy as np
import pandas as pd

def apply_likert(df):

    metadata_cols = [
        'Respondent ID','Collector ID','Start Date','End Date',
        'IP Address','Email Address','First Name','Last Name'
    ]

    survey_cols = [c for c in df.columns if c not in metadata_cols]

    df[survey_cols] = df[survey_cols].astype(object)

    likert_map = {
        'poor': 0,
        'fair': 1,
        'good': 2,
        'very good': 3}

def process_likert_columns(df):

    likert_cols = []

    for col in survey_cols:

        cleaned_vals = df[col].dropna().apply(clean_text).unique()

        if len(cleaned_vals) > 0 and all(
            val in likert_map or val == 'n/a'
            for val in cleaned_vals
        ):
            likert_cols.append(col)

    for col in likert_cols:

        df[col] = df[col].apply(clean_text)

        df[col] = df[col].apply(
            lambda x: np.nan
            if pd.isna(x) or x == 'n/a'
            else likert_map.get(x, np.nan)
        )

        df[col] = pd.to_numeric(df[col])

    return df

def staff_binary(df):
    staff_cols = [
        c for c in df.columns
        if c.startswith("Thinking about the staff")
        or c.startswith("Please select the words")
    ]

    for col in staff_cols:
        df[col] = df[col].apply(
            lambda x: 1 if pd.notna(x) and str(x).strip() != "" else 0
        )

    return df

def fix_dates(df):
    if 'date_of_your_visit_this_visit' in df.columns:
        df['date_of_your_visit_this_visit'] = pd.to_datetime(
            df['date_of_your_visit_this_visit'],
            dayfirst=True,
            errors='coerce'
        )
    return df

def parse_time(val):
    if pd.isna(val):
        return None

    val = str(val).strip().lower().replace(' ', '')
    val = val.replace(':', '.')
    val = re.sub(r'^(\d{1,2})(am|pm)$', r'\1.00\2', val)

    for fmt in ['%I.%M%p', '%I.%M.%S%p']:
        try:
            return datetime.strptime(val, fmt).strftime('%-I.%M %p').lower()
        except:
            continue
    return None


def fix_time(df):
    for col in df.columns:
        if 'time' in col and 'arrived' in col:
            df[col] = df[col].apply(parse_time)
    return df

def clean_spend(val):
    if pd.isna(val):
        return None
    val = re.sub(r'[^\d.]', '', str(val))
    try:
        return float(val)
    except:
        return None

def fix_spend(df):
    spend_cols = [c for c in df.columns if 'expenditure' in c or 'total' in c]
    for col in spend_cols:
        df[col] = df[col].apply(clean_spend)
    return df

def clean_gender(df):

    col = 'Please enter your details here: Gender:'

    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                'female': 'Female',
                'f': 'Female',
                'male': 'Male',
                'm': 'Male'
            })
        )

        print("Gender fixed:")
        print(df[col].value_counts(dropna=False))

    return df

def parse_guests(val):
    if pd.isna(val):
        return None
    try:
        return w2n.word_to_num(str(val).strip().lower())
    except:
        try:
            return int(val)
        except:
            return None

def fix_guests(df):
    col = 'please_enter_the_number_of_guests_in_your_party_response'
    if col in df.columns:
        df[col] = df[col].apply(parse_guests)
    return df

import numpy as np
import pandas as pd

def map_speed_scores(df):

    drinks_order_map = {
        'within three minutes': 3,
        'between three and four minutes': 2,
        'between four and five minutes': 1,
        'greater than five minutes': 0,
    }

    food_order_map = {
        'within 8 minutes': 3,
        'between 8 and 10 minutes': 2,
        'between 10 and 15 minutes': 1,
        'greater than 15 minutes': 0,
    }

    food_service_map = {
        'within 12 minutes': 3,
        'between 12 and 15 minutes': 2,
        'between 15 and 18 minutes': 1,
        'greater than 18 minutes': 0,
    }

    hot_bev_map = {
        'within five minutes': 3,
        'between five and six minutes': 2,
        'between six and eight minutes': 1,
        'greater than eight minutes': 0,
    }

    speed_col_maps = {
        'Speed of initial drinks order taking How quickly was your drinks order taken from arrival at bar?': drinks_order_map,
        'Speed of initial drinks service How long did it take for your drinks to arrive from time of order?': drinks_order_map,
        'Speed of food ordering How quickly was your food order taken from time of arrival at order point?': food_order_map,
        'Speed of food service How long did it take to receive your food from time of order?': food_service_map,
        'Speed of any additional food service How long did it take to receive any subsequent food from time of order?': food_order_map,
        'Speed of hot beverage service How long did it take to receive your coffee/hot beverage from ordering?': hot_bev_map,
    }

def clean_speed_value(x, mapping):

    if pd.isna(x):
        return np.nan

    x_clean = str(x).strip().lower()

    if x_clean in ["n/a", "na"]:
        return np.nan

    return mapping.get(x_clean, np.nan)


for col, mapping in speed_col_maps.items():

    if col in df.columns:

        df[col] = df[col].apply(
            lambda x: clean_speed_value(x, mapping)
        )

        df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"Mapped: {col}")
        print(f"Values: {df[col].value_counts(dropna=False).to_dict()}")




def process_file(file_path):


    try:
        df = load_data(file_path)
        df = fix_datatypes(df)
        df = drop_footer_last_column(df)
        df = apply_likert(df)
        df = map_recommendation_scores(df)
        df = map_speed_scores(df)
        df = clean_gender(df)
        df = staff_binary(df)
        df = fix_dates(df)
        df = fix_time(df)
        df = fix_spend(df)
        df = fix_guests(df)

        return df

    except Exception as e:
        print(f"Error: {file_path} → {e}")
        return None

def add_client_id(df, client_name, client_id):
    df['client_id'] = client_id
    df['client_name'] = client_name
    return df

def reorder_columns(df):

    priority_cols = ['client_id', 'client_name', 'venue_name']

    cols = priority_cols + [
        c for c in df.columns if c not in priority_cols
    ]

    return df[cols]

base_path = "/content/drive/MyDrive/P000251DA_INVISIBLE GUEST (GROUP 2)/"

for i, client in enumerate(os.listdir(base_path)):

    client_path = os.path.join(base_path, client)
    if not os.path.isdir(client_path):
        continue

    all_dfs = []

    for file in os.listdir(client_path):
        if file.endswith(".xlsx"):

            file_path = os.path.join(client_path, file)

            df = process_file(file_path)

            if df is not None:
                all_dfs.append(df)

    if all_dfs:
        client_id = f"C{str(i+1).zfill(3)}"
        master_df = pd.concat(all_dfs, ignore_index=True, sort=False)
        master_df = add_client_id(master_df, client, client_id)
        master_df = reorder_columns(master_df)

        output_path = os.path.join(base_path, f"{client}_master.xlsx")
        master_df.to_excel(output_path, index=False)

        print(f" Created: {client}_master.xlsx")

import psycopg2

NEON_CONNECTION_STRING = "postgresql://neondb_owner:npg_QrhOFA4P0TqR@ep-small-bonus-annsy0ax-pooler.c-6.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

conn = psycopg2.connect(NEON_CONNECTION_STRING)
print("Connected to Neon")

import re
import pandas as pd

def clean_column_name(col):
    col = str(col).strip().lower()
    col = re.sub(r'[^a-z0-9]+', '_', col)
    col = re.sub(r'_+', '_', col)
    col = col.strip('_')
    if re.match(r'^\d', col):
        col = 'col_' + col
    return col

def make_sql_columns_unique(columns):
    seen = {}
    final_cols = []

    for col in columns:
        base = clean_column_name(col)

        if base not in seen:
            seen[base] = 0
            final_cols.append(base)
        else:
            seen[base] += 1
            final_cols.append(f"{base}_{seen[base]}")

    return final_cols

def prepare_dataframe_for_sql(df):
    df = df.copy()

    # make final cleaned SQL-safe columns unique
    df.columns = make_sql_columns_unique(df.columns)

    # replace NaN with None
    df = df.where(pd.notnull(df), None)

    return df

base_path = "/content/drive/MyDrive/P000251DA_INVISIBLE GUEST (GROUP 2)/"

for client_file in os.listdir(base_path):
    if client_file.endswith("_master.xlsx"):
        print(f"\nChecking: {client_file}")
        file_path = os.path.join(base_path, client_file)
        df = pd.read_excel(file_path)
        sql_df = prepare_dataframe_for_sql(df)

        dupes = pd.Series(sql_df.columns)[pd.Series(sql_df.columns).duplicated()].tolist()

        if dupes:
            print("Still duplicated:", dupes)
        else:
            print("No duplicate columns")
            print(sql_df.columns.tolist()[:30])  # first 30 columns preview

conn.close()

import psycopg2
conn = psycopg2.connect(NEON_CONNECTION_STRING)

base_path = "/content/drive/MyDrive/P000251DA_INVISIBLE GUEST (GROUP 2)/"

for client_file in os.listdir(base_path):
    if client_file.endswith("_master.xlsx"):

        print(f"\nProcessing: {client_file}")

        try:
            file_path = os.path.join(base_path, client_file)
            df = pd.read_excel(file_path)

            sql_df = prepare_dataframe_for_sql(df)
            table_name = clean_column_name(client_file.replace(".xlsx", ""))

            create_table_if_not_exists(conn, table_name, sql_df)
            insert_dataframe(conn, table_name, sql_df)

            print(f"Uploaded: {table_name}")

        except Exception as e:
            print(f"Failed for {client_file}: {e}")
            conn.rollback()

conn.close()
print("All possible files processed")

import re
import pandas as pd

MAX_PG_IDENTIFIER_LEN = 63

def clean_column_name(col):
    col = str(col).strip().lower()
    col = re.sub(r'[^a-z0-9]+', '_', col)
    col = re.sub(r'_+', '_', col)
    col = col.strip('_')
    if re.match(r'^\d', col):
        col = 'col_' + col
    return col

def make_postgres_safe_unique(columns, max_len=63):
    seen = {}
    final_cols = []

    for col in columns:
        base = clean_column_name(col)

        # reserve space for suffix if needed
        candidate = base[:max_len]

        if candidate not in seen:
            seen[candidate] = 0
            final_cols.append(candidate)
        else:
            seen[candidate] += 1
            suffix = f"_{seen[candidate]}"
            trimmed_base = base[:max_len - len(suffix)]
            new_name = f"{trimmed_base}{suffix}"

            # extra safety in case this also somehow exists
            while new_name in seen:
                seen[candidate] += 1
                suffix = f"_{seen[candidate]}"
                trimmed_base = base[:max_len - len(suffix)]
                new_name = f"{trimmed_base}{suffix}"

            seen[new_name] = 0
            final_cols.append(new_name)

    return final_cols

def prepare_dataframe_for_sql(df):
    df = df.copy()
    df.columns = make_postgres_safe_unique(df.columns, max_len=MAX_PG_IDENTIFIER_LEN)
    df = df.where(pd.notnull(df), None)
    return df

base_path = "/content/drive/MyDrive/P000251DA_INVISIBLE GUEST (GROUP 2)/"

for client_file in os.listdir(base_path):
    if client_file.endswith("_master.xlsx"):
        print(f"\nChecking: {client_file}")
        file_path = os.path.join(base_path, client_file)
        df = pd.read_excel(file_path)

        sql_df = prepare_dataframe_for_sql(df)
        cols = list(sql_df.columns)

        dupes = pd.Series(cols)[pd.Series(cols).duplicated()].tolist()

        print("Any duplicates:", len(dupes) > 0)
        print("Longest column length:", max(len(c) for c in cols))
        print([c for c in cols if "conditions_at_the_time_of_visit" in c])

import os
import re
import pandas as pd
import psycopg2
from psycopg2 import sql

def clean_column_name(col):
    col = str(col).strip().lower()
    col = re.sub(r'[^a-z0-9_]+', '_', col)
    col = re.sub(r'_+', '_', col)
    col = col.strip('_')
    return col[:63]

def prepare_dataframe_for_sql(df):
    df = df.copy()

    cleaned_cols = [clean_column_name(c) for c in df.columns]

    seen = {}
    final_cols = []

    for col in cleaned_cols:
        base = col[:55]  # leave space for suffix
        if base in seen:
            seen[base] += 1
            final_cols.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 0
            final_cols.append(base)

    df.columns = final_cols

    df = df.where(pd.notnull(df), None)

    return df

def infer_pg_type(series):
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    elif pd.api.types.is_float_dtype(series):
        return "DOUBLE PRECISION"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    else:
        return "TEXT"

def create_table_if_not_exists(conn, table_name, df):
    columns_sql = []

    for col in df.columns:
        pg_type = infer_pg_type(df[col])
        columns_sql.append(
            sql.SQL("{} {}").format(
                sql.Identifier(col),
                sql.SQL(pg_type)
            )
        )

    query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(columns_sql)
    )

    with conn.cursor() as cur:
        cur.execute(query)

    conn.commit()

def insert_dataframe(conn, table_name, df):
    if df.empty:
        return

    columns = list(df.columns)

    insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(map(sql.Identifier, columns)),
        sql.SQL(", ").join(sql.Placeholder() * len(columns))
    )

    values = [tuple(row) for row in df.to_numpy()]

    with conn.cursor() as cur:
        cur.executemany(insert_query, values)

    conn.commit()

import psycopg2
import os
import pandas as pd

conn = psycopg2.connect(NEON_CONNECTION_STRING)

base_path = "/content/drive/MyDrive/P000251DA_INVISIBLE GUEST (GROUP 2)/"

for client_file in os.listdir(base_path):
    if client_file.endswith("_master.xlsx"):

        print(f"\nProcessing: {client_file}")

        try:
            file_path = os.path.join(base_path, client_file)
            df = pd.read_excel(file_path)

            sql_df = prepare_dataframe_for_sql(df)
            table_name = clean_column_name(client_file.replace(".xlsx", ""))[:63]

            drop_table_if_exists(conn, table_name)
            create_table(conn, table_name, sql_df)
            insert_dataframe(conn, table_name, sql_df)

            print(f"Uploaded fresh table: {table_name}")

        except Exception as e:
            print(f"Failed for {client_file}: {e}")
            conn.rollback()

conn.close()
print("All files processed successfully")

import os
import re
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import sql

NEON_CONNECTION_STRING = "postgresql://neondb_owner:npg_nEe9Ts2yUBqN@ep-small-bonus-annsy0ax-pooler.c-6.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

def clean_column_name(col):
    col = str(col).strip().lower()
    col = re.sub(r'[^a-z0-9]+', '_', col)
    col = re.sub(r'_+', '_', col)
    col = col.strip('_')
    if re.match(r'^\d', col):
        col = 'col_' + col
    return col

def make_postgres_safe_unique(columns, max_len=63):
    seen = {}
    final_cols = []

    for col in columns:
        base = clean_column_name(col)
        candidate = base[:max_len]

        if candidate not in seen:
            seen[candidate] = 0
            final_cols.append(candidate)
        else:
            seen[candidate] += 1
            suffix = f"_{seen[candidate]}"
            trimmed_base = base[:max_len - len(suffix)]
            new_name = f"{trimmed_base}{suffix}"

            while new_name in final_cols:
                seen[candidate] += 1
                suffix = f"_{seen[candidate]}"
                trimmed_base = base[:max_len - len(suffix)]
                new_name = f"{trimmed_base}{suffix}"

            final_cols.append(new_name)

    return final_cols

def prepare_dataframe_for_sql(df):
    df = df.copy()
    df.columns = make_postgres_safe_unique(df.columns)

    # Convert NaN/NaT to None
    df = df.replace({np.nan: None})
    df = df.where(pd.notnull(df), None)

    return df

def infer_pg_type(series):
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    elif pd.api.types.is_float_dtype(series):
        return "DOUBLE PRECISION"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    else:
        return "TEXT"

def drop_table_if_exists(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                sql.Identifier(table_name)
            )
        )
    conn.commit()

def create_table(conn, table_name, df):
    columns_sql = []

    for col in df.columns:
        columns_sql.append(
            sql.SQL("{} {}").format(
                sql.Identifier(col),
                sql.SQL(infer_pg_type(df[col]))
            )
        )

    query = sql.SQL("CREATE TABLE {} ({})").format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(columns_sql)
    )

    with conn.cursor() as cur:
        cur.execute(query)

    conn.commit()

def insert_dataframe(conn, table_name, df):
    if df.empty:
        return

    columns = list(df.columns)

    insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(map(sql.Identifier, columns)),
        sql.SQL(", ").join(sql.Placeholder() * len(columns))
    )

    values = [tuple(row) for row in df.to_numpy()]

    with conn.cursor() as cur:
        cur.executemany(insert_query, values)

    conn.commit()


import psycopg2
NEON_CONNECTION_STRING = "postgresql://neondb_owner:npg_QrhOFA4P0TqR@ep-small-bonus-annsy0ax-pooler.c-6.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

conn = psycopg2.connect(NEON_CONNECTION_STRING)

base_path = "/content/drive/MyDrive/P000251DA_INVISIBLE GUEST (GROUP 2)/"

for client_file in os.listdir(base_path):
    if client_file.endswith("_master.xlsx"):

        print(f"\nProcessing: {client_file}")

        try:
            file_path = os.path.join(base_path, client_file)
            df = pd.read_excel(file_path)

            sql_df = prepare_dataframe_for_sql(df)
            table_name = clean_column_name(client_file.replace(".xlsx", ""))[:63]

            print("Table name:", table_name)
            print("Rows:", len(sql_df))
            print("Columns:", len(sql_df.columns))
            print("Duplicate columns:", sql_df.columns.duplicated().sum())
            print("Max column length:", max(len(c) for c in sql_df.columns))

            drop_table_if_exists(conn, table_name)
            create_table(conn, table_name, sql_df)
            insert_dataframe(conn, table_name, sql_df)

            print(f"Uploaded fresh table: {table_name}")

        except Exception as e:
            print(f"Failed for {client_file}: {e}")
            conn.rollback()

conn.close()
print("All files processed successfully")

import psycopg2

NEON_CONNECTION_STRING = "postgresql://neondb_owner:npg_pMPV6awiDF3U@ep-small-bonus-annsy0ax-pooler.c-6.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

conn = psycopg2.connect(NEON_CONNECTION_STRING)
cur = conn.cursor()

sql_views = r"""

-- =====================================================
-- DROP DEPENDENT POWER BI / REPORTING VIEWS FIRST
-- Do NOT drop base section views here:
-- vw_service_score, vw_selling_score, vw_product_score, vw_environment_score
-- =====================================================

drop view if exists vw_comments_powerbi cascade;
drop view if exists vw_report_base_powerbi cascade;
drop view if exists vw_staff_service_words_powerbi cascade;
drop view if exists vw_staff_behaviour_powerbi cascade;
drop view if exists vw_full_survey_comments_powerbi cascade;
drop view if exists vw_question_scores_powerbi cascade;
drop view if exists vw_section_breakdown_final cascade;
drop view if exists vw_question_scores cascade;
drop view if exists vw_trend_score_final cascade;
drop view if exists vw_previous_visits_score_final cascade;
drop view if exists vw_latest_visit_score_final cascade;
drop view if exists vw_powerbi_master_final cascade;
drop view if exists vw_total_score_final cascade;

-- =====================================================
-- 1. TOTAL SCORE FINAL VIEW WITH SAFE DATE LOGIC
-- Fixes bad dates like 05/24/0018
-- =====================================================

create view vw_total_score_final as
with all_scores as (
    select
        client_name,
        venue_name,
        respondent_id::text as respondent_id,
        assessor_id::text as assessor_id,
        visit_type,
        visit_date_raw,
        service_score,
        null::numeric as selling_score,
        null::numeric as product_score,
        null::numeric as environment_score
    from vw_service_score

    union all

    select
        client_name,
        venue_name,
        respondent_id::text as respondent_id,
        assessor_id::text as assessor_id,
        visit_type,
        visit_date_raw,
        null::numeric as service_score,
        selling_score,
        null::numeric as product_score,
        null::numeric as environment_score
    from vw_selling_score

    union all

    select
        client_name,
        venue_name,
        respondent_id::text as respondent_id,
        assessor_id::text as assessor_id,
        visit_type,
        visit_date_raw,
        null::numeric as service_score,
        null::numeric as selling_score,
        product_score,
        null::numeric as environment_score
    from vw_product_score

    union all

    select
        client_name,
        venue_name,
        respondent_id::text as respondent_id,
        assessor_id::text as assessor_id,
        visit_type,
        visit_date_raw,
        null::numeric as service_score,
        null::numeric as selling_score,
        null::numeric as product_score,
        environment_score
    from vw_environment_score
),
cleaned as (
    select
        concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id) as visit_id,
        client_name,
        trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')) as venue_name,
        venue_name as venue_name_original,
        respondent_id,
        trim(split_part(assessor_id, '.', 1)) as assessor_id,
        visit_type,
        visit_date_raw,

        case
            when visit_date_raw ~ '^\d{2}/\d{2}/\d{4}$'
             and split_part(visit_date_raw, '/', 3)::int between 2000 and 2100
             and split_part(visit_date_raw, '/', 1)::int between 1 and 31
             and split_part(visit_date_raw, '/', 2)::int between 1 and 12
            then to_date(visit_date_raw, 'DD/MM/YYYY')
            else null
        end as visit_date,

        round(max(service_score)::numeric, 1) as service_score,
        round(max(selling_score)::numeric, 1) as selling_score,
        round(max(product_score)::numeric, 1) as product_score,
        round(max(environment_score)::numeric, 1) as environment_score
    from all_scores
    group by client_name, venue_name, respondent_id, assessor_id, visit_type, visit_date_raw
)
select
    *,
    round((
        coalesce(service_score, 0) +
        coalesce(selling_score, 0) +
        coalesce(product_score, 0) +
        coalesce(environment_score, 0)
    ) /
    nullif(
        (case when service_score is not null then 1 else 0 end) +
        (case when selling_score is not null then 1 else 0 end) +
        (case when product_score is not null then 1 else 0 end) +
        (case when environment_score is not null then 1 else 0 end),
        0
    ), 1) as total_score
from cleaned
where visit_date >= date '2023-01-01'
  and venue_name not ilike '%report%'
  and venue_name not ilike '%ishop%'
  and venue_name not ilike '%isales%';

-- =====================================================
-- 2. POWER BI MASTER FINAL VIEW
-- Explicit columns prevent duplicate visit_id
-- =====================================================

create view vw_powerbi_master_final as
with ranked as (
    select
        visit_id,
        client_name,
        venue_name,
        venue_name_original,
        respondent_id,
        assessor_id,
        visit_type,
        visit_date_raw,
        visit_date,
        service_score,
        selling_score,
        product_score,
        environment_score,
        total_score,
        row_number() over (
            partition by client_name, venue_name
            order by visit_date desc nulls last, respondent_id desc
        ) as visit_rank
    from vw_total_score_final
)
select
    visit_id,
    client_name,
    venue_name,
    venue_name_original,
    respondent_id,
    assessor_id,
    visit_type,
    visit_date_raw,
    visit_date,
    service_score,
    selling_score,
    product_score,
    environment_score,
    total_score,
    visit_rank,
    case when visit_rank = 1 then true else false end as is_latest_visit,
    case when visit_rank between 2 and 11 then true else false end as is_previous_10_visits,
    case
        when visit_rank = 1 then 'Latest Visit'
        when visit_rank between 2 and 11 then 'Previous 10 Visits'
        else 'Older Visit'
    end as visit_group,
    case
        when total_score >= 90 then 'Excellent'
        when total_score >= 80 then 'Good'
        when total_score >= 70 then 'Needs Attention'
        else 'Critical'
    end as performance_band
from ranked;

create view vw_latest_visit_score_final as
select *
from vw_powerbi_master_final
where is_latest_visit = true;

create view vw_previous_visits_score_final as
select *
from vw_powerbi_master_final
where is_previous_10_visits = true;

create view vw_trend_score_final as
select *
from vw_powerbi_master_final;

-- =====================================================
-- 3. QUESTION SCORES - SECTION/CATEGORY LEVEL WITH VISIT_ID
-- =====================================================

create view vw_question_scores as

-- SERVICE
select
    concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text) as visit_id,
    client_name,
    trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')) as venue_name,
    respondent_id::text as respondent_id,
    assessor_id::text as assessor_id,
    visit_date_raw,
    'A - Service' as section,
    'Pre-arrival' as question_name,
    round(service_pre_arrival_score::numeric, 2) as question_score
from vw_service_score
where service_pre_arrival_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'A - Service', 'Arrival', round(service_arrival_score::numeric, 2)
from vw_service_score
where service_arrival_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'A - Service', 'Delivery', round(service_delivery_score::numeric, 2)
from vw_service_score
where service_delivery_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'A - Service', 'Timing', round(service_timing_score::numeric, 2)
from vw_service_score
where service_timing_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'A - Service', 'Departure', round(service_departure_score::numeric, 2)
from vw_service_score
where service_departure_score is not null

-- SELLING
union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'B - Selling', 'Written Information', round(selling_written_score::numeric, 2)
from vw_selling_score
where selling_written_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'B - Selling', 'Verbal Information', round(selling_verbal_score::numeric, 2)
from vw_selling_score
where selling_verbal_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'B - Selling', 'Recommendations', round(selling_recommendation_score::numeric, 2)
from vw_selling_score
where selling_recommendation_score is not null

-- PRODUCT
union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'C - Product', 'Food Concept', round(product_food_concept_score::numeric, 2)
from vw_product_score
where product_food_concept_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'C - Product', 'Food Application', round(product_food_application_score::numeric, 2)
from vw_product_score
where product_food_application_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'C - Product', 'Food Output', round(product_food_output_score::numeric, 2)
from vw_product_score
where product_food_output_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'C - Product', 'Cold Drinks', round(product_cold_drinks_score::numeric, 2)
from vw_product_score
where product_cold_drinks_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'C - Product', 'Wine', round(product_wine_score::numeric, 2)
from vw_product_score
where product_wine_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'C - Product', 'Coffee', round(product_coffee_score::numeric, 2)
from vw_product_score
where product_coffee_score is not null

-- ENVIRONMENT
union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'D - Environment', 'Bathrooms', round(env_bathroom_score::numeric, 2)
from vw_environment_score
where env_bathroom_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'D - Environment', 'Smoking Area', round(env_smoking_score::numeric, 2)
from vw_environment_score
where env_smoking_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'D - Environment', 'Furnishings', round(env_furnishings_score::numeric, 2)
from vw_environment_score
where env_furnishings_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'D - Environment', 'Hygiene', round(env_hygiene_score::numeric, 2)
from vw_environment_score
where env_hygiene_score is not null

union all
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id::text, visit_date_raw, 'D - Environment', 'Ambiance', round(env_ambiance_score::numeric, 2)
from vw_environment_score
where env_ambiance_score is not null;

create view vw_question_scores_powerbi as
select
    visit_id,
    client_name,
    venue_name,
    respondent_id,
    assessor_id,
    visit_date_raw,
    section,
    question_name,
    round(question_score::numeric, 2) as question_score,
    case
        when question_score is null then null
        when question_score >= 90 then 'Very Good'
        when question_score >= 75 then 'Good'
        when question_score >= 50 then 'Fair'
        else 'Poor'
    end as rating
from vw_question_scores;

-- =====================================================
-- 4. SECTION BREAKDOWN FINAL
-- =====================================================

create view vw_section_breakdown_final as
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text) as visit_id, client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')) as venue_name, respondent_id::text as respondent_id, 'Service' as section, 'Pre-arrival' as category, service_pre_arrival_score as score from vw_service_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Service', 'Arrival', service_arrival_score from vw_service_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Service', 'Delivery', service_delivery_score from vw_service_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Service', 'Timing', service_timing_score from vw_service_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Service', 'Departure', service_departure_score from vw_service_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Selling', 'Written Information', selling_written_score from vw_selling_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Selling', 'Verbal Information', selling_verbal_score from vw_selling_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Selling', 'Recommendations', selling_recommendation_score from vw_selling_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Product', 'Food Concept', product_food_concept_score from vw_product_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Product', 'Food Application', product_food_application_score from vw_product_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Product', 'Food Output', product_food_output_score from vw_product_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Product', 'Cold Drinks', product_cold_drinks_score from vw_product_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Product', 'Wine', product_wine_score from vw_product_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Product', 'Coffee / Hot Beverages', product_coffee_score from vw_product_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Environment', 'Bathrooms', env_bathroom_score from vw_environment_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Environment', 'Smoking Area', env_smoking_score from vw_environment_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Environment', 'Furnishings / Equipment', env_furnishings_score from vw_environment_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Environment', 'Hygiene / Safety', env_hygiene_score from vw_environment_score
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, 'Environment', 'Ambiance', env_ambiance_score from vw_environment_score;

-- =====================================================
-- 5. STAFF BEHAVIOUR METRICS
-- =====================================================

create view vw_staff_behaviour_powerbi as
select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text) as visit_id, client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')) as venue_name, respondent_id::text as respondent_id, assessor_id, visit_date_raw, 'Staff Behaviour' as section, 'Staff warmth' as behaviour_metric, round(staff_warmth_score::numeric / 3.0 * 100, 2) as score from vw_feedback_service_all where staff_warmth_score::text ~ '^\d+(\.\d+)?$'
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id, visit_date_raw, 'Staff Behaviour', 'Staff presentation', round(staff_presentation_score::numeric / 3.0 * 100, 2) from vw_feedback_service_all where staff_presentation_score::text ~ '^\d+(\.\d+)?$'
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id, visit_date_raw, 'Staff Behaviour', 'Staff grooming', round(staff_grooming_score::numeric / 3.0 * 100, 2) from vw_feedback_service_all where staff_grooming_score::text ~ '^\d+(\.\d+)?$'
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id, visit_date_raw, 'Staff Behaviour', 'Staff attitude', round(staff_attitude_score::numeric / 3.0 * 100, 2) from vw_feedback_service_all where staff_attitude_score::text ~ '^\d+(\.\d+)?$'
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id, visit_date_raw, 'Staff Behaviour', 'Staff teamwork', round(staff_teamwork_score::numeric / 3.0 * 100, 2) from vw_feedback_service_all where staff_teamwork_score::text ~ '^\d+(\.\d+)?$'
union all select concat_ws('|', client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text), client_name, trim(regexp_replace(venue_name, '\s*\(\d+\)$', '')), respondent_id::text, assessor_id, visit_date_raw, 'Staff Behaviour', 'Staff clarity', round(staff_clarity_score::numeric / 3.0 * 100, 2) from vw_feedback_service_all where staff_clarity_score::text ~ '^\d+(\.\d+)?$';

-- =====================================================
-- 6. FULL SURVEY COMMENTS FROM RAW MASTER TABLES
-- =====================================================

do $$
declare
    t text;
    r record;
    sql_text text := '';
    part text;
    tables text[] := array[
        'bodriggy_group_master',
        'club_beenleigh_master',
        'corner_group_master',
        'ganley_group_master',
        'morris_hospitality_master',
        'neighbourhood_group_master',
        'peter_bouchier_master',
        'san_telmo_group_master',
        'sebastian_group_master',
        'sense_of_self_master',
        'trader_house_master'
    ];
begin
    create temp table tmp_comment_map (
        section text,
        comment_type text,
        column_name text
    ) on commit drop;

    insert into tmp_comment_map values
    ('Service', 'Web / Social Media Comments', 'web_and_social_media_sites_comments'),
    ('Service', 'Website Comments', 'web_site_comments'),
    ('Service', 'Signage Comments', 'signage_customer_information_comments'),
    ('Service', 'Exterior Comments', 'exterior_d_cor_lighting_environs_comments'),
    ('Service', 'Welcome Comments', 'welcome_comments'),
    ('Service', 'Welcome Reception Comments', 'welcome_reception_comments'),
    ('Service', 'Staff Description', 'staff_please_describe_your_server_including_name_if_possible_fo'),
    ('Service', 'Staff Description', 'staff_please_describe_your_server_s_including_name_if_possible_'),
    ('Service', 'Service Quality Generic Comments', 'service_quality_generic_comments'),
    ('Service', 'Service Quality Comments', 'service_quality_comments'),
    ('Service', 'Service Quality Food Comments', 'service_quality_food_comments'),
    ('Service', 'Service Quality Beverage Comments', 'service_quality_beverage_comments'),
    ('Service', 'Timing Comments', 'general_timing_comments'),
    ('Service', 'Timing Comments', 'timing_comments'),
    ('Service', 'Speed of Service Comments', 'speed_of_hot_beverage_service_comments_regarding_speed_of_food_'),
    ('Service', 'Payment Comments', 'payment_procedure_comments'),
    ('Service', 'Farewell Comments', 'farewell_comments'),
    ('Selling', 'Written Information Comments', 'information_written_comments'),
    ('Selling', 'Verbal Information Comments', 'information_verbal_comments'),
    ('Selling', 'Recommendation Comments', 'please_rate_how_successful_your_server_was_in_recommending_comm'),
    ('Product', 'Food Concept Comments', 'food_concept_comments_regarding_food_concept'),
    ('Product', 'Food Application Comments', 'food_application_comments_regarding_food_application'),
    ('Product', 'Food Review', 'food_output_please_write_a_review_of_all_the_food_you_sampled_i'),
    ('Product', 'Food Review', 'food_output_please_write_a_review_of_all_the_food_you_sampled_1'),
    ('Product', 'Cold Drinks Comments', 'cold_drinks_cocktails_beers_made_drinks_soft_drinks_details_and'),
    ('Product', 'Cold Drinks Comments', 'cold_drinks_cocktails_made_drinks_soft_drinks_details_and_descr'),
    ('Product', 'Wine Comments', 'wines_details_and_description_of_wines_sampled_and_any_addition'),
    ('Product', 'Wine / Beer Comments', 'wines_and_or_beers_details_and_description_of_wines_and_or_beer'),
    ('Product', 'Coffee / Hot Beverage Comments', 'coffee_hot_beverages_details_and_description_of_coffee_hot_beve'),
    ('Product', 'Product Suggestions', 'if_there_is_anything_you_would_love_to_see_added_to_the_product'),
    ('Environment', 'Bathroom Comments', 'lavatories_bathrooms_comments'),
    ('Environment', 'Lavatories Comments', 'lavatories_comments'),
    ('Environment', 'Smoking Area Comments', 'smoking_area_comments'),
    ('Environment', 'Furnishings Comments', 'furnishings_glass_and_equipment_comments'),
    ('Environment', 'Furnishings Comments', 'furnishings_glass_comments'),
    ('Environment', 'Equipment / Safety Comments', 'equipment_and_safety_comments'),
    ('Environment', 'Hygiene and Safety Comments', 'hygiene_and_safety_comments'),
    ('Environment', 'Ambiance Comments', 'ambiance_comments'),
    ('Environment', 'Car Parking Comments', 'car_parking_access_comments'),
    ('Overall', 'Overall Summary', 'please_provide_an_overall_summary_of_your_visit_in_approx_120_t'),
    ('Overall', 'Overall Summary', 'please_provide_an_overall_summary_of_your_visit_open_ended_resp'),
    ('Overall', 'Expectation Gap', 'in_what_ways_did_it_fail_to_meet_your_expectations_open_ended_r'),
    ('Overall', 'Staff Improvement', 'describe_any_ways_in_which_you_feel_the_staff_could_have_direct'),
    ('Overall', 'Improvement Point 1', 'please_list_up_to_three_bullet_points_summarising_things_you_fe'),
    ('Overall', 'Improvement Point 2', 'please_list_up_to_three_bullet_points_summarising_things_you__1'),
    ('Overall', 'Improvement Point 3', 'please_list_up_to_three_bullet_points_summarising_things_you__2');

    foreach t in array tables loop
        for r in
            select *
            from tmp_comment_map
            where exists (
                select 1
                from information_schema.columns
                where table_schema = 'public'
                  and table_name = t
                  and column_name = tmp_comment_map.column_name
            )
        loop
            part := format(
                $f$
                select
                    concat_ws('|', client_name::text, trim(regexp_replace(venue_name::text, '\s*\(\d+\)$', '')), respondent_id::text) as visit_id,
                    client_name::text as client_name,
                    trim(regexp_replace(venue_name::text, '\s*\(\d+\)$', '')) as venue_name,
                    respondent_id::text as respondent_id,
                    %L as section,
                    %L as comment_type,
                    %I::text as comment_text,
                    %L as source_table
                from %I
                where %I is not null
                  and nullif(trim(%I::text), '') is not null
                  and lower(trim(%I::text)) not in ('nan','na','n/a','null','none')
                $f$,
                r.section,
                r.comment_type,
                r.column_name,
                t,
                t,
                r.column_name,
                r.column_name,
                r.column_name
            );

            if sql_text = '' then
                sql_text := part;
            else
                sql_text := sql_text || ' union all ' || part;
            end if;
        end loop;
    end loop;

    if sql_text = '' then
        execute 'create view vw_full_survey_comments_powerbi as select null::text as visit_id, null::text as client_name, null::text as venue_name, null::text as respondent_id, null::text as section, null::text as comment_type, null::text as comment_text, null::text as source_table where false';
    else
        execute 'create view vw_full_survey_comments_powerbi as ' || sql_text;
    end if;
end $$;

create view vw_comments_powerbi as
select
    visit_id,
    client_name,
    venue_name,
    respondent_id,
    section,
    comment_type,
    comment_text,
    source_table
from vw_full_survey_comments_powerbi;

-- =====================================================
-- 7. STAFF / SERVICE WORD CHECKBOX VIEW FROM RAW TABLES
-- =====================================================

do $$
declare
    t text;
    r record;
    sql_text text := '';
    part text;
    tables text[] := array[
        'bodriggy_group_master',
        'club_beenleigh_master',
        'corner_group_master',
        'ganley_group_master',
        'morris_hospitality_master',
        'neighbourhood_group_master',
        'peter_bouchier_master',
        'san_telmo_group_master',
        'sebastian_group_master',
        'sense_of_self_master',
        'trader_house_master'
    ];
begin
    create temp table tmp_word_map (
        word_group text,
        word_label text,
        column_name text
    ) on commit drop;

    insert into tmp_word_map values
    ('Staff Behaviour', 'Adept', 'thinking_about_the_staff_please_select_the_words_that_best_desc'),
    ('Staff Behaviour', 'Awkward', 'thinking_about_the_staff_please_select_the_words_that_best_de_1'),
    ('Staff Behaviour', 'Confident', 'thinking_about_the_staff_please_select_the_words_that_best_de_2'),
    ('Staff Behaviour', 'Nervous', 'thinking_about_the_staff_please_select_the_words_that_best_de_3'),
    ('Staff Behaviour', 'Comfortable', 'thinking_about_the_staff_please_select_the_words_that_best_de_4'),
    ('Staff Behaviour', 'Resentful', 'thinking_about_the_staff_please_select_the_words_that_best_de_5'),
    ('Staff Behaviour', 'Empathetic', 'thinking_about_the_staff_please_select_the_words_that_best_de_6'),
    ('Staff Behaviour', 'With own agenda', 'thinking_about_the_staff_please_select_the_words_that_best_de_7'),
    ('Staff Behaviour', 'Perky', 'thinking_about_the_staff_please_select_the_words_that_best_de_8'),
    ('Staff Behaviour', 'Bored', 'thinking_about_the_staff_please_select_the_words_that_best_de_9'),
    ('Staff Behaviour', 'Engaging', 'thinking_about_the_staff_please_select_the_words_that_best_d_10'),
    ('Staff Behaviour', 'Aloof', 'thinking_about_the_staff_please_select_the_words_that_best_d_11'),
    ('Service Behaviour', 'Anticipatory', 'please_select_the_words_that_best_describe_the_service_that_you'),
    ('Service Behaviour', 'Reactive', 'please_select_the_words_that_best_describe_the_service_that_y_1'),
    ('Service Behaviour', 'Helpful', 'please_select_the_words_that_best_describe_the_service_that_y_2'),
    ('Service Behaviour', 'Unhelpful', 'please_select_the_words_that_best_describe_the_service_that_y_3'),
    ('Service Behaviour', 'Calm', 'please_select_the_words_that_best_describe_the_service_that_y_4'),
    ('Service Behaviour', 'Pressured', 'please_select_the_words_that_best_describe_the_service_that_y_5'),
    ('Service Behaviour', 'Precise', 'please_select_the_words_that_best_describe_the_service_that_y_6'),
    ('Service Behaviour', 'Clumsy', 'please_select_the_words_that_best_describe_the_service_that_y_7'),
    ('Service Behaviour', 'Attentive', 'please_select_the_words_that_best_describe_the_service_that_y_8'),
    ('Service Behaviour', 'Distracted', 'please_select_the_words_that_best_describe_the_service_that_y_9'),
    ('Service Behaviour', 'Active', 'please_select_the_words_that_best_describe_the_service_that__10'),
    ('Service Behaviour', 'Passive', 'please_select_the_words_that_best_describe_the_service_that__11');

    foreach t in array tables loop
        for r in
            select *
            from tmp_word_map
            where exists (
                select 1
                from information_schema.columns
                where table_schema = 'public'
                  and table_name = t
                  and column_name = tmp_word_map.column_name
            )
        loop
            part := format(
                $f$
                select
                    concat_ws('|', client_name::text, trim(regexp_replace(venue_name::text, '\s*\(\d+\)$', '')), respondent_id::text) as visit_id,
                    client_name::text as client_name,
                    trim(regexp_replace(venue_name::text, '\s*\(\d+\)$', '')) as venue_name,
                    respondent_id::text as respondent_id,
                    %L as word_group,
                    %L as word_label,
                    1 as selected_flag,
                    %L as source_table
                from %I
                where trim(%I::text) in ('1','1.0','true','TRUE','Yes','yes')
                $f$,
                r.word_group,
                r.word_label,
                t,
                t,
                r.column_name
            );

            if sql_text = '' then
                sql_text := part;
            else
                sql_text := sql_text || ' union all ' || part;
            end if;
        end loop;
    end loop;

    if sql_text = '' then
        execute 'create view vw_staff_service_words_powerbi as select null::text as visit_id, null::text as client_name, null::text as venue_name, null::text as respondent_id, null::text as word_group, null::text as word_label, null::int as selected_flag, null::text as source_table where false';
    else
        execute 'create view vw_staff_service_words_powerbi as ' || sql_text;
    end if;
end $$;

-- =====================================================
-- 8. REPORT BASE POWER BI VIEW
-- Do not recreate visit_id using *, because master already has visit_id
-- =====================================================

create view vw_report_base_powerbi as
select
    visit_id,
    client_name,
    venue_name,
    venue_name_original,
    respondent_id,
    assessor_id,
    visit_type,
    visit_date_raw,
    visit_date,
    service_score,
    selling_score,
    product_score,
    environment_score,
    total_score,
    visit_rank,
    is_latest_visit,
    is_previous_10_visits,
    visit_group,
    performance_band
from vw_powerbi_master_final;

"""

cur.execute(sql_views)
conn.commit()

cur.close()
conn.close()

print("Views created successfully")
