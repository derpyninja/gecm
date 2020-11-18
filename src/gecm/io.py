import os
import pickle
import numpy as np
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


def config_describe(config_file):
    config_file_sections = config_file.sections()
    for cf_section in config_file_sections:
        print("\n--- {} ---".format(cf_section))
        for k, v in config_file[cf_section].items():
            print("{}: {}".format(k, v))
    return None


def parse_list(config_string, sep=",", cast=None):
    """Parse list separated by sep."""
    parsed_list = list(
        filter(None, (x.strip() for x in config_string.split(sep)))
    )
    if cast is None:
        return parsed_list
    else:
        return np.array(parsed_list).astype(cast)


def get_google_sheet(spreadsheet_id, range_name, credentials, scopes):
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials, scopes
            )
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    service = build("sheets", "v4", credentials=creds)

    try:
        # Call the Sheets API
        return (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=range_name)
            .execute()
        )
    except HttpError as msg:
        print("Sheet data could not be retrieved. Check if names really match.")
        return None


def gsheet2df(gsheet, header=0, stop=None):
    """
    Converts Google sheet data to a Pandas DataFrame.
    Note: This script assumes that your data contains a header file on the first row!
    Also note that the Google API returns 'none' from empty cells - in order for the code
    below to work, you'll need to make sure your sheet doesn't contain empty cells,
    or update the code to account for such instances.
    """
    data_idx = header + 1
    header = gsheet.get("values", [])[header]  # Assumes first line is header!
    values = gsheet.get("values", [])[data_idx:stop]  # Everything else is data.

    if not values:
        raise ValueError("No data found.")
    else:
        all_data = []
        for col_id, col_name in enumerate(header):
            column_data = []

            for row in values:
                try:
                    s = row[col_id]
                except IndexError as msg:
                    s = np.nan
                column_data.append(s)
            ds = pd.Series(data=column_data, name=col_name)
            all_data.append(ds)
        return pd.concat(all_data, axis=1)


def parse_mgmt_decisions(spreadsheet_id, sheets, credentials_fpath, scopes, unstack_data=True):
    # TODO [mid]: create a separate class for all the mgmt decisions stuff, including gdrive connection
    sheet_dict = {}

    # fetch data via gdrive api
    for i, sheet_name in enumerate(sheets):
        print(sheet_name)

        # 1) fetch data
        data_dict = get_google_sheet(
            credentials=credentials_fpath,
            spreadsheet_id=spreadsheet_id,
            range_name=sheet_name,
            scopes=scopes
        )

        # 2) convert to data frame
        df_raw = gsheet2df(data_dict, header=0, stop=11)
        df_raw = df_raw.set_index("Round")

        # 3) convert to numeric
        df = df_raw.copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce",
                                    downcast="integer")

        # 4) append to dict
        sheet_dict[sheet_name] = df

    # Concatenate and tidy management decisions
    df_all = pd.concat(sheet_dict.values(), keys=sheet_dict.keys())
    df_all.index = df_all.index.set_names(["player", "round"])
    df_all = df_all.reset_index()
    df_all["round"] = pd.to_numeric(df_all["round"], errors="coerce",
                                    downcast="integer")

    # Unstack data: based on
    # https://stackoverflow.com/questions/25386870/pandas-plotting-with-multi-index
    df_final = df_all.set_index(["round", "player"]).sort_index()

    if unstack_data:
        return df_final.unstack(level="player")
    else:
        return df_final
