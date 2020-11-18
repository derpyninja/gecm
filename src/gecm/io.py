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
