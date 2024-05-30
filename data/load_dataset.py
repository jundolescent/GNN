### GNN
import pandas as pd


file_path = './data/orchestration_result.xlsx'
export_file_path = '../result.csv'


def fill_missing_links(df):
    empty_links = ['link' + str(i) for i in range(8, 29)]
    for link in empty_links:
        if link not in df.columns:
            df[link] = 0  # Add the missing link columns and fill with 0
    for index, row in df.iterrows():
        num_servers = int(row['server'])  # Count non-null values from the 8th column onwards
        max_edges = int(num_servers * (num_servers - 1) // 2 ) # Maximum possible number of edges
        # Reset all links to 0


        # 여기 로직 아예 새로 고쳐야 함
        # Fill missing links
        for i in range(num_servers, max_edges + 1):
            if row[f'link{i}'] == 0:  # Check if the link in this row is empty
                # Find adjacent links
                adjacent_links = []
                if i > num_servers:  # Not the first link
                    adjacent_links.append(row[f'link{i - 1}'])
                if i < max_edges:  # Not the last link
                    adjacent_links.append(row[f'link{i + 1}'])

                # Calculate the average of adjacent links
                avg_delay = sum(adjacent_links) / len(adjacent_links)

                # Assign the average delay to the missing link for this row
                df.at[index, f'link{i}'] = avg_delay

        # Set all links beyond max_edges to 0
        for i in range(max_edges + 1, 29):
            df.at[index, f'link{i}'] = 0

    return df


def split_combination(combination):
    comb_list = list(map(int, str(combination)))
    comb_list += [0] * (8 - len(comb_list))
    return comb_list


def split_delay(delay):
    delay_split = delay.split('.')
    delay_split_padded = delay_split + ['0'] * (7 - len(delay_split))
    return [int(value) for value in delay_split_padded[:7]]


def load_data(export=False):
    # mininet_df = pd.read_excel(file_path, sheet_name='mininet')
    # delay_df = pd.read_excel(file_path, sheet_name='output_delay_delay')
    delay_cb_df = pd.read_excel(file_path, sheet_name='output_delay_delay_cb')

    # check if it's correctly downloaded
    # print(mininet_df.head())
    # print(output_delay_delay_df.head())
    # print(output_delay_delay_cb_df.head())

    delay_cb_df[['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7']] = delay_cb_df['delay'].apply(
        lambda x: pd.Series(split_delay(x)))

    delay_cb_df[['server1', 'server2', 'server3', 'server4', 'server5', 'server6', 'server7', 'server8']] = delay_cb_df[
        'combination'].apply(lambda x: pd.Series(split_combination(x)))

    delay_cb_df = delay_cb_df.drop(columns=['delay'])
    delay_cb_df = delay_cb_df.drop(columns=['chaincode'])

    delay_cb_df = remove_iqr(delay_cb_df, 'TPS')

    # delay_cb_df = fill_missing_links(delay_cb_df)

    pd.set_option('display.max_columns', None)
    # print(delay_cb_df.head())
    pd.reset_option('display.max_columns')
    # print('data type: ', delay_cb_df.dtypes)
    if export:
        delay_cb_df.to_csv(export_file_path, index=False)
    return delay_cb_df


def remove_iqr(df, column):
    q1 = df[column].quantile(0.45)
    q3 = df[column].quantile(0.55)
    iqr = q3 - q1
    lower_bound = q1 - 1.0 * iqr
    upper_bound = q3 + 1.0 * iqr
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return df[~outliers]
