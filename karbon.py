import pandas as pd
import streamlit as st
import logging
from threading import Lock
import os

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

lock = Lock()

def safe_get_value(row, col):
    return row[col] if col in row and pd.notna(row[col]) else 0


def find_karbon_expenses(df_filtered):
    karbon_expenses_data = []
    columns_to_check = ['date(karbon)', 'site name', 'expense item', 'reason for expense', 'expense type', 'price', 'pax', 'amount', 'mode of payment','bill to','requested by','approved by']
    for index, row in df_filtered.iterrows():
        if any(pd.notna(row[col]) and row[col] != 0 for col in columns_to_check):
            karbon_expenses_data.append({
                'Row': index + 3,
                'Date': row['date(karbon)'],
                'Site Name': row['site name'],
                'Expense Item': row['expense item'],
                'Reason for Expense': row['reason for expense'],
                'Expense Type': row['expense type'],
                'Price': row['price'],
                'Pax': row['pax'],
                'Amount': row['amount'],
                'Mode Of Payment': row['mode of payment'],
                'Bill to': row['bill to'],
                'Requested By': row['requested by'],
                'Approved By': row['approved by']
            })

    return karbon_expenses_data

def analysis_data(analysis_df, months_to_include):

    analysis_df['karbon'] = analysis_df['amount']

    # Group data by month and calculate sums
    metrics = ['karbon']
    grouped_df = analysis_df.groupby('month')[metrics].sum().reset_index()

    # Pivot the grouped data for summary
    summary_df = grouped_df.set_index('month').T
    summary_df.columns.name = None  # Remove the name of columns

    # Reorder columns to match the order of the months to display
    summary_df = summary_df[months_to_include]

    # Calculate percentage change for specific rows only (up to 'net_revenue')
    if len(summary_df.columns) > 1:
        latest_month, previous_month = summary_df.columns[-1], summary_df.columns[-2]

        # Define rows for which the change percentage should be calculated
        rows_to_calculate = ['karbon']

        # Calculate the percentage change and round to 1 decimal place
        change_percentage = ((summary_df[latest_month] - summary_df[previous_month]) / summary_df[previous_month]) * 100
        summary_df['change_percentage'] = change_percentage.round(1)

        # Ensure only specific rows have the change calculated
        summary_df.loc[~summary_df.index.isin(rows_to_calculate), 'change_percentage'] = None
    else:
        # If only one month is present, set 'change_percentage' as 'NA'
        summary_df['change_percentage'] = 'NA'
    
    return summary_df

def calculate_aggregated_values(df_filtered):
    sum_amount = df_filtered['amount'].sum()

    aggregated_data = {
        'Karbon Amount': sum_amount
    }

    return aggregated_data



def format_dataframe(df):
    for column in df.select_dtypes(include=['float', 'int']).columns:
        df[column] = df[column].map(lambda x: f"{x:.1f}")
    return df

def fmt_inr(df):
    for column in df.select_dtypes(include=['float', 'int']).columns:
        df[column] = df[column].map(lambda x: f"{int(x):,}")
    return df

def format_all_columns_with_color(df):
    # Format numerical columns to one decimal place
    for column in df.select_dtypes(include=['float', 'int']).columns:
        if column != 'change_percentage':  # Avoid formatting 'change_percentage' twice
            df[column] = df[column].map(lambda x: f"{x:.1f}")

    # Format 'change_percentage' column to display with a percentage symbol
    if 'change_percentage' in df.columns:
        df['change_percentage'] = df['change_percentage'].map(lambda x: f"{x:.1f}%")

    # Apply conditional formatting to the 'change_percentage' column
    def highlight_change_percentage(val):
        try:
            color = 'red' if float(val.strip('%')) < 0 else 'green'
        except ValueError:
            color = 'black'  # Default color if conversion fails
        return f'color: {color}'

    # Apply the styling to the DataFrame
    styled_df = df.style.applymap(highlight_change_percentage, subset=['change_percentage'])
    
    return styled_df

def display_dataframes(karbon_expenses_data, aggregated_data, summary_df):
    st.subheader("Only Karbon Data of Clients")

    if karbon_expenses_data:
        karbon_expenses_df = pd.DataFrame(karbon_expenses_data)
        st.subheader("Karbon Expenses")
        st.table(format_dataframe(karbon_expenses_df))
        st.markdown("---")
    else:
        st.write("No Karbon expenses found.")
        st.markdown("---")

    aggregated_df = pd.DataFrame(list(aggregated_data.items()), columns=['Parameter', 'Value'])
    st.subheader("Aggregated Values")
    st.table(fmt_inr(aggregated_df))

    on = st.toggle("View Analysis")
    if on:
        st.table(format_all_columns_with_color(summary_df))
    st.markdown("---")

def karbon(df_selected_month, analysis_df, months_to_include):
    
    # Sidebar dropdown for site name
    selected_site = st.sidebar.selectbox("Select Site Name", df_selected_month['site name'].unique())

    # Filter dataframe by selected site
    df_filtered = df_selected_month[df_selected_month['site name'] == selected_site]

    # Perform business logic on selected month and last three months data
    aggregated_data = calculate_aggregated_values(df_filtered)
    karbon_expenses_data = find_karbon_expenses(df_filtered)
    
    # Perform analysis on the last three months data
    summary_df = analysis_data(analysis_df, months_to_include)  # Pass months_to_include here

    # Display all relevant dataframes and results
    display_dataframes(karbon_expenses_data, aggregated_data, summary_df)

#-----------------------------------------------------auto Pnl------------------------------------------------------------------
def load_business_logic(df_filtered):
    try:
        # Clean the data by stripping spaces and converting strings to lowercase
        df_filtered = df_filtered.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

        # Raise an error if the filtered dataframe is empty
        if df_filtered.empty:
            raise ValueError("No data available for the selected month.")

        # Group by 'cost centre' to process each group separately
        grouped = df_filtered.groupby('cost centre')

        pnl_data_list = []  # Store individual P&L data

        # Iterate through each group by cost centre
        for cost_centre, group in grouped:
            # Process each group
            full_data = group

            # Apply the filtering logic within the current group
            regular_pax = group[group['order type'].isin(['regular', 'regular-pop-up', 'food trial'])]
            regular_amt = group[group['order type'].isin(['regular', 'smartq-pop-up', 'food trial', 'regular-pop-up', 'tuckshop', 'live'])]
            event_amt = group[group['order type'].isin(['event', 'event-pop-up', 'adhoc'])]
            no_of_days = group[(group['buying pax'] > 0) | (group['selling pax'] > 0)]

            pnl_data = pd.DataFrame({
                'days': [no_of_days['date'].nunique()],
                'buying pax': [regular_pax['buying pax'].sum()],
                'selling pax': [regular_pax['selling pax'].sum()],
                'regular buying amount': [regular_amt['buying amt ai'].sum()],
                'regular selling amount': [regular_amt['selling amount'].sum()],
                'event buying amount': [event_amt['buying amt ai'].sum()],
                'event selling amount': [event_amt['selling amount'].sum()],
                'penalty on vendor': [full_data['penalty on vendor'].sum()],
                'penalty on smartq': [full_data['penalty on smartq'].sum()],
                'sams': [full_data['amount'].sum()]
            })

            pnl_data_list.append(pnl_data)  # Append the result for each cost centre

        if pnl_data_list:
            final_pnl_data = pd.concat(pnl_data_list, ignore_index=True)
            return format_pnl_dataframe(final_pnl_data)
        else:
            st.warning("No valid data to process.")
            return None

    except Exception as e:
        st.error(f"Error loading Auto P&L logic data: {e}")
        logging.error(f"Error loading Auto P&L logic data: {e}")
        return None

def format_pnl_dataframe(df_filtered):
    for column in df_filtered.select_dtypes(include=['float', 'int']).columns:
        df_filtered[column] = df_filtered[column].map(lambda x: f"{x:.1f}")
    return df_filtered

def load_pnl_excel_data(p_and_l_file_path):
    try:
        pnl_df = pd.read_excel(p_and_l_file_path, header=0)
        pnl_df = pnl_df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
        return pnl_df
    except FileNotFoundError:
        st.error("P&L file not found. Please check the file path.")
        logging.error("P&L file not found.")
        return None

def save_pnl_excel_data(df_filtered, p_and_l_file_path):
    try:
        with pd.ExcelWriter(p_and_l_file_path, mode='w', engine='openpyxl') as writer:
            df_filtered.to_excel(writer, index=False)
        if os.path.exists(p_and_l_file_path):
            os.chmod(p_and_l_file_path, 0o666)
        else:
            st.error(f"File not found: {p_and_l_file_path}")
    except PermissionError:
        st.error("Permission denied: You don't have the necessary permissions to change the permissions of this file.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        logging.error(f"Unexpected error saving Excel data: {e}")

def process_data(pnl_df, pnl_data):
    try:
        # Mapping based on the column names
        pnl_mapping = {
            'days': 'days',
            'buying pax': 'buying pax',
            'selling pax': 'selling pax',
            'regular buying amount': 'regular buying amount',
            'regular selling amount': 'regular selling amount',
            'event buying amount': 'event buying amount',
            'event selling amount': 'event selling amount',
            'penalty on vendor': 'penalty on vendor',
            'penalty on smartq': 'penalty on smartq',
            'sams': 'sams'
        }

        # Check columns exist before mapping
        for source_col, target_col in pnl_mapping.items():
            if source_col in pnl_data.columns and target_col in pnl_df.columns:
                pnl_df[target_col] = pnl_data[source_col]
            else:
                logging.warning(f"Column '{source_col}' in pnl_data or '{target_col}' in pnl_df not found.")

        logging.info("Data processed and mapped successfully.")
        return pnl_df

    except Exception as e:
        st.error(f"Error processing data: {e}")
        logging.error(f"Error processing data: {e}")
        return pnl_df

def punch_pnl_data_to_blank_row(df_filtered, month, p_and_l_file_path): 
    try:
        pnl_df = load_pnl_excel_data(p_and_l_file_path)
        if pnl_df is None:
            return

        if 'month' not in df_filtered.columns or 'month' not in pnl_df.columns:
            st.error("The 'month' column is missing from one of the dataframes.")
            return

        # Create 'identifier' column, prioritize 'review id' completely if it exists, else use 'cost centre'
        if 'review id' in df_filtered.columns and not df_filtered['review id'].isna().all():
            df_filtered['identifier'] = df_filtered['review id']
        else:
            df_filtered['identifier'] = df_filtered['cost centre']
            
        if 'review id' in pnl_df.columns and not pnl_df['review id'].isna().all():
            pnl_df['identifier'] = pnl_df['review id']
        else:
            pnl_df['identifier'] = pnl_df['cost centre']

        # Track if any record was successfully punched
        record_punched = False

        # Iterate through each group by 'identifier' in the filtered data
        for key_value, group in df_filtered.groupby('identifier'):
            st.write(f"Processing identifier: {key_value}")

            pnl_data = load_business_logic(group)
            if pnl_data is None:
                continue

            # Get matching rows by identifier and month
            pnl_df_matching = pnl_df[(pnl_df['identifier'] == key_value) & (pnl_df['month'] == month)]

            if pnl_df_matching.empty:
                st.error(f"No matching records found for identifier: {key_value} and month: {month}")
                continue

            # Ensure the number of rows matches between pnl_data and pnl_df_matching
            if len(pnl_data) == 1 and len(pnl_df_matching) > 1:
                st.warning(f"More than one matching record found for identifier: {key_value} and month: {month}. Aggregating pnl_data.")
                pnl_data = pnl_data.iloc[0]  # Take the first row in pnl_data if there's only one

            # Update specific columns only
            pnl_df.loc[(pnl_df['identifier'] == key_value) & (pnl_df['month'] == month), 
                       ['days', 'buying pax', 'selling pax', 'regular buying amount', 'regular selling amount', 
                        'event buying amount', 'event selling amount', 'penalty on vendor', 'penalty on smartq', 'sams']] = pnl_data[['days', 'buying pax', 'selling pax', 'regular buying amount', 'regular selling amount', 
                        'event buying amount', 'event selling amount', 'penalty on vendor', 'penalty on smartq', 'sams']].values

            # If matching records found, set the flag to True
            record_punched = True

        # Save the P&L Excel data if any record was punched
        if record_punched:
            save_pnl_excel_data(pnl_df, p_and_l_file_path)
            st.success(f"P&L Data punched successfully for month: {month}")
        else:
            st.warning(f"No records were punched for month: {month}")

    except Exception as e:
        st.error(f"Error processing P&L data: {e}")
        logging.error(f"Error processing P&L data: {e}")
def clear_pnl_data(p_and_l_file_path, month, identifier=None):
    """
    Clears the P&L data for the specified month and matching identifier by setting relevant fields to zero.
    """
    try:
        # Load the P&L data
        pnl_df = load_pnl_excel_data(p_and_l_file_path)
        if pnl_df is None:
            return

        # Ensure that the 'month' column exists
        if 'month' not in pnl_df.columns:
            st.error("The 'month' column is missing from the P&L data.")
            return

        # Create the 'identifier' column for matching
        pnl_df['identifier'] = pnl_df['review id'].combine_first(pnl_df['cost centre']) if 'review id' in pnl_df.columns else pnl_df['cost centre']

        # Define the columns to clear
        columns_to_clear = [
            'days', 'buying pax', 'selling pax', 'regular buying amount', 
            'regular selling amount', 'event buying amount', 
            'event selling amount', 'penalty on vendor', 'penalty on smartq', 'sams'
        ]

        # Ensure an identifier is passed
        if identifier is None:
            st.error("No identifier provided for clearing P&L data.")
            return

        # Clear only rows that match both identifier and month
        pnl_df.loc[(pnl_df['month'] == month) & (pnl_df['identifier'] == identifier), columns_to_clear] = 0

        # Save the updated P&L data
        save_pnl_excel_data(pnl_df, p_and_l_file_path)
        st.success(f"P&L data cleared successfully for identifier: {identifier} and month: {month}")
        st.dataframe(pnl_df)

    except Exception as e:
        st.error(f"Error clearing P&L data: {e}")
        logging.error(f"Error clearing P&L data: {e}")

#-------------------------------------------------------Auto Dump--------------------------------------------------

def dump_data(df_filtered, month, dump_file_path):
    dump_mapping = {
        'date': 'date',
        'month': 'month',
        'day': 'day',
        'cost centre': 'cost centre', 
        'site name': 'site name',
        'vendor code': 'vendor code',
        'vendor': 'vendor',
        'session': 'session',
        'meal type': 'meal type',
        'order type': 'order type',
        'client mg/pre order': 'client mg/pre order',
        'ordered pax/vendor mg': 'ordered pax/vendor mg',
        'actual consumption': 'actual consumption',
        'buying pax': 'buying pax',
        'buying price': 'buying price',
        'buying price ai': 'buying price ai',
        'buying transportation': 'buying transportation',
        'buying amt ai': 'buying amt ai',
        'selling pax': 'selling pax',
        'selling price': 'selling price',
        'selling transportation': 'selling transportation',
        'selling amount': 'selling amount',
        'penalty on vendor': 'penalty on vendor',
        'penalty on smartq': 'penalty on smartq',
        'commission': 'commission',
        'amount': 'amount'
    }
    
    try:
        dump_df = load_dump_data(dump_file_path)
        if dump_df is None:
            return

        if not dump_df.empty:
            last_row = dump_df.iloc[-1]
            last_row_df = last_row.to_frame().T
            last_row_df.insert(0, 'row number', len(dump_df) + 1)
            st.write("Last updated row before current dump:")
            st.dataframe(last_row_df)

        mapped_df = pd.DataFrame()
        for dump_col, df_col in dump_mapping.items():
            if df_col in df_filtered.columns and dump_col in dump_df.columns:
                mapped_df[dump_col] = df_filtered[df_col]

        if 'selling management fee' in df_filtered.columns:
            # Group by 'site name' and calculate the sum of 'selling management fee' for each group
            grouped = df_filtered.groupby('site name')

            for site_name, group in grouped:
                selling_sum = group['selling management fee'].sum()
                
                # Create a new row for each group with the aggregated data
                new_row = pd.DataFrame({
                    'month': [month],
                    'site name': [site_name],
                    'order type': ['management fee'],
                    'selling pax': 1,
                    'selling price': [selling_sum],
                    'selling amount': [selling_sum]
                })
                
            # Append the new row to the dump_df DataFrame
            dump_df = pd.concat([dump_df, new_row], ignore_index=True)

        updated_df = pd.concat([dump_df, mapped_df], ignore_index=True)
        save_updated_dump_data(updated_df, dump_file_path)
        logging.info("Filtered data appended to the dump file successfully.")
        st.success("Filtered data appended to the dump file successfully.")

    except Exception as e:
        st.error(f"Error dumping data: {e}")
        logging.error(f"Error dumping data: {e}")

def load_dump_data(dump_file_path):
    try:
        dump_df = pd.read_excel(dump_file_path, header=0)
        dump_df.columns = dump_df.columns.str.lower().str.strip()
        return dump_df
    except FileNotFoundError:
        st.write("Output file not found. Please check the file path.")
        return None

def save_updated_dump_data(df, dump_file_path):
    try:
        df.to_excel(dump_file_path, index=False)
        if os.path.exists(dump_file_path):
            os.chmod(dump_file_path, 0o666)
        else:
            st.write("File not found:", dump_file_path)
    except PermissionError:
        st.write("Permission denied: You don't have the necessary permissions to change the permissions of this file.")
