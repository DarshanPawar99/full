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

def check_mismatch(row, index, column_name, expected_value, mismatched_data):
    actual_value = safe_get_value(row, column_name)
    if actual_value != expected_value:
        mismatched_data.append({
            'Row': index + 3,
            'Date': row['date'],
            'Column': column_name,
            'Expected': expected_value,
            'Actual': actual_value
        })


def pivot_and_average_prices(df_filtered):
    # Replace blank values with 'N/A'
    df_filtered = df_filtered.fillna('N/A')
    
    # Create the pivot table
    combined_df = df_filtered.pivot_table(
        index=['site name', 'vendor','meal type (only lunch)','buying price ai', 'selling price'],
        aggfunc='size'
    ).reset_index(name='days')

    total_pax = df_filtered.groupby(['site name', 'vendor','meal type (only lunch)','buying price ai', 'selling price']).agg(
        total_buying_pax=('total pax buying', 'sum'),
        total_selling_pax=('total pax selling', 'sum')
    ).reset_index()
    
    pivot_df = combined_df.merge(total_pax, on=['site name', 'vendor','meal type (only lunch)','buying price ai', 'selling price'])
    return pivot_df


def find_mismatches(df_filtered):
    mismatched_data = []
    for index, row in df_filtered.iterrows():
        try:
            # Calculate Buying Price AI
            meal_type = safe_get_value(row, 'meal type (only lunch)')
            buying_mg_pax = safe_get_value(row, 'buying mg/pax')

            if meal_type == "veg":
                if buying_mg_pax <= 500:
                    buying_price_ai = 42.5
                elif buying_mg_pax <= 900:
                    buying_price_ai = 42.5
                else:
                    buying_price_ai = 42.5
            elif meal_type == "non-veg":
                if buying_mg_pax <= 500:
                    buying_price_ai = 52.5
                elif buying_mg_pax <= 900:
                    buying_price_ai = 52.5
                else:
                    buying_price_ai = 52.5
            else:
                buying_price_ai = ""

            check_mismatch(row, index, 'buying price ai', buying_price_ai, mismatched_data)

            # for delta pax
            calculated_delta_pax = max(
                safe_get_value(row, 'buying mg/pax') - (
                    safe_get_value(row, 'actual consumption/employee') +
                    safe_get_value(row, 'partners(direct cash sales)') +
                    safe_get_value(row, 'manual entry')), 
                safe_get_value(row, 'food coupon'), 
                0
            )
            check_mismatch(row, index, 'delta pax(gap between mg and consumption)', calculated_delta_pax, mismatched_data)

            # for total pax buying
            total_pax_buying = max((
                safe_get_value(row, 'actual consumption/employee') +
                safe_get_value(row, 'partners(direct cash sales)') +
                safe_get_value(row, 'manual entry') +
                safe_get_value(row, 'delta pax(gap between mg and consumption)')
            ),safe_get_value(row, 'buying mg/pax'))
            check_mismatch(row, index, 'total pax buying', total_pax_buying, mismatched_data)

            # for buying amount
            buying_amount = safe_get_value(row, 'total pax buying') * safe_get_value(row, 'buying price ai') * 2
            check_mismatch(row, index, 'buying amount', buying_amount, mismatched_data)

            # for selling price
            meal_type = safe_get_value(row, 'meal type (only lunch)')
            selling_mg_pax = safe_get_value(row, 'selling mg/pax')

            if meal_type == "veg":
                if selling_mg_pax <= 500:
                    selling_price = 55
                elif selling_mg_pax <= 900:
                    selling_price = 55
                else:
                    selling_price = 55
            elif meal_type == "non-veg":
                if selling_mg_pax <= 500:
                    selling_price = 60
                elif selling_mg_pax <= 900:
                    selling_price = 60
                else:
                    selling_price = 60
            else:
                selling_price = ""

            check_mismatch(row, index, 'selling price', selling_price, mismatched_data)

            # for delta pax(gap between mg and consumption) BTC
            delta_pax_gap_btc = max(
                safe_get_value(row, 'selling mg/pax') - (
                    safe_get_value(row, 'actual consumption/employee') +
                    safe_get_value(row, 'manual entry')
                ),
                safe_get_value(row, 'food coupon btc'),
                0
            )

            check_mismatch(row, index, 'delta pax(gap between mg and consumption) btc', delta_pax_gap_btc, mismatched_data)

            # for total pax selling
            actual_consumption_employee = safe_get_value(row, 'actual consumption/employee')
            partners_direct_cash_sales = safe_get_value(row, 'partners(direct cash sales)')
            manual_entry = safe_get_value(row, 'manual entry')
            food_coupon_btc = safe_get_value(row, 'food coupon btc')
            selling_mg_pax = safe_get_value(row, 'selling mg/pax')

            if (actual_consumption_employee + partners_direct_cash_sales + manual_entry + food_coupon_btc) < selling_mg_pax:
                total_pax_selling = selling_mg_pax
            else:
                total_pax_selling = (actual_consumption_employee + partners_direct_cash_sales + manual_entry + food_coupon_btc)

            check_mismatch(row, index, 'total pax selling', total_pax_selling, mismatched_data)

            # for partners(direct cash sales) + employee 50%
            partners_direct_cash_sales = safe_get_value(row, 'partners(direct cash sales)')
            selling_price = safe_get_value(row, 'selling price')
            actual_consumption_employee = safe_get_value(row, 'actual consumption/employee')
            manual_entry = safe_get_value(row, 'manual entry')

            partners_employee_50_percent = (
                (partners_direct_cash_sales * selling_price * 2) + 
                ((actual_consumption_employee + manual_entry) * selling_price)
            )

            check_mismatch(row, index, 'partners(direct cash sales) +employee 50%', partners_employee_50_percent, mismatched_data)

            # for total sales
            actual_consumption_employee = safe_get_value(row, 'actual consumption/employee')
            manual_entry = safe_get_value(row, 'manual entry')
            selling_price = safe_get_value(row, 'selling price')
            total_pax_selling = safe_get_value(row, 'delta pax(gap between mg and consumption) btc')
            partners_employee_50_percent = safe_get_value(row, 'partners(direct cash sales) +employee 50%')

            total_sales = (
                ((actual_consumption_employee + manual_entry) * selling_price) +
                ((total_pax_selling * selling_price) * 2) +
                partners_employee_50_percent
            )

            check_mismatch(row, index, 'total sales', total_sales, mismatched_data)

            # for btc
            total_sales = safe_get_value(row, 'total sales')
            partners_employee_50_percent = safe_get_value(row, 'partners(direct cash sales) +employee 50%')

            btc = total_sales - partners_employee_50_percent

            check_mismatch(row, index, 'btc', btc, mismatched_data)


            # for comission
            total_sales = safe_get_value(row, 'total sales')
            buying_amount = safe_get_value(row, 'buying amount')

            comission = total_sales - buying_amount

            check_mismatch(row, index, 'comission', comission, mismatched_data)

            
        except Exception as e:
            logging.error(f"Error processing row {index + 3}: {e}")

    return mismatched_data


def analysis_data(analysis_df, months_to_include):

    # Fill missing values with 0 for specified columns
    columns_to_fill = ['total pax selling', 'total pax buying', 'btc', 'comission', 'buying price ai']
    analysis_df[columns_to_fill] = analysis_df[columns_to_fill].fillna(0)

    # Calculate additional columns
    analysis_df['paxs_difference'] = analysis_df['total pax selling'] - analysis_df['total pax buying']
    analysis_df['pd_revenue'] = analysis_df['paxs_difference'] * analysis_df['buying price ai']
    analysis_df['total_gmv'] = analysis_df['btc']
    analysis_df['total_pax'] = analysis_df['total pax selling']

    
    # Group data by month and calculate sums for relevant metrics
    metrics = ['comission', 'paxs_difference', 'pd_revenue', 'total_gmv','total_pax']
    grouped_df = analysis_df.groupby('month', as_index=False)[metrics].sum()

    

    # Calculate 'pd_revenue_percentage' and 'event_gmv_percentage'
    grouped_df['pd_revenue_percentage'] = (grouped_df['pd_revenue'] / grouped_df['comission']).replace([float('inf'), -float('inf')], 0).fillna(0) * 100
    grouped_df['pd_pax_percentage'] = (grouped_df['paxs_difference'] / grouped_df['total_pax']).replace([float('inf'), -float('inf')], 0).fillna(0) * 100
    

    grouped_df = grouped_df.drop(columns=['total_pax'])


    # Pivot the grouped data for summary
    summary_df = grouped_df.set_index('month').T
    summary_df.columns.name = None  # Remove the name of columns

    # Reorder columns based on months to include
    summary_df = summary_df[months_to_include]

    # Calculate percentage change for specific rows only (up to 'net_revenue')
    if len(summary_df.columns) > 1:
        latest_month, previous_month = summary_df.columns[-1], summary_df.columns[-2]

        # Define rows for which the change percentage should be calculated
        rows_to_calculate = ['comission', 'paxs_difference', 'pd_revenue', 'total_gmv']

        # Calculate the percentage change and handle cases where previous_month is 0 to avoid division by zero
        change_percentage = ((summary_df[latest_month] - summary_df[previous_month]) / summary_df[previous_month]) * 100

        # Replace infinite values (when previous_month is 0) with 100%, and NaN values with 0
        change_percentage = change_percentage.replace([float('inf'), -float('inf')], 100).fillna(0)

        # Round to 1 decimal place
        summary_df['change_percentage'] = change_percentage.round(1)

        # Ensure only specific rows have the change calculated
        summary_df.loc[~summary_df.index.isin(rows_to_calculate), 'change_percentage'] = None
    else:
        # If only one month is present, set 'change_percentage' as 'NA'
        summary_df['change_percentage'] = 'NA'

    return summary_df



def pd_session_wise(df_filtered):
    
    normal_paxs = df_filtered.copy()
    # Ensure the required columns exist
    if 'meal type (only lunch)' in normal_paxs.columns and 'total pax selling' in normal_paxs.columns and 'total pax buying' in normal_paxs.columns:
        
        # Calculate pax difference
        normal_paxs['pax difference'] = normal_paxs['total pax selling'] - normal_paxs['total pax buying']
        normal_paxs['pax difference revenue'] = normal_paxs['pax difference'] * normal_paxs['buying price ai']*2
        
        # Handle division by zero in percentage calculation
        normal_paxs['pax difference percentage'] = normal_paxs.apply(
            lambda row: (row['pax difference'] / row['total pax selling'] * 100) if row['total pax selling'] != 0 else 0, axis=1
        )
        
        # Group by session and calculate total pax difference for each session
        session_wise_df = normal_paxs.groupby('meal type (only lunch)').agg(
            total_selling_pax=('total pax selling', 'sum'),
            total_buying_pax=('total pax buying', 'sum'),
            total_pax_difference=('pax difference', 'sum'),
            total_pd_revenue=('pax difference revenue', 'sum'),
            total_pd_percentage=('pax difference percentage', 'mean')  # Adjusted to use 'mean' for percentage
        ).reset_index()

        # Format the 'total_pd_percentage' column with the percent symbol
        session_wise_df['total_pd_percentage'] = session_wise_df['total_pd_percentage'].apply(lambda x: f"{x:.1f}%")

        # Display the session-wise data
        return session_wise_df
    else:
        raise ValueError("Required columns (session, selling pax, buying pax) are missing in the DataFrame.")



def calculate_aggregated_values(df_filtered):
    sum_buying_pax_regular = df_filtered['total pax buying'].sum()
    sum_selling_pax_regular = df_filtered['total pax selling'].sum()
    
    sum_buying_amt_ai_regular= df_filtered['buying amount'].sum()
    sum_selling_amt_regular = df_filtered['btc'].sum()
    sum_cash_recived = df_filtered['partners(direct cash sales) +employee 50%'].sum()
    sum_commission = df_filtered['comission'].sum()
    

    valid_dates_df = df_filtered[(df_filtered['total pax buying'] > 0) | (df_filtered['total pax selling'] > 0)]
    number_of_days = valid_dates_df['date'].nunique()

    aggregated_data = {
        'Number of Days': number_of_days,
        'Buying Pax (Regular)': sum_buying_pax_regular,
        'Selling Pax (Regular)': sum_selling_pax_regular,
        'Buying Amt AI (Regular)': sum_buying_amt_ai_regular,
        'Selling Amt (Regular)': sum_selling_amt_regular,
        'Cash Recived from Employee': sum_cash_recived,
        'Commission': sum_commission,
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


def display_dataframes(pivot_df, mismatched_data, aggregated_data, summary_df, session_wise_df):
    st.write("This site as Food Coupon")
    st.write("It is a subsidiary model")
    st.markdown("---")
    st.subheader("Average Buying Price and Selling Price")
    st.dataframe(format_dataframe(pivot_df), use_container_width=True)
    st.markdown("---")

    if mismatched_data:
        mismatched_df = pd.DataFrame(mismatched_data)
        st.write("<span style='color:red'>Mismatched Data:heavy_exclamation_mark:</span>", unsafe_allow_html=True)
        st.table(format_dataframe(mismatched_df))
    else:
        st.write("<span style='color:green'>No mismatch found.</span> :white_check_mark:", unsafe_allow_html=True)
    st.markdown("---")

    aggregated_df = pd.DataFrame(list(aggregated_data.items()), columns=['Parameter', 'Value'])
    st.subheader("Aggregated Values")
    st.table(fmt_inr(aggregated_df))

    on = st.toggle("View Analysis")
    if on:
        st.write("Session wise PD")
        st.table(fmt_inr(session_wise_df))
        st.write("3-Months Comparison")
        st.table(format_all_columns_with_color(summary_df))
        
    st.markdown("---")


def business_logic_46(df_selected_month, analysis_df, months_to_include):
    # Perform business logic on selected month and last three months data
    pivot_df = pivot_and_average_prices(df_selected_month)
    mismatched_data = find_mismatches(df_selected_month)
    aggregated_data = calculate_aggregated_values(df_selected_month)
    session_wise_df = pd_session_wise(df_selected_month)
    
    # Perform analysis on the last three months data
    summary_df = analysis_data(analysis_df, months_to_include)  # Pass months_to_include here

    # Display all relevant dataframes and results
    display_dataframes(pivot_df, mismatched_data, aggregated_data, summary_df, session_wise_df)

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
            no_of_days = group[(group['total pax buying'] > 0) | (group['total pax selling'] > 0)]

            pnl_data = pd.DataFrame({
                'days': [no_of_days['date'].nunique()],
                'buying pax': [full_data['total pax buying'].sum()],
                'selling pax': [full_data['total pax selling'].sum()],
                'regular buying amount': [full_data['buying amount'].sum()],
                'regular selling amount': [full_data['btc'].sum()],
                'cash recevied' : [full_data['partners(direct cash sales) +employee 50%'].sum()],
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
            'cash recevied': 'cash recevied',
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
                       ['days', 'buying pax', 'selling pax', 'regular buying amount', 'regular selling amount', 'cash recevied']] = pnl_data[['days', 'buying pax', 'selling pax', 'regular buying amount', 'regular selling amount', 
                        'cash recevied']].values

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
def clear_pnl_data(p_and_l_file_path, month):
    """
    Clears the P&L data for the specified month by setting relevant fields to zero.
    """
    try:
        pnl_df = load_pnl_excel_data(p_and_l_file_path)
        if pnl_df is None:
            return

        # Ensure that the 'month' column exists
        if 'month' not in pnl_df.columns:
            st.error("The 'month' column is missing from the P&L data.")
            return

        # Create the 'identifier' column for matching
        pnl_df['identifier'] = pnl_df['review id'].combine_first(pnl_df['cost centre']) if 'review id' in pnl_df.columns else pnl_df['cost centre']

        # Filter the data for the selected month
        pnl_df_matching = pnl_df[pnl_df['month'] == month]

        if pnl_df_matching.empty:
            st.warning(f"No data found for the month: {month}")
            return

        # Define the columns to clear
        columns_to_clear = [
            'days', 'buying pax', 'selling pax', 'regular buying amount', 
            'regular selling amount','cash recevied'
        ]

        # Clear the relevant columns for the selected month
        pnl_df.loc[pnl_df['month'] == month, columns_to_clear] = 0

        save_pnl_excel_data(pnl_df, p_and_l_file_path)
        st.success(f"P&L data cleared successfully for the month: {month}")
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
        'cost centre' : 'cost centre', 
        'site name': 'site name',
        'vendor code': 'vendor code',
        'vendor': 'vendor',
        'meal type': 'meal type (only lunch)',
        'buying pax': 'total pax buying',
        'buying amt ai': 'buying amount',
        'selling pax': 'total pax selling',
        'selling amount': 'btc',
        'commission': 'comission',
        'cash recived' : 'partners(direct cash sales) +employee 50%'
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
