import pandas as pd
import logging
import streamlit as st
import importlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Paths to the predefined files
P_AND_L_FILE_PATH = r"C:\Users\Darshan.Pawar\OneDrive - CPGPLC\Auto P&L\P&L.xlsx"
DUMP_FILE_PATH = r"C:\Users\Darshan.Pawar\OneDrive - CPGPLC\Auto P&L\Dump.xlsx"

def setup_page():
    st.set_page_config(page_title="Monthly MIS Checker", layout="wide")
    st.title("MIS Reviewer :chart_with_upwards_trend:")

def upload_file():
    return st.sidebar.file_uploader('Upload Excel file', type=['xlsx', 'xls'])

def read_excel_file(uploaded_file):
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        logging.info("Excel file uploaded successfully.")
        return excel_file
    except ValueError as e:
        st.error(f"Error reading the Excel file: {e}")
        logging.error(f"ValueError reading the Excel file: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        logging.error(f"Unexpected error reading the Excel file: {e}")
    return None

def select_sheet(excel_file):
    sheet_names = excel_file.sheet_names
    return st.sidebar.selectbox('Select a sheet to display', sheet_names)

def read_sheet_to_dataframe(uploaded_file, selected_sheet):
    try:
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, header=1)
        logging.info(f"Sheet '{selected_sheet}' loaded successfully.")
        return df
    except ValueError as e:
        st.error(f"ValueError reading the sheet '{selected_sheet}': {e}")
        logging.error(f"ValueError reading the sheet '{selected_sheet}': {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        logging.error(f"Unexpected error reading the sheet '{selected_sheet}': {e}")
    return None

def preprocess_dataframe(df):
    try:
        df.columns = df.columns.str.lower().str.strip()
        columns_to_convert = df.columns.difference(['date'])
        df[columns_to_convert] = df[columns_to_convert].apply(lambda col: col.str.lower().str.strip() if col.dtype == 'object' else col)
        logging.info("Columns converted to lower case successfully.")
    except Exception as e:
        st.error(f"Error processing the data: {e}")
        logging.error(f"Error processing the data: {e}")
    return df

def filter_dataframe_by_month(df):
    try:
        if 'month' in df.columns:
            available_months = df['month'].unique()
            if 'selected_month' not in st.session_state or st.session_state.selected_month not in available_months:
                st.session_state.selected_month = available_months[0]  # Default to the first month if not set or invalid
            month = st.sidebar.selectbox("Select the month for review", available_months, index=available_months.tolist().index(st.session_state.selected_month))
            st.session_state.selected_month = month
            df_filtered = df[df['month'] == month]
            logging.info(f"Data filtered by month '{month}' successfully.")
            return df_filtered, month
        else:
            st.error("The 'month' column is not present in the dataframe.")
            logging.error("The 'month' column is not present in the dataframe.")
            return None, None
    except KeyError as e:
        st.error(f"KeyError filtering data by month: {e}")
        logging.error(f"KeyError filtering data by month: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        logging.error(f"Unexpected error filtering data by month: {e}")
    return None, None

def analyze_last_three_months(df, selected_month):
    # Analyze the last three months from the selected month
    if 'month' in df.columns:
        # Convert 'month' column to a categorical type to maintain the order dynamically
        months_order = pd.Categorical(df['month'], 
                                      categories=['january', 'february', 'march', 'april', 'may', 'june', 
                                                  'july', 'august', 'september', 'october', 'november', 'december'], 
                                      ordered=True)
        
        # Add 'N/A' category if needed
        if 'N/A' not in months_order.categories:
            months_order = months_order.add_categories(['N/A'])

        df['month'] = months_order
        
        # Find the index of the selected month
        selected_month_index = months_order.categories.get_loc(selected_month)
        
        # Calculate the last three months from the selected month
        months_to_include = [months_order.categories[(selected_month_index - i) % 12] for i in range(2, -1, -1)]
        
        # Filter data for the last 3 months
        analysis_df = df[df['month'].isin(months_to_include)]
        
        return analysis_df, months_to_include
    else:
        logging.error("The DataFrame does not contain a 'month' column.")
        return pd.DataFrame(), []

def apply_business_logic(df_filtered, analysis_df, months_to_include, selected_sheet, month):
    # Define the business logic mapping
    business_logic_sheets = {
        # Your business logic mapping definitions
        # Map sheet names to business logic module names

            "business_logic_1": ["Gojek_NCR"], #gojek is ncr
            "business_logic_2": ["Odessa","Scaler-Prequin","Quzizz","Ather Mumbai","Groww Mumbai.","MPL-Delhi",
                                 "Tadano","Epam", "O9 Solutions","Telstra Event sheet","Dynasty","pure_mg","Quest Hyderabad"],
            "business_logic_3": ["Groww"],
            "business_logic_4": ["Medtrix","MG Eli Lilly", "Awfis","Amadeus","Ather - Main Meal","Ajuba",
                                 "Drivers Tea HYD","Drivers Tea Chennai","Drivers Tea Blore","Vector_BLR","12D Weekend Packed Meal","LTIMindTree-event",
                                 "Clario", "Citrix Driver's Lunch & Dinner","Citrix-Tuckshop","MIQ","Vector","Lam Research","HD Works","Synergy",
                                 "DTCC Company Paid","Gartner","Plan View","Siemens RGA","Solera","Master Card", "Moengage","Groww Jaipur","L & T  S1 Hebbal & Rga ","L & T Executive Lunch"],
           
            "business_logic_5": ["Microchip Main Meal","pure_highest_consumption","Trinity "],
            "business_logic_6": ["Plain View"],
            "business_logic_7": ["MPL"],
            "business_logic_8": ["STRIPE MRP"],
            "business_logic_9": ["Rippling","Tessolve","Corning","Pratilipi","SAEL Delhi","Cloudera "],
            "business_logic_10": ["MPL - Infinity Plates",],
            "business_logic_11": ["Telstra MainMeal(Cash & Carry)"],
            "business_logic_12": ["Eli Lilly Wallet."], # same like semiemens tech
            "business_logic_13": ["Schneider Sodexo Card."],
            "business_logic_14": ["STRIPE MIS","TEA-Breakfast"],
            "business_logic_15": ["Waters"], # used BL6 and might be same for seminens
            "business_logic_16": ["COFFEE VENDING MACHINE"],
            "business_logic_17": ["Tekion_Chn",],
            "business_logic_18": ["H&M"],
            "business_logic_19": ["PhonePe"],
            "business_logic_20": ["Micochip Juice Junction"],
            "business_logic_21": ["Ather BLR"],
            "business_logic_22": ["Ather Plant 1.","Ather Plant 2."],  
            "business_logic_23": ["Continental","actual_selling"],# need to update logic
            "business_logic_24": ["FRUIT N JUICE MIS"],
            "business_logic_25": ["Siemens"],
            "business_logic_26": ["DTCC Wallet"],
            "business_logic_27": ["Siemens_Pune"],
            "business_logic_28": ["CSG-Pune"],
            "business_logic_29": ["Salesforce"],
            "business_logic_30": ["Toasttab"],
            "business_logic_31": [""],
            "business_logic_32": ["Siemens_NCR"], # NCR
            "business_logic_33": [""],
            "business_logic_34": ["Sinch"],
            "business_logic_35": [""],
            "business_logic_36": ["Stryker"],
            "business_logic_37": [""],
            "business_logic_38": ["Truecaller"],
            "business_logic_39": [""],
            "business_logic_40": ["Gold Hill-Main Meal","Goldhill Juice Junction.","Healthineer International","Priteck - Main meal","Pritech park Juice junction"],
            "business_logic_41": ["Siemens-BLR","Siemens Juice Counter"],
            "business_logic_42": ["Healthineer Factory"],
            "business_logic_43": ["Airtel Center","Airtel  Plot 5","Airtel NOC Non veg","Airtel international"],
            "business_logic_44": [""],
            "business_logic_45": [""],
            "business_logic_46": ["Airtel Noida"],
            "business_logic_47": ["Airtel NOC"],
            "business_logic_48": ["Airtel-Jaya"],
            "business_logic_49": [""],
            "business_logic_50": ["MIQ MRP"],
            "business_logic_51": ["Telstra New"],
            "business_logic_52": ["FOODIE FRIDAYACTIVITY"],
            "business_logic_53": ["Accenture MDC2B","BDC7A Transport Tea","HDC 5A Transport Tea","HDC 1i OLD ","HDC 1i Sky View 10",
                                  "MIS Transport Tea DDC 4","MIS Transport Tea - DDC 3","Pan India Event MIS"],
            "business_logic_54": ["Gojek"],
            "business_logic_55": ["Junglee MIS"],
            "business_logic_56": ["Tonbo"],
            "business_logic_57": ["Sinch_BLR"],
            "business_logic_58": ["Schneider-2"],
            "business_logic_59": ["other Events"],
            "business_logic_60": ["Telstra-Tuck Shop"],
            "business_logic_61": ["Airtel Adhoc", "Boomerang MIS"],
            "business_logic_62": ["LPG"],
            "business_logic_63": ["ABM -MEAL"],
            "business_logic_64": ["Junglee_NCR"],
            "business_logic_65": ["Nasdaq", "Cohesity"],


            "event_logic_1": ["WF Chennai Events","WF Hyd Events","WF BLR Events","Tekion.","Citrix Events","Amazon PNQ Events","infosys Event Sales","Other Events."], 
            "event_logic_2": [""],
            "event_logic_3": ["ICON CHN EVENT","Other Revenues Mumbai & Pune"],
            "event_logic_4": [""],
            "event_logic_5": [""],
            "event_logic_6": [],
            "event_logic_8": [""],
            "event_logic_9": [""],
            "event_logic_10": [""],
            "event_logic_11": [""],
            "event_logic_12": ["Airtel Event"],
            "event_logic_13": ["Icon-event-Bangalore"],


            "other_revenues": ["New Other Revenues"],
            "welfrgo_other_revenues": ["wellsFargo Other Revenues"],
            "karbon":["Karbon"]
    }

    business_logic_module = None
    for module_name, sheets in business_logic_sheets.items():
        if selected_sheet in sheets:
            business_logic_module = module_name
            break

    if business_logic_module:
        try:
            module = importlib.import_module(business_logic_module)
            business_logic_function = getattr(module, business_logic_module)
            # Pass both the single month df and the last 3 months df to the business logic
            business_logic_function(df_filtered, analysis_df, months_to_include)
            logging.info(f"Business logic '{business_logic_module}' applied successfully.")
            return module  # Return the imported module for further use
        except ModuleNotFoundError:
            st.error(f"Business logic module '{business_logic_module}' not found.")
            logging.error(f"Business logic module '{business_logic_module}' not found.")
        except AttributeError:
            st.error(f"Function '{business_logic_module}' not found in the module.")
            logging.error(f"Function '{business_logic_module}' not found in the module.")
        except Exception as e:
            st.error(f"Error applying business logic: {e}")
            logging.error(f"Error applying business logic: {e}")
    else:
        st.write("No business logic defined for this sheet.")
        logging.warning("No business logic defined for the selected sheet.")
    return None

def handle_pnl_section(df_filtered, month, module):
    """Handles the Auto P&L process with Punch and Clear buttons."""
    try:
        # Verify if the required P&L functions exist in the module
        if hasattr(module, 'punch_pnl_data_to_blank_row') and hasattr(module, 'clear_pnl_data'):
            st.write(f"Processing Auto P&L for the month: {month}")

            # Define the punch and clear functions
            punch_pnl_data = getattr(module, 'punch_pnl_data_to_blank_row')
            clear_pnl_data = getattr(module, 'clear_pnl_data')

            # Create two tabs: one for Punch and one for Clear actions
            tab1, tab2 = st.tabs(["Punch Data", "Clear Data"])

            # Ensure 'review id' or 'cost centre' exist and create 'identifier'
            if 'review id' in df_filtered.columns:
                df_filtered['identifier'] = df_filtered['review id']
            elif 'cost centre' in df_filtered.columns:
                df_filtered['identifier'] = df_filtered['cost centre']
            else:
                st.error("Neither 'review id' nor 'cost centre' columns found in the data.")
                return  # Stop execution if neither column exists

            # Process the P&L data directly using the identifier
            identifier = df_filtered['identifier'].iloc[0]  # Get the first identifier (assuming only one)

            with tab1:
                if st.button("Punch P&L"):
                    if not df_filtered.empty:
                        punch_pnl_data(df_filtered, month, P_AND_L_FILE_PATH)
                    else:
                        st.error("Filtered DataFrame is empty. Cannot punch data.")

            with tab2:
                if st.button("Clear P&L Data"):
                    if identifier:
                        clear_pnl_data(P_AND_L_FILE_PATH, month, identifier)
                    else:
                        st.error("No identifier available for clearing P&L data.")

            st.markdown("---")

        else:
            st.error(f"The module '{module.__name__}' does not contain the necessary P&L functions.")
            logging.error(f"The module '{module.__name__}' does not contain the required 'punch_pnl_data_to_blank_row' or 'clear_pnl_data' functions.")

    except Exception as e:
        st.error(f"Error in Auto P&L section: {e}")
        logging.error(f"Error in Auto P&L section: {e}")




# ----------------------------------- Dump Handling Section -----------------------------------

def handle_dump_section(df_filtered, month, module):
    """Handles the Dump creation process."""
    try:
        if st.button("Create Dump"):
            dump_function = getattr(module, 'dump_data')
            dump_function(df_filtered, month, DUMP_FILE_PATH)
        st.markdown("---")
    except Exception as e:
        st.error(f"Error in dump section: {e}")
        logging.error(f"Error in dump section: {e}")


def main():
    setup_page()
    uploaded_file = upload_file()

    if uploaded_file:
        if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.excel_file = None
            st.session_state.df = None
            st.session_state.selected_sheet = None

        if 'excel_file' not in st.session_state or st.session_state.excel_file is None:
            st.session_state.excel_file = read_excel_file(uploaded_file)
        if st.session_state.excel_file:
            selected_sheet = select_sheet(st.session_state.excel_file)
            if 'df' not in st.session_state or st.session_state.selected_sheet != selected_sheet:
                st.session_state.selected_sheet = selected_sheet
                st.session_state.df = read_sheet_to_dataframe(uploaded_file, selected_sheet)
                if st.session_state.df is not None:
                    st.session_state.df = preprocess_dataframe(st.session_state.df)
            if st.session_state.df is not None:
                df_filtered, month = filter_dataframe_by_month(st.session_state.df)
                if df_filtered is not None:
                    analysis_df, months_to_include = analyze_last_three_months(st.session_state.df, month)
                    module = apply_business_logic(df_filtered, analysis_df, months_to_include, selected_sheet, month)
                    if module:  # Only proceed if a valid module is returned
                        handle_pnl_section(df_filtered, month, module)  # P&L section with Punch and Clear buttons
                        handle_dump_section(df_filtered, month, module)  # Dump section

    else:
        st.write("Please upload an Excel file to proceed.")

if __name__ == "__main__":
    main()

