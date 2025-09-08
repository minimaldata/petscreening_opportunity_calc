import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="PetScreening Revenue Simulator", layout="wide")


# ---------- Sidebar inputs
st.sidebar.header("Technical Assumptions")

# Pet profile assumptions
st.sidebar.subheader("Pet Profile Settings")
compliance = st.sidebar.number_input(
    "Compliance Rate (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=95.0, 
    step=1.0,
    format="%.1f"
) / 100.0

pets_per_compliant = st.sidebar.number_input(
    "Pets per Compliant Lease", 
    min_value=0.0, 
    max_value=5.0, 
    value=0.3, 
    step=0.05,
    format="%.2f"
)

# Revenue settings
st.sidebar.subheader("Revenue Settings")
monthly_rent = st.sidebar.number_input(
    "Monthly Pet Rent ($)", 
    min_value=0.0, 
    max_value=1000.0, 
    value=30.0, 
    step=1.0,
    format="%.0f"
)

deposit_amount = st.sidebar.number_input(
    "Pet Deposit ($)", 
    min_value=0.0, 
    max_value=5000.0, 
    value=250.0, 
    step=5.0,
    format="%.0f"
)

# Additional inputs (collapsed by default)
with st.sidebar.expander("Additional Inputs"):
    incrementality_factor = st.number_input(
        "Incrementality Factor (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=30.0, 
        step=1.0,
        format="%.1f"
    ) / 100.0
    
    retention_rate = st.number_input(
        "Lease Retention Rate (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=60.0, 
        step=1.0,
        format="%.1f"
    ) / 100.0
    
    st.write("---")
    st.write(f"**Lease Start Seasonality**")
    st.caption("Monthly % distribution of lease starts")

    default_seasonality = [5,5,7,8,10,12,12,11,9,7,6,8]  # Jan..Dec
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    
    seasonality = []
    cols = st.columns(3)
    for i, (month, default_pct) in enumerate(zip(month_names, default_seasonality)):
        with cols[i % 3]:
            seasonality.append(st.number_input(
                month, 
                min_value=0.0, 
                max_value=100.0, 
                value=float(default_pct), 
                step=1.0,
                format="%.1f",
                key=f"seasonality_{month}"
            ))
    
    # Check if seasonality sums to 100% and show warning
    seasonality_sum = sum(seasonality)
    warning_text = " ⚠️ Distribution doesn't add to 100%" if abs(seasonality_sum - 100) > 0.1 else ""
    
    st.write(f"{warning_text}")
    
    # Normalize seasonality to sum to 100%
    if seasonality_sum > 0:
        seasonality = [s / seasonality_sum * 100 for s in seasonality]
    else:
        seasonality = default_seasonality

# ---------- Helper functions
def get_month_year_options():
    """Generate month/year options for selector"""
    current_date = dt.date.today()
    options = []
    for i in range(60):  # 5 years of options
        date = current_date.replace(day=1) + dt.timedelta(days=32*i)
        date = date.replace(day=1)  # Ensure first of month
        options.append(date)
    return options

def format_month_year(date):
    """Format date as 'Month YYYY'"""
    return date.strftime("%B %Y")

# ---------- Main page inputs
st.title("PetScreening Revenue Simulator")
st.caption("Compare revenue impact of different rollout scenarios")

# Top section: 2x2 grid layout
col1, col2 = st.columns(2)

# Left column: Project Setup and Ramp Schedule
with col1:
    st.subheader("Project Setup")
    
    # Month/Year selector
    month_options = get_month_year_options()
    month_labels = [format_month_year(d) for d in month_options]
    
    selected_month_idx = st.selectbox(
        "Start Month",
        range(len(month_options)),
        format_func=lambda x: month_labels[x],
        index=0
    )
    start_date = month_options[selected_month_idx]
    
    # Unit count
    unit_count = st.number_input(
        "Total Units",
        min_value=1,
        max_value=10000000,
        value=5000,
        step=100,
        format="%d"
    )
    
    # Ramp Schedule
    st.subheader("Ramp Schedule")
    st.caption("Percentage of units live at each milestone")
    
    # Calculate milestone dates (including start month)
    milestone_months = [0, 3, 6, 9, 12, 18]
    milestone_dates = []
    for months in milestone_months:
        if months == 0:
            milestone_dates.append(start_date)
        else:
            # Add months to start date
            year = start_date.year
            month = start_date.month + months
            while month > 12:
                month -= 12
                year += 1
            milestone_dates.append(dt.date(year, month, 1))
    
    # Create ramp table
    ramp_data = []
    # Default values: A = [10,50,100,100,100,100], B = [0,0,0,10,50,100]
    default_a_values = [10, 50, 100, 100, 100, 100]
    default_b_values = [0, 0, 0, 10, 50, 100]
    
    for i, (months, date) in enumerate(zip(milestone_months, milestone_dates)):
        if months == 0:
            milestone_label = "Start"
        else:
            milestone_label = f"{months} months"
        
        ramp_data.append({
            "Milestone": milestone_label,
            "Month": format_month_year(date),
            "Scenario A (%)": default_a_values[i],
            "Scenario B (%)": default_b_values[i]
        })
    
    # Display editable table and capture the values
    edited_ramp_data = st.data_editor(
        pd.DataFrame(ramp_data),
        column_config={
            "Milestone": st.column_config.TextColumn("Milestone", disabled=True),
            "Month": st.column_config.TextColumn("Month", disabled=True),
            "Scenario A (%)": st.column_config.NumberColumn("Scenario A (%)", min_value=0, max_value=100, step=5),
            "Scenario B (%)": st.column_config.NumberColumn("Scenario B (%)", min_value=0, max_value=100, step=5)
        },
        hide_index=True,
        use_container_width=True,
        key="ramp_table"
    )
    
    # Extract ramp percentages from the edited table
    ramp_A_percentages = edited_ramp_data["Scenario A (%)"].tolist()
    ramp_B_percentages = edited_ramp_data["Scenario B (%)"].tolist()

# Right column: Graph controls and graph
with col2:
    st.subheader("Visual Analysis")
    
    # Graph controls
    data_type = st.selectbox(
        "What to display:",
        ["pets", "pet rent revenue", "pet deposit revenue", "total revenue"],
        index=3,
        key="data_type_selectbox"
    )
    
    show_cumulative = st.checkbox("Show Cumulative", value=True, key="cumulative_checkbox")
    
    # Graph will be created and displayed here after calculations

# ---------- Core calculations
def month_series(start_date: dt.date, months: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start_date, periods=months, freq="MS")

def rolling_tail(series: pd.Series, tail_months: int) -> pd.Series:
    # Sum of last N months including current month
    return series.rolling(tail_months, min_periods=1).sum()

def cumulative(series: pd.Series) -> pd.Series:
    return series.cumsum()

def month_name_idx(dti: pd.DatetimeIndex) -> pd.Index:
    return dti.month - 1  # 0..11 to index Jan..Dec rows

# Build timeline (24 months)
dates = month_series(start_date, 24)
midx = month_name_idx(dates)  # 0..11
month_names = dates.strftime("%b %Y")

# Lease starts from seasonality
lease_starts = (unit_count * np.array(seasonality)[midx] / 100).round().astype(int)

# Build ramps as step functions over dates
def ramp_vector(dates_idx: pd.DatetimeIndex, ramp_percentages: list) -> np.ndarray:
    out = np.zeros(len(dates_idx))
    milestone_months = [0, 3, 6, 9, 12, 18]
    
    for i, d in enumerate(dates_idx):
        # Find which milestone this month falls into
        months_from_start = (d.year - start_date.year) * 12 + (d.month - start_date.month)
        
        # Find the appropriate ramp percentage
        ramp_pct = 0
        for j, milestone in enumerate(milestone_months):
            if months_from_start >= milestone:
                ramp_pct = ramp_percentages[j]
            else:
                break
        out[i] = ramp_pct / 100.0
    return out

# Use the ramp percentages from the editable table

rA = ramp_vector(dates, ramp_A_percentages)
rB = ramp_vector(dates, ramp_B_percentages)

# Pet profiles (new per month) - matching the Google Sheet formula
base_factor = compliance * pets_per_compliant
new_A = np.round(lease_starts * base_factor * rA * incrementality_factor).astype(int)
new_B = np.round(lease_starts * base_factor * rB * incrementality_factor).astype(int)

# Revenues
# Deposit: collected on creation (matching Google Sheet: B38*$B$34*$B$35)
dep_A = new_A * deposit_amount * retention_rate
dep_B = new_B * deposit_amount * retention_rate

# Pet-rent: rolling 12-month tail (matching Google Sheet MMULT formula)
active_A = rolling_tail(pd.Series(new_A), 12)
active_B = rolling_tail(pd.Series(new_B), 12)

rent_A = (active_A * monthly_rent).round(2)
rent_B = (active_B * monthly_rent).round(2)

# Totals and cumulative
monthly_rev_A = rent_A + dep_A
monthly_rev_B = rent_B + dep_B
cum_rev_A = cumulative(monthly_rev_A)
cum_rev_B = cumulative(monthly_rev_B)

# ---------- Output table
st.write("---")
st.subheader("Full Table")

# Toggle to show/hide detailed table
show_table = st.checkbox("Show Full Table", value=False)

# Create the main results table
df = pd.DataFrame({
    "Month": month_names,
    "Scenario A (incremental pet profiles)": new_A,
    "Scenario B (incremental pet profiles)": new_B,
    "A (cumulative)": cumulative(pd.Series(new_A)),
    "B (cumulative)": cumulative(pd.Series(new_B)),
    "A (pet rent revenue)": rent_A,
    "B (pet rent revenue)": rent_B,
    "A (pet rent cum.)": cumulative(rent_A),
    "B (pet rent cum.)": cumulative(rent_B),
    "A (deposit rev)": dep_A,
    "B (deposit rev)": dep_B,
    "A (deposit cum.)": cumulative(dep_A),
    "B (deposit cum.)": cumulative(dep_B),
    "A (rev cum.)": cum_rev_A,
    "B (rev cum)": cum_rev_B,
})

# Display the table only if checkbox is checked
if show_table:
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------- Core calculations

# Prepare data based on selections and create graph in right column
with col2:
    if data_type == "pets":
        if show_cumulative:
            y_A = cumulative(pd.Series(new_A))
            y_B = cumulative(pd.Series(new_B))
            y_label = "Cumulative Pet Profiles"
        else:
            y_A = new_A
            y_B = new_B
            y_label = "New Pet Profiles per Month"
    elif data_type == "pet rent revenue":
        if show_cumulative:
            y_A = cumulative(rent_A)
            y_B = cumulative(rent_B)
            y_label = "Cumulative Pet Rent Revenue ($)"
        else:
            y_A = rent_A
            y_B = rent_B
            y_label = "Monthly Pet Rent Revenue ($)"
    elif data_type == "pet deposit revenue":
        if show_cumulative:
            y_A = cumulative(dep_A)
            y_B = cumulative(dep_B)
            y_label = "Cumulative Pet Deposit Revenue ($)"
        else:
            y_A = dep_A
            y_B = dep_B
            y_label = "Monthly Pet Deposit Revenue ($)"
    elif data_type == "total revenue":
        if show_cumulative:
            y_A = cum_rev_A
            y_B = cum_rev_B
            y_label = "Cumulative Total Revenue ($)"
        else:
            y_A = monthly_rev_A
            y_B = monthly_rev_B
            y_label = "Monthly Total Revenue ($)"

    # Create the graph data
    graph_data = pd.DataFrame({
        "Month": dates,  # Use actual dates instead of formatted strings
        "Scenario A": y_A,
        "Scenario B": y_B
    })

    # Create Plotly line chart
    fig = go.Figure()

    # Calculate difference for hover text
    difference = graph_data["Scenario A"] - graph_data["Scenario B"]
    
    # Format numbers for tooltip
    def format_number(num):
        if abs(num) < 1000:
            return f"{int(num):,}"
        elif abs(num) < 1_000_000:
            return f"{num/1000:.2f}K"
        elif abs(num) < 1_000_000_000:
            return f"{num/1_000_000:.2f}M"
        elif abs(num) < 1_000_000_000_000:
            return f"{num/1_000_000_000:.2f}B"
        else:
            return f"{num/1_000_000_000_000:.2f}T"
    
    # Format the data for tooltips
    formatted_a = [format_number(x) for x in graph_data["Scenario A"]]
    formatted_b = [format_number(x) for x in graph_data["Scenario B"]]
    formatted_diff = [format_number(x) for x in difference]
    
    # Add Scenario A line
    fig.add_trace(go.Scatter(
        x=graph_data["Month"],
        y=graph_data["Scenario A"],
        mode='lines+markers',
        name='Scenario A',
        hovertemplate='<b>Scenario A:</b> %{customdata[0]}<br><i>diff: %{customdata[2]}</i><br><extra></extra>',
        customdata=list(zip(formatted_a, formatted_b, formatted_diff))
    ))

    # Add Scenario B line
    fig.add_trace(go.Scatter(
        x=graph_data["Month"],
        y=graph_data["Scenario B"],
        mode='lines+markers',
        name='Scenario B',
        hovertemplate='<b>Scenario B:</b> %{customdata[1]}<br><extra></extra>',
        customdata=list(zip(formatted_a, formatted_b, formatted_diff))
    ))

    # Update layout
    fig.update_layout(
        title=f"{y_label}",
        xaxis_title="Month",
        yaxis_title=y_label,
        hovermode='x unified',
        height=500,
        xaxis=dict(showgrid=False, showline=True, linecolor='black'),
        yaxis=dict(showgrid=False, showline=True, linecolor='black'),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add a caption
    st.caption(f"**{y_label}** - {data_type.replace('_', ' ').title()}")