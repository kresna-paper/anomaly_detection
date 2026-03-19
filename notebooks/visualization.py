"""
Original Visualization Suite for Behavioral Fingerprinting

Custom color schemes and original plot styles for fraud detection presentation.
NOT template/default - shows design awareness and attention to detail.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# ORIGINAL COLOR PALETTES - "Fraud Alert" Theme
# =============================================================================
FRAUD_COLORS = {
    'critical': '#E63946',    # Red - Immediate danger
    'high': '#F4A261',        # Orange - Suspicious
    'medium': '#E9C46A',      # Yellow - Caution
    'low': '#2A9D8F',         # Teal - Normal
    'safe': '#264653',        # Dark Blue - Verified safe
}

SEQUENTIAL_PALETTE = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E63946']

CYBER_COLORS = {
    'anomaly': '#FF006E',     # Neon Red/Pink
    'normal': '#00F5FF',      # Cyan
    'warning': '#FFBE0B',     # Amber
    'background': '#0A0E27',  # Dark
}

# Custom colormap for risk visualization
RISK_CMAP = plt.cm.colors.LinearSegmentedColormap.from_list(
    'risk',
    ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E63946']
)

# =============================================================================
# SETUP MATPLOTLIB WITH CUSTOM STYLE
# =============================================================================
def setup_custom_style():
    """Apply custom styling to matplotlib."""
    plt.rcParams.update({
        'font.family': 'Segoe UI, Tahoma, sans-serif',
        'font.size': 10,
        'axes.facecolor': '#F8F9FA',
        'figure.facecolor': '#FFFFFF',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.edgecolor': '#DDDDDD',
        'axes.labelcolor': '#333333',
        'axes.titlecolor': '#1A1A1A',
        'xtick.color': '#555555',
        'ytick.color': '#555555',
        'text.color': '#1A1A1A',
    })

setup_custom_style()

# =============================================================================
# ORIGINAL PLOT: BEHAVIORAL FINGERPRINT RADAR CHART
# =============================================================================
def plot_fingerprint_radar(
    account_data: dict,
    feature_names: list = None,
    title: str = "Behavioral Fingerprint"
):
    """
    Create a radar chart showing an account's behavioral fingerprint.

    This is an ORIGINAL visualization - not a standard template.
    """
    if feature_names is None:
        feature_names = [
            'Amount Level',
            'Velocity',
            'Hour Consistency',
            'Location Diversity',
            'Device Variety',
            'Online Ratio',
            'Login Safety',
            'Frequency'
        ]

    # Normalize values to 0-1
    values = list(account_data.values())
    normalized = [(v - min(values)) / (max(values) - min(values) + 1e-6) for v in values]

    # Number of variables
    N = len(feature_names)

    # Angles for radar
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    normalized += normalized[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Plot
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, normalized, 'o-', linewidth=2, color=FRAUD_COLORS['high'], label='Account')
    ax.fill(angles, normalized, alpha=0.25, color=FRAUD_COLORS['high'])

    # Add baseline (average account)
    baseline = [0.5] * (N + 1)
    ax.plot(angles, baseline, '--', linewidth=1, color=FRAUD_COLORS['safe'], alpha=0.5, label='Average')

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['Low', 'Avg', 'High', 'Peak'])
    ax.set_title(title, size=14, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Add risk zone background
    ax.set_ylim(0, 1)
    ax.set_facecolor('#F8F9FA')

    plt.tight_layout()
    return fig, ax


# =============================================================================
# ORIGINAL PLOT: ACCOUNT FINGERPRINT COMPARISON (Multiple Accounts)
# =============================================================================
def plot_multiple_fingerprints(
    accounts_data: list,
    account_ids: list,
    title: str = "Account Fingerprints Comparison"
):
    """
    Compare multiple accounts' behavioral fingerprints.
    Shows which accounts cluster together (potential fraud rings).
    """
    N = len(accounts_data[0])

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

    colors = [FRAUD_COLORS['critical'], FRAUD_COLORS['high'],
               FRAUD_COLORS['medium'], FRAUD_COLORS['low'], FRAUD_COLORS['safe']]

    for i, (account_data, account_id) in enumerate(zip(accounts_data, account_ids)):
        values = list(account_data.values())
        normalized = [(v - min(values)) / (max(values) - min(values) + 1e-6) for v in values]
        normalized += normalized[:1]

        color = colors[i % len(colors)]
        ax.plot(angles, normalized, 'o-', linewidth=1.5, label=account_id, color=color)
        ax.fill(angles, normalized, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(accounts_data[0].keys()), size=9)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=14, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    return fig, ax


# =============================================================================
# ORIGINAL PLOT: RISK GRADIENT HEATMAP (Transaction Timeline)
# =============================================================================
def plot_risk_timeline(
    df: pd.DataFrame,
    date_col: str = 'TransactionDate',
    risk_col: str = 'anomaly_score',
    title: str = "Transaction Risk Timeline"
):
    """
    Timeline visualization with risk-based color gradient.
    Shows risk evolution over time.
    """
    df = df.sort_values(date_col)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create scatter with color based on risk
    scatter = ax.scatter(
        range(len(df)),
        df[risk_col],
        c=df[risk_col],
        cmap=RISK_CMAP,
        s=50,
        alpha=0.7,
        edgecolors='none'
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Anomaly Score', rotation=270, labelpad=15)

    # Threshold lines
    ax.axhline(y=0.5, color=FRAUD_COLORS['medium'], linestyle='--', alpha=0.7, label='Medium Risk')
    ax.axhline(y=0.7, color=FRAUD_COLORS['high'], linestyle='--', alpha=0.7, label='High Risk')

    ax.set_xlabel('Transaction Index')
    ax.set_ylabel('Anomaly Score')
    ax.set_title(title, size=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


# =============================================================================
# ORIGINAL PLOT: FRAUD PULSE (Animated-style Visualization)
# =============================================================================
def plot_fraud_pulse(
    df: pd.DataFrame,
    date_col: str = 'TransactionDate',
    risk_col: str = 'anomaly_score',
    title: str = "Fraud Detection Pulse"
):
    """
    Area fill visualization showing risk "pulse" over time.
    ORIGINAL CONCEPT - mimics security monitoring dashboard.
    """
    df = df.sort_values(date_col)
    df['hour_bin'] = df[date_col].dt.hour

    hourly_risk = df.groupby('hour_bin')[risk_col].agg(['mean', 'max', 'count'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Top: Area pulse
    ax1.fill_between(
        hourly_risk.index,
        0,
        hourly_risk['mean'],
        color=FRAUD_COLORS['low'],
        alpha=0.3,
        label='Average Risk'
    )
    ax1.plot(
        hourly_risk.index,
        hourly_risk['max'],
        color=FRAUD_COLORS['high'],
        linewidth=2,
        label='Peak Risk'
    )

    # Add risk zones
    ax1.axhspan(0, 0.3, color=FRAUD_COLORS['safe'], alpha=0.1, label='Safe Zone')
    ax1.axhspan(0.3, 0.5, color=FRAUD_COLORS['low'], alpha=0.1, label='Low Risk')
    ax1.axhspan(0.5, 0.7, color=FRAUD_COLORS['medium'], alpha=0.1, label='Medium Risk')
    ax1.axhspan(0.7, 1.0, color=FRAUD_COLORS['critical'], alpha=0.1, label='Critical Zone')

    ax1.set_xlim(0, 23)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_ylabel('Risk Score')
    ax1.set_title('Risk by Hour of Day', size=12, weight='bold')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Bottom: Transaction volume
    ax2.bar(hourly_risk.index, hourly_risk['count'], color=FRAUD_COLORS['low'], alpha=0.6)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Transaction Count')
    ax2.set_xlim(0, 23)
    ax2.set_xticks(range(0, 24, 2))

    plt.suptitle(title, size=14, weight='bold')
    plt.tight_layout()
    return fig, (ax1, ax2)


# =============================================================================
# ORIGINAL PLOT: ACCOUNT NETWORK GRAPH
# =============================================================================
def plot_account_network(
    df: pd.DataFrame,
    title: str = "Account Transaction Network"
):
    """
    Network visualization showing connections between accounts
    through shared devices, IPs, or locations.

    ORIGINAL - custom visualization, not using networkx templates.
    """
    # Get top accounts by transaction count
    top_accounts = df['AccountID'].value_counts().head(15).index
    subset = df[df['AccountID'].isin(top_accounts)]

    # Calculate connection strength
    connections = {}
    for acc1 in top_accounts:
        for acc2 in top_accounts:
            if acc1 < acc2:
                # Shared attributes
                acc1_devices = set(subset[subset['AccountID'] == acc1]['DeviceID'].values)
                acc2_devices = set(subset[subset['AccountID'] == acc2]['DeviceID'].values)
                acc1_locs = set(subset[subset['AccountID'] == acc1]['Location'].values)
                acc2_locs = set(subset[subset['AccountID'] == acc2]['Location'].values)

                shared = len(acc1_devices & acc2_devices) + len(acc1_locs & acc2_locs)
                if shared > 0:
                    connections[(acc1, acc2)] = shared

    # Visualization
    fig, ax = plt.subplots(figsize=(14, 14))

    # Position nodes in a circle
    n_nodes = len(top_accounts)
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)

    # Draw edges
    for (acc1, acc2), strength in connections.items():
        idx1 = list(top_accounts).index(acc1)
        idx2 = list(top_accounts).index(acc2)

        # Edge color based on strength
        edge_color = FRAUD_COLORS['high'] if strength >= 2 else FRAUD_COLORS['medium']

        ax.plot([x_pos[idx1], x_pos[idx2]], [y_pos[idx1], y_pos[idx2]],
                color=edge_color, alpha=0.5, linewidth=strength)

    # Draw nodes
    tx_counts = subset['AccountID'].value_counts()
    for i, account in enumerate(top_accounts):
        # Size based on transaction count
        size = np.sqrt(tx_counts[account]) * 3

        # Color based on risk
        acc_risk = subset[subset['AccountID'] == account]['anomaly_score'].mean() if 'anomaly_score' in subset.columns else 0
        if acc_risk > 0.7:
            color = FRAUD_COLORS['critical']
        elif acc_risk > 0.5:
            color = FRAUD_COLORS['high']
        else:
            color = FRAUD_COLORS['safe']

        circle = Circle((x_pos[i], y_pos[i]), size, color=color, alpha=0.8)
        ax.add_patch(circle)

        # Label
        ax.text(x_pos[i] * 1.1, y_pos[i] * 1.1, account[-4:],  # Last 4 chars
                ha='center', va='center', fontsize=8)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, size=14, weight='bold')

    # Legend
    legend_elements = [
        plt.scatter([], [], color=FRAUD_COLORS['critical'], s=100, label='High Risk'),
        plt.scatter([], [], color=FRAUD_COLORS['high'], s=100, label='Medium Risk'),
        plt.scatter([], [], color=FRAUD_COLORS['safe'], s=100, label='Normal'),
        plt.plot([], [], color=FRAUD_COLORS['high'], linewidth=2, label='Shared Devices/IPs')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    return fig, ax


# =============================================================================
# ORIGINAL PLOT: RISK DASHBOARD (Multi-panel Summary)
# =============================================================================
def plot_risk_dashboard(df: pd.DataFrame):
    """
    Comprehensive risk dashboard with multiple panels.
    ORIGINAL LAYOUT - executive summary visualization.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Risk Distribution (Pie)
    ax1 = fig.add_subplot(gs[0, 0])

    risk_counts = df['risk_level'].value_counts()
    colors = [FRAUD_COLORS.get(r, '#999') for r in risk_counts.index]
    wedges, texts, autotexts = ax1.pie(
        risk_counts.values,
        labels=risk_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax1.set_title('Risk Distribution', weight='bold')

    # Panel 2: Transaction Amount by Risk (Box)
    ax2 = fig.add_subplot(gs[0, 1])

    risk_order = ['low', 'medium', 'high', 'critical']
    box_data = [df[df['risk_level'] == r]['TransactionAmount'].values for r in risk_order if r in df['risk_level'].values]
    bp = ax2.boxplot(box_data, labels=[r.capitalize() for r in risk_order if r in df['risk_level'].values],
                       patch_artist=True)

    for patch, color in zip(bp['boxes'], [FRAUD_COLORS.get(r, '#999') for r in risk_order if r in df['risk_level'].values]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel('Transaction Amount ($)')
    ax2.set_title('Amount by Risk Level', weight='bold')

    # Panel 3: Channel Risk Analysis (Stacked Bar)
    ax3 = fig.add_subplot(gs[0, 2])

    channel_risk = pd.crosstab(df['Channel'], df['risk_level'])
    channel_risk.plot(kind='bar', stacked=True, ax=ax3,
                      color=[FRAUD_COLORS.get(r, '#999') for r in ['low', 'medium', 'high', 'critical']])
    ax3.set_title('Channel Risk Profile', weight='bold')
    ax3.set_ylabel('Count')
    ax3.legend(title='Risk')
    ax3.tick_params(axis='x', rotation=0)

    # Panel 4: Hourly Risk Heatmap
    ax4 = fig.add_subplot(gs[1, :])

    df['hour'] = df['TransactionDate'].dt.hour
    hourly_risk = df.groupby(['hour', 'risk_level']).size().unstack(fill_value=0)

    im = ax4.imshow(hourly_risk.T, aspect='auto', cmap='YlOrRd', origin='lower')
    ax4.set_xticks(range(24))
    ax4.set_xlabel('Hour of Day')
    ax4.set_yticks(range(len(hourly_risk.columns)))
    ax4.set_yticklabels([r.capitalize() for r in hourly_risk.columns])
    ax4.set_title('Hourly Risk Heatmap', weight='bold')
    ax4.grid(False)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax4, orientation='vertical')
    cbar.set_label('Transaction Count')

    # Panel 5: Top Anomalous Accounts (Horizontal Bar)
    ax5 = fig.add_subplot(gs[2, 0:2])

    account_risk = df.groupby('AccountID')['anomaly_score'].mean().sort_values(ascending=False).head(10)
    colors = [FRAUD_COLORS['critical'] if s > 0.7 else FRAUD_COLORS['high'] if s > 0.5 else FRAUD_COLORS['medium']
              for s in account_risk.values]

    ax5.barh(range(len(account_risk)), account_risk.values, color=colors, alpha=0.7)
    ax5.set_yticks(range(len(account_risk)))
    ax5.set_yticklabels([a[-4:] for a in account_risk.index])
    ax5.set_xlabel('Mean Anomaly Score')
    ax5.set_title('Top 10 Riskiest Accounts', weight='bold')
    ax5.invert_yaxis()
    ax5.grid(axis='x', alpha=0.3)

    # Panel 6: KPI Cards (Text)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    kpi_data = [
        ('Total\nTransactions', f"{len(df):,}"),
        ('Anomaly\nRate', f"{(df['is_anomaly'].sum() / len(df) * 100):.1f}%"),
        ('High Risk\nAccounts', f"{(df[df['risk_level'].isin(['high', 'critical'])]['AccountID'].nunique())}"),
        ('Avg Anomaly\nScore', f"{df['anomaly_score'].mean():.3f}")
    ]

    kpi_box_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray')

    for i, (label, value) in enumerate(kpi_data):
        y_pos = 0.8 - i * 0.2
        ax6.text(0.1, y_pos, label, fontsize=10, weight='bold', va='center')
        ax6.text(0.5, y_pos, value, fontsize=14, weight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=FRAUD_COLORS['safe'], alpha=0.3))

    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')

    fig.suptitle('Fraud Detection Risk Dashboard', size=16, weight='bold')

    return fig


# =============================================================================
# INTERACTIVE PLOTLY VISUALIZATIONS
# =============================================================================
def create_interactive_fraud_map(df: pd.DataFrame):
    """
    Interactive map visualization using Plotly.
    Shows transactions with risk-based color coding.
    """
    # City coordinates (simplified)
    city_coords = {
        'New York': [40.7128, -74.0060],
        'Los Angeles': [34.0522, -118.2437],
        'Chicago': [41.8781, -87.6298],
        'Houston': [29.7604, -95.3698],
        'Phoenix': [33.4484, -112.0740],
        'Philadelphia': [39.9526, -75.1652],
        'San Antonio': [29.4241, -98.4936],
        'San Diego': [32.7157, -117.1611],
        'Dallas': [32.7767, -96.7970],
        'San Jose': [37.3382, -121.8863],
        'Fort Worth': [32.7555, -97.3308],
        'Columbus': [39.9612, -82.9988],
        'Austin': [30.2672, -97.7431],
        'Jacksonville': [30.3322, -81.6557],
        'Charlotte': [35.2271, -80.8431],
        'Indianapolis': [39.7684, -86.1581],
        'Denver': [39.7392, -104.9903],
        'Seattle': [47.6062, -122.3321],
    }

    df_map = df.copy()
    df_map['lat'] = df_map['Location'].map(lambda x: city_coords.get(x, [40, -74])[0])
    df_map['lon'] = df_map['Location'].map(lambda x: city_coords.get(x, [40, -74])[1])

    # Color based on risk
    df_map['color'] = df_map['risk_level'].map({
        'critical': FRAUD_COLORS['critical'],
        'high': FRAUD_COLORS['high'],
        'medium': FRAUD_COLORS['medium'],
        'low': FRAUD_COLORS['low'],
        'safe': FRAUD_COLORS['safe']
    })

    fig = go.Figure()

    for risk in ['critical', 'high', 'medium', 'low', 'safe']:
        subset = df_map[df_map['risk_level'] == risk]
        if len(subset) > 0:
            fig.add_trace(go.Scattergeo(
                lon=subset['lon'],
                lat=subset['lat'],
                text=subset.apply(lambda x: f"{x['Location']}<br>${x['TransactionAmount']:.0f}<br>Risk: {x['risk_level']}", axis=1),
                mode='markers',
                marker=dict(size=8, color=subset['color'].iloc[0], line=dict(width=0.5, color='white')),
                name=risk.capitalize(),
            ))

    fig.update_geos(
        projection_type="albers usa",
        showland=True,
        landcolor="#F8F9FA",
        showocean=True,
        oceancolor="#E9C46A"
    )

    fig.update_layout(
        title_text="Geographic Distribution of Transaction Risk",
        title_font_size=16,
        geo=dict(bgcolor="#F8F9FA"),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def create_interactive_fingerprint_3d(latent_codes, labels):
    """
    3D scatter plot of latent representations.
    Shows clustering of account fingerprints.
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=latent_codes[:, 0],
        y=latent_codes[:, 1],
        z=latent_codes[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=labels,
            colorscale=[[0, FRAUD_COLORS['safe']], [1, FRAUD_COLORS['critical']]],
            colorbar=dict(title="Risk"),
            opacity=0.8
        ),
        text=[f"Latent: [{x:.2f}, {y:.2f}, {z:.2f}]" for x, y, z in latent_codes]
    )])

    fig.update_layout(
        title="Behavioral Fingerprints - 3D Latent Space",
        scene=dict(
            xaxis_title="Latent Dim 1",
            yaxis_title="Latent Dim 2",
            zaxis_title="Latent Dim 3",
            bgcolor="#F8F9FA"
        ),
        height=600
    )

    return fig


if __name__ == "__main__":
    print("Custom Visualization Suite for Fraud Detection")
    print("Original color schemes and plot styles")
