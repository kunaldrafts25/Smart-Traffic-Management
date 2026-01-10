"""Dashboard CSS styles as Python string constants."""

DASHBOARD_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00D4FF;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }

    .metric-card {
        background-color: #1E2329;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00D4FF;
        border: 1px solid #2D3748;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    .status-good { color: #48BB78; }
    .status-warning { color: #ED8936; }
    .status-error { color: #F56565; }

    .stApp {
        background-color: #0E1117;
    }

    .css-1d391kg {
        background-color: #1E2329;
    }

    [data-testid="metric-container"] {
        background-color: #1E2329;
        border: 1px solid #2D3748;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    .stAlert {
        background-color: #1E2329;
        border: 1px solid #2D3748;
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #1E2329;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #2D3748;
        color: #FAFAFA;
    }

    .stTabs [aria-selected="true"] {
        background-color: #00D4FF;
        color: #0E1117;
    }

    .js-plotly-plot {
        background-color: #1E2329 !important;
    }
</style>
"""

TRAFFIC_COLORS = {
    'RED': {
        'color': '#FF4444',
        'bg_color': '#2D1B1B',
        'text_color': '#FFFFFF',
        'emoji': '🔴',
        'icon': '⬛',
        'pattern': 'solid'
    },
    'YELLOW': {
        'color': '#FFB84D',
        'bg_color': '#2D2419',
        'text_color': '#000000',
        'emoji': '🟡',
        'icon': '⬜',
        'pattern': 'diagonal'
    },
    'GREEN': {
        'color': '#48BB78',
        'bg_color': '#1A2E1A',
        'text_color': '#FFFFFF',
        'emoji': '🟢',
        'icon': '⬜',
        'pattern': 'dots'
    },
    'EMERGENCY': {
        'color': '#FF6B6B',
        'bg_color': '#2D1A1A',
        'text_color': '#FFFFFF',
        'emoji': '🚨',
        'icon': '⚠️',
        'pattern': 'flash'
    }
}

DENSITY_COLORS = {
    'LOW': {
        'color': '#48BB78',
        'bg_color': '#1A2E1A',
        'text': 'Low Traffic',
        'emoji': '🟢',
        'threshold': 0.3
    },
    'MEDIUM': {
        'color': '#FFB84D',
        'bg_color': '#2D2419',
        'text': 'Medium Traffic',
        'emoji': '🟡',
        'threshold': 0.7
    },
    'HIGH': {
        'color': '#FF4444',
        'bg_color': '#2D1B1B',
        'text': 'High Traffic',
        'emoji': '🔴',
        'threshold': 1.0
    }
}


def get_traffic_density_color(density: float) -> dict:
    """Get color scheme based on traffic density with accessibility features."""
    if density <= DENSITY_COLORS['LOW']['threshold']:
        return DENSITY_COLORS['LOW']
    elif density <= DENSITY_COLORS['MEDIUM']['threshold']:
        return DENSITY_COLORS['MEDIUM']
    else:
        return DENSITY_COLORS['HIGH']
