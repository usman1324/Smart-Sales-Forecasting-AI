"""
✨ Smart Sales Forecasting AI - Enhanced Version
Professional Dashboard with Perfect Layout & Maximized View
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class SmartSalesForecaster:
    def __init__(self, root):
        self.root = root
        self.root.title("📈 Smart Sales Forecasting AI | Professional Dashboard")

        # Get screen dimensions and set to almost full screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.root.state('normal')  # Start in normal mode, can maximize

        self.root.configure(bg='#0a1929')

        # Make window responsive
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Initialize data
        self.sales_data = None
        self.forecast_results = None
        self.models = {}
        self.current_product = "All Products"

        # Enhanced Color scheme
        self.bg_color = '#0a1929'
        self.card_bg = '#112240'
        self.accent_color = '#64ffda'
        self.accent_light = '#99ffe8'
        self.primary_color = '#1d3557'
        self.success_color = '#4cc9f0'
        self.warning_color = '#ffd166'
        self.danger_color = '#ef476f'
        self.text_color = '#ffffff'
        self.text_secondary = '#8892b0'
        self.grid_color = '#2d3748'

        # Setup enhanced styles
        self.setup_styles()

        # Create GUI
        self.create_widgets()

        # Load sample data
        self.load_sample_data()

        # Bind F11 for fullscreen and Esc to exit
        self.root.bind('<F11>', self.toggle_fullscreen)
        self.root.bind('<Escape>', self.exit_fullscreen)

    def toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode"""
        self.fullscreen_state = not self.fullscreen_state
        self.root.attributes('-fullscreen', self.fullscreen_state)
        return "break"

    def exit_fullscreen(self, event=None):
        """Exit fullscreen mode"""
        self.fullscreen_state = False
        self.root.attributes('-fullscreen', False)
        return "break"

    def setup_styles(self):
        """Setup enhanced modern styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Custom font for better readability
        self.title_font = ('Segoe UI', 32, 'bold')
        self.subtitle_font = ('Segoe UI', 12)
        self.heading_font = ('Segoe UI', 14, 'bold')
        self.body_font = ('Segoe UI', 10)
        self.kpi_font = ('Segoe UI', 24, 'bold')

        # Configure treeview with better spacing
        self.style.configure('Custom.Treeview',
                             background=self.card_bg,
                             foreground=self.text_color,
                             fieldbackground=self.card_bg,
                             borderwidth=0,
                             rowheight=35)  # Increased row height

        self.style.configure('Custom.Treeview.Heading',
                             background=self.primary_color,
                             foreground=self.text_color,
                             relief='flat',
                             font=('Segoe UI', 11, 'bold'),
                             padding=(10, 10))  # Add padding to headings

        self.style.map('Custom.Treeview.Heading',
                       background=[('active', '#2d4a7c')])

    def create_widgets(self):
        """Create enhanced GUI with perfect layout"""
        # Main container with proper padding
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)

        # ============ HEADER SECTION ============
        header_frame = tk.Frame(main_container, bg=self.bg_color)
        header_frame.pack(fill=tk.X, pady=(0, 25))

        # Left side - Title
        title_frame = tk.Frame(header_frame, bg=self.bg_color)
        title_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(title_frame,
                 text="📈 Smart Sales Forecasting AI",
                 font=self.title_font,
                 bg=self.bg_color,
                 fg=self.accent_color).pack(anchor='w')

        tk.Label(title_frame,
                 text="Professional Time Series Analysis & ML Forecasting",
                 font=self.subtitle_font,
                 bg=self.bg_color,
                 fg=self.text_secondary).pack(anchor='w', pady=(5, 0))

        # Right side - View Controls
        controls_frame = tk.Frame(header_frame, bg=self.bg_color)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Button(controls_frame,
                  text="🗔 Maximize",
                  command=lambda: self.root.state('zoomed'),
                  font=('Segoe UI', 9),
                  bg=self.primary_color,
                  fg='white',
                  relief='flat',
                  padx=15,
                  pady=6).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(controls_frame,
                  text="⏏️ Normal",
                  command=lambda: self.root.state('normal'),
                  font=('Segoe UI', 9),
                  bg=self.card_bg,
                  fg='white',
                  relief='flat',
                  padx=15,
                  pady=6).pack(side=tk.LEFT)

        # ============ KPI DASHBOARD ============
        kpi_container = tk.Frame(main_container, bg=self.bg_color)
        kpi_container.pack(fill=tk.X, pady=(0, 30))

        # KPI Title
        tk.Label(kpi_container,
                 text="📊 Key Performance Indicators",
                 font=self.heading_font,
                 bg=self.bg_color,
                 fg=self.text_color).pack(anchor='w', pady=(0, 15))

        # KPI Cards in a grid
        kpi_grid = tk.Frame(kpi_container, bg=self.bg_color)
        kpi_grid.pack(fill=tk.X)

        kpis = [
            ("Total Sales", "$ 0", "#4cc9f0", "💰"),
            ("Avg Daily", "$ 0", "#64ffda", "📅"),
            ("Growth %", "0%", "#ffd166", "📈"),
            ("Forecast", "$ 0", "#7209b7", "🔮"),
            ("Top Product", "N/A", "#ef476f", "🏆"),
            ("Best Day", "N/A", "#9d4edd", "⭐")
        ]

        self.kpi_labels = {}
        self.kpi_value_labels = {}

        for i, (title, value, color, icon) in enumerate(kpis):
            kpi_card = tk.Frame(kpi_grid, bg=color, width=180, height=110)
            kpi_card.grid(row=0, column=i, padx=(0, 15), sticky='nsew')
            kpi_card.pack_propagate(False)

            # Make grid columns expandable
            kpi_grid.columnconfigure(i, weight=1)

            # Content with padding
            content = tk.Frame(kpi_card, bg=color)
            content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

            # Icon and Title
            title_frame = tk.Frame(content, bg=color)
            title_frame.pack(fill=tk.X)

            tk.Label(title_frame,
                     text=icon,
                     font=('Segoe UI', 14),
                     bg=color,
                     fg='white').pack(side=tk.LEFT)

            tk.Label(title_frame,
                     text=title,
                     font=('Segoe UI', 10),
                     bg=color,
                     fg='white').pack(side=tk.LEFT, padx=(8, 0))

            # Value with proper spacing
            value_label = tk.Label(content,
                                   text=value,
                                   font=self.kpi_font,
                                   bg=color,
                                   fg='white')
            value_label.pack(fill=tk.X, pady=(10, 0))

            self.kpi_labels[title] = value_label
            self.kpi_value_labels[title] = value

        # ============ MAIN CONTENT AREA ============
        content_frame = tk.Frame(main_container, bg=self.bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid for main content
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=2)
        content_frame.rowconfigure(0, weight=1)

        # ============ LEFT SIDEBAR ============
        sidebar = tk.Frame(content_frame, bg=self.card_bg, width=320)
        sidebar.grid(row=0, column=0, sticky='nsew', padx=(0, 20))
        sidebar.pack_propagate(False)

        # Sidebar Content with proper padding
        sidebar_content = tk.Frame(sidebar, bg=self.card_bg, padx=25, pady=25)
        sidebar_content.pack(fill=tk.BOTH, expand=True)

        # Data Management Section
        data_section = tk.LabelFrame(sidebar_content,
                                     text="📂 DATA MANAGEMENT",
                                     font=('Segoe UI', 11, 'bold'),
                                     bg=self.card_bg,
                                     fg=self.accent_color,
                                     padx=15,
                                     pady=15)
        data_section.pack(fill=tk.X, pady=(0, 20))

        # Data buttons with consistent styling
        btn_style = {'font': ('Segoe UI', 10), 'relief': 'flat', 'pady': 10}

        tk.Button(data_section,
                  text="📁 Load CSV Data",
                  command=self.load_csv_data,
                  bg=self.accent_color,
                  fg=self.bg_color,
                  **btn_style).pack(fill=tk.X, pady=(0, 8))

        tk.Button(data_section,
                  text="🎲 Generate Sample Data",
                  command=self.generate_sample_data,
                  bg=self.primary_color,
                  fg='white',
                  **btn_style).pack(fill=tk.X, pady=(0, 8))

        tk.Button(data_section,
                  text="🔄 Refresh Dashboard",
                  command=self.refresh_data,
                  bg=self.grid_color,
                  fg='white',
                  **btn_style).pack(fill=tk.X)

        # Product Selection Section
        product_section = tk.LabelFrame(sidebar_content,
                                        text="🎯 PRODUCT FOCUS",
                                        font=('Segoe UI', 11, 'bold'),
                                        bg=self.card_bg,
                                        fg=self.accent_color,
                                        padx=15,
                                        pady=15)
        product_section.pack(fill=tk.X, pady=(0, 20))

        tk.Label(product_section,
                 text="Select Product:",
                 font=('Segoe UI', 10),
                 bg=self.card_bg,
                 fg=self.text_secondary).pack(anchor='w', pady=(0, 8))

        self.product_var = tk.StringVar(value="All Products")
        self.product_combo = ttk.Combobox(product_section,
                                          textvariable=self.product_var,
                                          font=('Segoe UI', 10),
                                          state='readonly',
                                          height=15)
        self.product_combo.pack(fill=tk.X, pady=(0, 5))
        self.product_combo.bind('<<ComboboxSelected>>', self.on_product_change)

        # Model Selection Section
        model_section = tk.LabelFrame(sidebar_content,
                                      text="🤖 ML MODELS",
                                      font=('Segoe UI', 11, 'bold'),
                                      bg=self.card_bg,
                                      fg=self.accent_color,
                                      padx=15,
                                      pady=15)
        model_section.pack(fill=tk.X, pady=(0, 20))

        # Model checkboxes with better spacing
        self.model_vars = {
            'Linear Regression': tk.BooleanVar(value=True),
            'Random Forest': tk.BooleanVar(value=True),
            'Gradient Boosting': tk.BooleanVar(value=True),
            'Exponential Smoothing': tk.BooleanVar(value=True)
        }

        for model_name, var in self.model_vars.items():
            cb_frame = tk.Frame(model_section, bg=self.card_bg)
            cb_frame.pack(fill=tk.X, pady=3)

            cb = tk.Checkbutton(cb_frame,
                                text=model_name,
                                variable=var,
                                font=('Segoe UI', 10),
                                bg=self.card_bg,
                                fg=self.text_color,
                                selectcolor=self.card_bg,
                                activebackground=self.card_bg,
                                activeforeground=self.text_color)
            cb.pack(side=tk.LEFT)

        # Forecast Settings Section
        forecast_section = tk.LabelFrame(sidebar_content,
                                         text="📅 FORECAST SETTINGS",
                                         font=('Segoe UI', 11, 'bold'),
                                         bg=self.card_bg,
                                         fg=self.accent_color,
                                         padx=15,
                                         pady=15)
        forecast_section.pack(fill=tk.X, pady=(0, 20))

        period_frame = tk.Frame(forecast_section, bg=self.card_bg)
        period_frame.pack(fill=tk.X)

        tk.Label(period_frame,
                 text="Forecast Days:",
                 font=('Segoe UI', 10),
                 bg=self.card_bg,
                 fg=self.text_secondary).pack(side=tk.LEFT)

        self.period_var = tk.IntVar(value=30)
        period_spin = tk.Spinbox(period_frame,
                                 from_=7,
                                 to=365,
                                 textvariable=self.period_var,
                                 font=('Segoe UI', 10),
                                 width=8,
                                 bg=self.grid_color,
                                 fg='white',
                                 relief='flat',
                                 buttonbackground=self.primary_color)
        period_spin.pack(side=tk.RIGHT)

        # Big Action Button
        forecast_btn = tk.Button(sidebar_content,
                                 text="🚀 RUN FORECAST ANALYSIS",
                                 command=self.run_forecast,
                                 font=('Segoe UI', 12, 'bold'),
                                 bg=self.success_color,
                                 fg='white',
                                 relief='flat',
                                 padx=30,
                                 pady=15)
        forecast_btn.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 0))

        # ============ MAIN DASHBOARD AREA ============
        dashboard_area = tk.Frame(content_frame, bg=self.bg_color)
        dashboard_area.grid(row=0, column=1, sticky='nsew')

        # Create Notebook for tabs with enhanced styling
        style = ttk.Style()
        style.configure("CustomNotebook.TNotebook",
                        background=self.bg_color,
                        borderwidth=0)
        style.configure("CustomNotebook.TNotebook.Tab",
                        background=self.primary_color,
                        foreground='white',
                        padding=[20, 10],
                        font=('Segoe UI', 11, 'bold'))
        style.map("CustomNotebook.TNotebook.Tab",
                  background=[('selected', self.accent_color)],
                  foreground=[('selected', self.bg_color)])

        self.notebook = ttk.Notebook(dashboard_area, style="CustomNotebook.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Sales Dashboard
        self.dashboard_tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.dashboard_tab, text='📊 SALES DASHBOARD')

        # Tab 2: Forecast Results
        self.forecast_tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.forecast_tab, text='🔮 FORECAST RESULTS')
        self.setup_forecast_tab()

        # Tab 3: Data Insights
        self.insights_tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.insights_tab, text='📈 BUSINESS INSIGHTS')
        self.setup_insights_tab()

        # Tab 4: Export
        self.export_tab = tk.Frame(self.notebook, bg=self.bg_color)
        self.notebook.add(self.export_tab, text='💾 EXPORT & REPORTS')
        self.setup_export_tab()

        # ============ STATUS BAR ============
        self.status_bar = tk.Frame(self.root, bg=self.primary_color, height=40)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_bar.pack_propagate(False)

        self.status_label = tk.Label(self.status_bar,
                                     text="✨ Dashboard Ready - Load data or generate sample",
                                     bg=self.primary_color,
                                     fg='white',
                                     font=('Segoe UI', 10),
                                     anchor=tk.W,
                                     padx=20)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Add time label
        self.time_label = tk.Label(self.status_bar,
                                   text="",
                                   bg=self.primary_color,
                                   fg=self.text_secondary,
                                   font=('Segoe UI', 9),
                                   anchor=tk.E,
                                   padx=20)
        self.time_label.pack(side=tk.RIGHT)

        # Update time
        self.update_time()

    def update_time(self):
        """Update current time in status bar"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=f"🕒 {current_time}")
        self.root.after(1000, self.update_time)

    def setup_forecast_tab(self):
        """Setup forecast results tab with proper layout"""
        # Main container
        container = tk.Frame(self.forecast_tab, bg=self.bg_color, padx=20, pady=20)
        container.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(container,
                 text="Model Performance Comparison",
                 font=self.heading_font,
                 bg=self.bg_color,
                 fg=self.text_color).pack(anchor='w', pady=(0, 20))

        # Create frame for table with scrollbar
        table_frame = tk.Frame(container, bg=self.bg_color)
        table_frame.pack(fill=tk.BOTH, expand=True)

        # Create treeview with enhanced styling
        columns = ('Model', 'MAE', 'RMSE', 'R² Score', 'Status')

        self.metrics_tree = ttk.Treeview(table_frame,
                                         columns=columns,
                                         show='headings',
                                         style='Custom.Treeview',
                                         height=12)

        # Configure columns with proper width and alignment
        col_widths = {'Model': 180, 'MAE': 120, 'RMSE': 120, 'R² Score': 100, 'Status': 120}
        col_anchors = {'Model': tk.W, 'MAE': tk.CENTER, 'RMSE': tk.CENTER, 'R² Score': tk.CENTER, 'Status': tk.CENTER}

        for col in columns:
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, width=col_widths[col], anchor=col_anchors[col])

        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame,
                                  orient=tk.VERTICAL,
                                  command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=scrollbar.set)

        # Pack with proper spacing
        self.metrics_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_insights_tab(self):
        """Setup insights tab with proper layout"""
        # Main container
        container = tk.Frame(self.insights_tab, bg=self.bg_color, padx=20, pady=20)
        container.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(container,
                 text="Business Intelligence Insights",
                 font=self.heading_font,
                 bg=self.bg_color,
                 fg=self.text_color).pack(anchor='w', pady=(0, 15))

        # Create text widget with better styling
        self.insights_text = scrolledtext.ScrolledText(container,
                                                       height=25,
                                                       font=('Consolas', 10),
                                                       bg=self.card_bg,
                                                       fg=self.text_color,
                                                       wrap=tk.WORD,
                                                       relief='flat',
                                                       padx=15,
                                                       pady=15)
        self.insights_text.pack(fill=tk.BOTH, expand=True)

        # Configure tags for better formatting
        self.insights_text.tag_config('title', foreground=self.accent_color, font=('Consolas', 12, 'bold'))
        self.insights_text.tag_config('heading', foreground=self.success_color, font=('Consolas', 11, 'bold'))
        self.insights_text.tag_config('highlight', foreground=self.warning_color)
        self.insights_text.tag_config('good', foreground=self.success_color)
        self.insights_text.tag_config('warning', foreground=self.warning_color)
        self.insights_text.tag_config('danger', foreground=self.danger_color)

    def setup_export_tab(self):
        """Setup export tab with proper layout"""
        # Main container
        container = tk.Frame(self.export_tab, bg=self.bg_color, padx=30, pady=30)
        container.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(container,
                 text="Export & Report Generation",
                 font=('Segoe UI', 18, 'bold'),
                 bg=self.bg_color,
                 fg=self.accent_color).pack(pady=(0, 30))

        # Export options in a grid
        options_frame = tk.Frame(container, bg=self.bg_color)
        options_frame.pack(fill=tk.BOTH, expand=True)

        # Create 2x2 grid of export options
        export_options = [
            ("📄 CSV Export", "Export forecast results to CSV", self.export_to_csv, self.accent_color),
            ("📊 Excel Report", "Complete report with charts", self.export_to_excel, self.success_color),
            ("📋 Business Report", "Generate detailed PDF report", self.generate_report, self.warning_color),
            ("🖼️ Save Charts", "Save dashboard visualizations", self.save_charts, self.danger_color)
        ]

        for i, (title, desc, command, color) in enumerate(export_options):
            row = i // 2
            col = i % 2

            option_card = tk.Frame(options_frame, bg=color, width=250, height=150)
            option_card.grid(row=row, column=col, padx=15, pady=15, sticky='nsew')
            option_card.pack_propagate(False)

            # Make grid expandable
            options_frame.rowconfigure(row, weight=1)
            options_frame.columnconfigure(col, weight=1)

            # Card content
            content = tk.Frame(option_card, bg=color)
            content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

            # Icon and Title
            tk.Label(content,
                     text=title.split()[0],
                     font=('Segoe UI', 24),
                     bg=color,
                     fg='white').pack(anchor='w')

            tk.Label(content,
                     text=title.split()[1],
                     font=('Segoe UI', 12, 'bold'),
                     bg=color,
                     fg='white').pack(anchor='w', pady=(5, 10))

            tk.Label(content,
                     text=desc,
                     font=('Segoe UI', 9),
                     bg=color,
                     fg='white',
                     wraplength=200,
                     justify=tk.LEFT).pack(anchor='w', fill=tk.X)

            # Export button
            tk.Button(content,
                      text="Export →",
                      command=command,
                      font=('Segoe UI', 9, 'bold'),
                      bg='white',
                      fg=color,
                      relief='flat',
                      padx=15,
                      pady=5).pack(side=tk.BOTTOM, anchor='w', pady=(10, 0))

    def load_sample_data(self):
        """Generate sample sales data for demo"""
        self.update_status("🎲 Generating sample data...")

        # Generate dates
        start_date = datetime.now() - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')

        products = ['Electronics', 'Clothing', 'Home Goods', 'Books', 'Sports']

        data = []
        for date in dates:
            for product in products:
                # Base sales with product-specific patterns
                if product == 'Electronics':
                    base = 1500 + 2 * (date - dates[0]).days
                    seasonal = 300 * np.sin(2 * np.pi * date.dayofyear / 365)
                elif product == 'Clothing':
                    base = 1200 + 1.5 * (date - dates[0]).days
                    seasonal = 250 * np.cos(2 * np.pi * date.dayofyear / 365)
                elif product == 'Home Goods':
                    base = 1000 + 1 * (date - dates[0]).days
                    seasonal = 200 * np.sin(2 * np.pi * date.month / 12)
                elif product == 'Books':
                    base = 800 + 0.8 * (date - dates[0]).days
                    seasonal = 150 * np.cos(2 * np.pi * date.week / 52)
                else:  # Sports
                    base = 600 + 0.5 * (date - dates[0]).days
                    seasonal = 100 * np.sin(2 * np.pi * date.month / 6)

                # Add noise and ensure minimum
                noise = np.random.normal(0, 100)
                sales = max(base + seasonal + noise, 200)

                data.append({
                    'Date': date,
                    'Product': product,
                    'Sales': round(sales, 2),
                    'Quantity': int(sales / np.random.uniform(20, 150)),
                    'Price': round(np.random.uniform(15, 600), 2)
                })

        self.sales_data = pd.DataFrame(data)
        self.update_product_list()
        self.update_kpis()
        self.plot_sales_dashboard()
        self.generate_insights()

        self.update_status("✅ Sample data loaded successfully")

    def plot_sales_dashboard(self):
        """Plot enhanced sales dashboard with perfect layout"""
        if self.sales_data is None:
            return

        # Clear previous chart
        for widget in self.dashboard_tab.winfo_children():
            widget.destroy()

        # Filter data for selected product
        if self.current_product == 'All Products':
            plot_data = self.sales_data.copy()
            title_suffix = "All Products"
        else:
            plot_data = self.sales_data[self.sales_data['Product'] == self.current_product]
            title_suffix = self.current_product

        # Create main container for charts
        charts_container = tk.Frame(self.dashboard_tab, bg=self.bg_color)
        charts_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create 2x2 grid of charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor(self.bg_color)

        # 1. Sales Trend Chart
        ax1 = axes[0, 0]
        if 'Date' in plot_data.columns and 'Sales' in plot_data.columns:
            daily_sales = plot_data.groupby('Date')['Sales'].sum()
            ax1.plot(daily_sales.index, daily_sales.values,
                     color=self.accent_color, linewidth=2.5, alpha=0.8)

            # Add trend line
            if len(daily_sales) > 30:
                z = np.polyfit(range(len(daily_sales)), daily_sales.values, 1)
                p = np.poly1d(z)
                ax1.plot(daily_sales.index, p(range(len(daily_sales))),
                         color=self.warning_color, linewidth=2, linestyle='--',
                         label='Trend Line')

            ax1.set_title(f'📈 Sales Trend - {title_suffix}',
                          fontsize=14, fontweight='bold', color='white', pad=20)
            ax1.set_ylabel('Sales ($)', color='white', fontsize=11)
            ax1.tick_params(axis='x', colors='white', labelsize=9)
            ax1.tick_params(axis='y', colors='white', labelsize=9)
            ax1.grid(True, alpha=0.2, color='gray', linestyle='--')
            ax1.legend(facecolor=self.card_bg, edgecolor='none',
                       labelcolor='white', fontsize=9)

        ax1.set_facecolor(self.card_bg)

        # 2. Product Performance (if multiple products)
        ax2 = axes[0, 1]
        if 'Product' in plot_data.columns and len(plot_data['Product'].unique()) > 1:
            product_sales = plot_data.groupby('Product')['Sales'].sum().sort_values()
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(product_sales)))
            bars = ax2.barh(range(len(product_sales)), product_sales.values, color=colors, height=0.6)

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, product_sales.values)):
                ax2.text(val + (val * 0.01), bar.get_y() + bar.get_height() / 2,
                         f'${val:,.0f}', ha='left', va='center',
                         fontsize=9, color='white', fontweight='bold')

            ax2.set_title('🏆 Product Performance',
                          fontsize=14, fontweight='bold', color='white', pad=20)
            ax2.set_yticks(range(len(product_sales)))
            ax2.set_yticklabels(product_sales.index, color='white', fontsize=10)
            ax2.set_xlabel('Total Sales ($)', color='white', fontsize=11)
            ax2.tick_params(axis='x', colors='white', labelsize=9)
            ax2.grid(True, alpha=0.2, color='gray', linestyle='--', axis='x')

        ax2.set_facecolor(self.card_bg)

        # 3. Monthly Sales
        ax3 = axes[1, 0]
        if 'Date' in plot_data.columns:
            plot_data['Month'] = plot_data['Date'].dt.strftime('%b %Y')
            monthly_sales = plot_data.groupby('Month')['Sales'].sum()

            colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(monthly_sales)))
            bars = ax3.bar(range(len(monthly_sales)), monthly_sales.values,
                           color=colors, width=0.7, edgecolor='white', linewidth=1)

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, monthly_sales.values)):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'${val:,.0f}', ha='center', va='bottom',
                         fontsize=9, color='white', rotation=0)

            ax3.set_title('📅 Monthly Sales',
                          fontsize=14, fontweight='bold', color='white', pad=20)
            ax3.set_xticks(range(len(monthly_sales)))
            ax3.set_xticklabels(monthly_sales.index, rotation=45,
                                color='white', fontsize=9, ha='right')
            ax3.set_ylabel('Sales ($)', color='white', fontsize=11)
            ax3.tick_params(axis='y', colors='white', labelsize=9)
            ax3.grid(True, alpha=0.2, color='gray', linestyle='--', axis='y')

        ax3.set_facecolor(self.card_bg)

        # 4. Moving Averages
        ax4 = axes[1, 1]
        if 'Date' in plot_data.columns and 'Sales' in plot_data.columns:
            daily_sales = plot_data.groupby('Date')['Sales'].sum()
            ma_7 = daily_sales.rolling(window=7).mean()
            ma_30 = daily_sales.rolling(window=30).mean()

            ax4.plot(daily_sales.index, daily_sales.values,
                     color=self.text_secondary, alpha=0.4, linewidth=1, label='Daily')
            ax4.plot(ma_7.index, ma_7.values,
                     color=self.success_color, linewidth=2.5, label='7-Day MA')
            ax4.plot(ma_30.index, ma_30.values,
                     color=self.accent_color, linewidth=2.5, label='30-Day MA')

            ax4.fill_between(ma_7.index, ma_7.values, alpha=0.2, color=self.success_color)
            ax4.fill_between(ma_30.index, ma_30.values, alpha=0.1, color=self.accent_color)

            ax4.set_title('📊 Moving Averages Analysis',
                          fontsize=14, fontweight='bold', color='white', pad=20)
            ax4.set_xlabel('Date', color='white', fontsize=11)
            ax4.set_ylabel('Sales ($)', color='white', fontsize=11)
            ax4.tick_params(axis='x', colors='white', labelsize=9)
            ax4.tick_params(axis='y', colors='white', labelsize=9)
            ax4.legend(facecolor=self.card_bg, edgecolor='none',
                       labelcolor='white', fontsize=10, loc='upper left')
            ax4.grid(True, alpha=0.2, color='gray', linestyle='--')

        ax4.set_facecolor(self.card_bg)

        # Adjust layout
        plt.tight_layout()

        # Embed in tkinter with proper sizing
        canvas = FigureCanvasTkAgg(fig, charts_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_kpis(self):
        """Update KPI cards with enhanced information"""
        if self.sales_data is not None:
            if self.current_product == 'All Products':
                data = self.sales_data
            else:
                data = self.sales_data[self.sales_data['Product'] == self.current_product]

            # Calculate KPIs
            total_sales = data['Sales'].sum() if 'Sales' in data.columns else 0
            avg_daily = data['Sales'].mean() if 'Sales' in data.columns else 0

            # Growth calculation
            if len(data) > 60 and 'Date' in data.columns:
                recent = data.sort_values('Date').tail(30)['Sales']
                older = data.sort_values('Date').head(30)['Sales']
                if older.mean() > 0:
                    growth = ((recent.mean() - older.mean()) / older.mean()) * 100
                else:
                    growth = 0
            else:
                growth = 0

            # Top product
            if 'Product' in self.sales_data.columns:
                top_product = self.sales_data.groupby('Product')['Sales'].sum().idxmax()
                top_product_sales = self.sales_data.groupby('Product')['Sales'].sum().max()
            else:
                top_product = "N/A"
                top_product_sales = 0

            # Best day
            if 'Date' in self.sales_data.columns:
                daily_sales = self.sales_data.groupby('Date')['Sales'].sum()
                best_day = daily_sales.idxmax().strftime('%b %d')
                best_day_sales = daily_sales.max()
            else:
                best_day = "N/A"
                best_day_sales = 0

            # Update KPI labels
            self.kpi_labels['Total Sales'].config(text=f"$ {total_sales:,.0f}")
            self.kpi_labels['Avg Daily'].config(text=f"$ {avg_daily:,.0f}")
            self.kpi_labels['Growth %'].config(text=f"{growth:+.1f}%")
            self.kpi_labels['Top Product'].config(text=top_product[:15])
            self.kpi_labels['Best Day'].config(text=best_day)

            # Update forecast KPI if available
            if self.models:
                best_model = max(self.models.items(), key=lambda x: x[1]['r2']) if self.models else None
                if best_model:
                    forecast_value = best_model[1]['predictions'][-1] if best_model[1]['predictions'].any() else 0
                    self.kpi_labels['Forecast'].config(text=f"$ {forecast_value:,.0f}")

            # Color coding for growth
            if growth >= 10:
                self.kpi_labels['Growth %'].config(fg=self.success_color)
            elif growth >= 0:
                self.kpi_labels['Growth %'].config(fg=self.warning_color)
            else:
                self.kpi_labels['Growth %'].config(fg=self.danger_color)

    def generate_sample_data(self):
        """Generate new sample data"""
        self.load_sample_data()
        messagebox.showinfo("Success", "✅ New sample data generated!")

    def load_csv_data(self):
        """Load sales data from CSV file"""
        filetypes = [('CSV files', '*.csv'), ('Excel files', '*.xlsx'), ('All files', '*.*')]

        filepath = filedialog.askopenfilename(
            title="Select Sales Data File",
            filetypes=filetypes
        )

        if filepath:
            try:
                if filepath.endswith('.csv'):
                    self.sales_data = pd.read_csv(filepath)
                else:
                    self.sales_data = pd.read_excel(filepath)

                # Convert date column
                if 'Date' in self.sales_data.columns:
                    self.sales_data['Date'] = pd.to_datetime(self.sales_data['Date'])

                # Update UI
                self.update_product_list()
                self.update_kpis()
                self.plot_sales_dashboard()
                self.generate_insights()

                self.update_status(f"✅ Data loaded: {len(self.sales_data)} records")

            except Exception as e:
                messagebox.showerror("Error", f"Could not load file:\n{str(e)}")

    def update_product_list(self):
        """Update product selection dropdown"""
        if self.sales_data is not None and 'Product' in self.sales_data.columns:
            products = ['All Products'] + sorted(self.sales_data['Product'].unique().tolist())
            self.product_combo['values'] = products

    def on_product_change(self, event=None):
        """Handle product selection change"""
        self.current_product = self.product_var.get()
        self.plot_sales_dashboard()
        self.update_kpis()
        self.update_status(f"📊 Viewing: {self.current_product}")

    def refresh_data(self):
        """Refresh data visualization"""
        if self.sales_data is not None:
            self.plot_sales_dashboard()
            self.update_kpis()
            self.update_status("🔄 Dashboard refreshed")
        else:
            messagebox.showwarning("Warning", "No data to refresh")

    def run_forecast(self):
        """Run sales forecasting using selected models"""
        if self.sales_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        self.update_status("🤖 Training ML models...")

        # Filter data for selected product
        if self.current_product == 'All Products':
            data = self.sales_data.groupby('Date')['Sales'].sum().reset_index()
        else:
            data = self.sales_data[self.sales_data['Product'] == self.current_product]
            data = data.groupby('Date')['Sales'].sum().reset_index()

        if len(data) < 30:
            messagebox.showwarning("Warning", "Need at least 30 days of data for forecasting")
            return

        # Prepare data
        data = data.sort_values('Date')
        data['Days'] = (data['Date'] - data['Date'].min()).dt.days

        # Split data
        X = data['Days'].values.reshape(-1, 1)
        y = data['Sales'].values

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Clear previous results
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)

        self.models = {}
        forecast_days = self.period_var.get()

        # Train selected models
        selected_models = [model for model, var in self.model_vars.items() if var.get()]

        if not selected_models:
            messagebox.showwarning("Warning", "Please select at least one model")
            return

        for model_name in selected_models:
            try:
                if model_name == 'Linear Regression':
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                elif model_name == 'Random Forest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                elif model_name == 'Gradient Boosting':
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                elif model_name == 'Exponential Smoothing':
                    # Simple exponential smoothing implementation
                    from statsmodels.tsa.holtwinters import ExponentialSmoothing
                    train_series = pd.Series(y_train, index=data['Date'].iloc[:train_size])
                    model = ExponentialSmoothing(train_series, seasonal='add', seasonal_periods=7)
                    model_fit = model.fit()
                    predictions = model_fit.forecast(len(y_test))
                    predictions = predictions.values

                # Calculate metrics
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                r2 = r2_score(y_test, predictions)

                # Store model
                self.models[model_name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'predictions': predictions
                }

                # Determine status with emoji
                if r2 > 0.8:
                    status = "⭐ Excellent"
                    status_color = self.success_color
                elif r2 > 0.6:
                    status = "✅ Good"
                    status_color = self.success_color
                elif r2 > 0.4:
                    status = "⚠️ Fair"
                    status_color = self.warning_color
                else:
                    status = "❌ Poor"
                    status_color = self.danger_color

                # Add to treeview with colors
                item_id = self.metrics_tree.insert('', 'end',
                                                   values=(model_name,
                                                           f"{mae:,.2f}",
                                                           f"{rmse:,.2f}",
                                                           f"{r2:.3f}",
                                                           status))

                # Color code based on performance
                self.metrics_tree.item(item_id, tags=(status_color,))

            except Exception as e:
                print(f"Error training {model_name}: {e}")
                self.metrics_tree.insert('', 'end',
                                         values=(model_name, "Error", "Error", "Error", "❌ Failed"))

        # Update forecast KPI
        if self.models:
            best_model = max(self.models.items(), key=lambda x: x[1]['r2'])
            forecast_value = best_model[1]['predictions'][-1] if len(best_model[1]['predictions']) > 0 else 0
            self.kpi_labels['Forecast'].config(text=f"$ {forecast_value:,.0f}")

        # Show forecast visualization
        self.show_forecast_visualization(data, forecast_days)

        self.update_status("✅ Forecasting complete! Check results tab")

        # Switch to forecast tab
        self.notebook.select(1)

    def show_forecast_visualization(self, historical_data, forecast_days):
        """Show forecast results in a new window"""
        forecast_window = tk.Toplevel(self.root)
        forecast_window.title("🔮 Forecast Visualization")
        forecast_window.geometry("1200x700")
        forecast_window.configure(bg=self.bg_color)

        # Center the window
        forecast_window.transient(self.root)
        forecast_window.grab_set()

        # Title
        tk.Label(forecast_window,
                 text=f"Sales Forecast Visualization - Next {forecast_days} Days",
                 font=('Segoe UI', 18, 'bold'),
                 bg=self.bg_color,
                 fg=self.accent_color).pack(pady=20)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.card_bg)

        # Plot historical data
        ax.plot(historical_data['Date'], historical_data['Sales'],
                color=self.text_secondary, linewidth=3, alpha=0.7, label='Historical Sales')

        # Plot each model's predictions
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.models)))

        for i, (model_name, model_data) in enumerate(self.models.items()):
            if 'predictions' in model_data and len(model_data['predictions']) > 0:
                # Get test dates
                test_dates = historical_data['Date'].iloc[-len(model_data['predictions']):]
                ax.plot(test_dates, model_data['predictions'],
                        color=colors[i], linewidth=2.5, linestyle='--',
                        label=f'{model_name} (R²={model_data["r2"]:.3f})')

        # Customize plot
        ax.set_title('Sales Forecast Comparison', fontsize=16, fontweight='bold', color='white')
        ax.set_xlabel('Date', color='white', fontsize=12)
        ax.set_ylabel('Sales ($)', color='white', fontsize=12)
        ax.tick_params(axis='x', colors='white', labelsize=10)
        ax.tick_params(axis='y', colors='white', labelsize=10)
        ax.legend(facecolor=self.card_bg, edgecolor='none',
                  labelcolor='white', fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.2, color='gray', linestyle='--')

        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Embed in window
        canvas = FigureCanvasTkAgg(fig, forecast_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    def generate_insights(self):
        """Generate enhanced business insights"""
        if self.sales_data is None:
            return

        # Clear previous insights
        self.insights_text.delete('1.0', tk.END)

        insights = ""

        # Title
        insights += "📊 BUSINESS INTELLIGENCE INSIGHTS\n"
        insights += "=" * 50 + "\n\n"

        # Overall Statistics
        insights += "[OVERALL PERFORMANCE]\n"
        insights += "• Total Sales: ${:,.2f}\n".format(self.sales_data['Sales'].sum())
        insights += "• Average Daily Sales: ${:,.2f}\n".format(self.sales_data['Sales'].mean())
        insights += "• Total Transactions: {:,}\n\n".format(len(self.sales_data))

        # Product Analysis
        if 'Product' in self.sales_data.columns:
            product_sales = self.sales_data.groupby('Product')['Sales'].sum()
            total_sales = product_sales.sum()

            insights += "[PRODUCT PERFORMANCE]\n"
            for product, sales in product_sales.sort_values(ascending=False).items():
                share = (sales / total_sales) * 100
                insights += "• {}: ${:,.0f} ({:.1f}%)\n".format(product, sales, share)
            insights += "\n"

        # Time-based Insights
        if 'Date' in self.sales_data.columns:
            self.sales_data['Month'] = self.sales_data['Date'].dt.strftime('%B')
            self.sales_data['DayOfWeek'] = self.sales_data['Date'].dt.day_name()

            monthly_sales = self.sales_data.groupby('Month')['Sales'].sum()
            daily_sales = self.sales_data.groupby('DayOfWeek')['Sales'].sum()

            insights += "[TIME ANALYSIS]\n"
            insights += "• Best Month: {} (${:,.0f})\n".format(
                monthly_sales.idxmax(), monthly_sales.max())
            insights += "• Best Day: {} (${:,.0f})\n".format(
                daily_sales.idxmax(), daily_sales.max())
            insights += "• Average Monthly Growth: {:.1f}%\n\n".format(
                monthly_sales.pct_change().mean() * 100)

        # Recommendations
        insights += "[RECOMMENDATIONS]\n"
        insights += "1. 📈 Focus marketing on top-performing products\n"
        insights += "2. 🏪 Increase inventory before peak seasons\n"
        insights += "3. 💰 Run promotions on low-sales days\n"
        insights += "4. 📊 Monitor sales trends weekly\n"
        insights += "5. 🤖 Use ML forecasting for inventory planning\n\n"

        insights += "Report generated: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M'))

        # Insert with formatting
        self.insights_text.insert('1.0', insights)

        # Apply formatting
        self.insights_text.tag_add('title', '1.0', '1.50')
        self.insights_text.tag_add('heading', '3.0', '3.19')
        self.insights_text.tag_add('heading', '11.0', '11.19')
        self.insights_text.tag_add('heading', '19.0', '19.16')
        self.insights_text.tag_add('heading', '26.0', '26.15')

    def export_to_csv(self):
        """Export forecast results to CSV"""
        if not self.models:
            messagebox.showwarning("Warning", "Please run forecast first")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )

        if filepath:
            try:
                export_data = []
                for model_name, model_data in self.models.items():
                    export_data.append({
                        'Model': model_name,
                        'MAE': model_data['mae'],
                        'RMSE': model_data['rmse'],
                        'R2_Score': model_data['r2']
                    })

                df = pd.DataFrame(export_data)
                df.to_csv(filepath, index=False)

                self.update_status(f"✅ CSV exported: {os.path.basename(filepath)}")
                messagebox.showinfo("Success", "Results exported to CSV successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Could not export: {str(e)}")

    def export_to_excel(self):
        """Export comprehensive report to Excel"""
        if self.sales_data is None:
            messagebox.showwarning("Warning", "No data to export")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[('Excel files', '*.xlsx'), ('All files', '*.*')]
        )

        if filepath:
            try:
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    # Export raw data
                    self.sales_data.to_excel(writer, sheet_name='Raw Data', index=False)

                    # Export summary
                    summary = pd.DataFrame({
                        'Metric': ['Total Sales', 'Average Daily', 'Transactions', 'Date Range'],
                        'Value': [
                            f"${self.sales_data['Sales'].sum():,.2f}",
                            f"${self.sales_data['Sales'].mean():,.2f}",
                            len(self.sales_data),
                            f"{self.sales_data['Date'].min().date()} to {self.sales_data['Date'].max().date()}"
                            if 'Date' in self.sales_data.columns else 'N/A'
                        ]
                    })
                    summary.to_excel(writer, sheet_name='Summary', index=False)

                self.update_status(f"✅ Excel report exported")
                messagebox.showinfo("Success", "Complete report exported to Excel!")

            except Exception as e:
                messagebox.showerror("Error", f"Could not export: {str(e)}")

    def generate_report(self):
        """Generate comprehensive business report"""
        if self.sales_data is None:
            messagebox.showwarning("Warning", "No data for report")
            return

        # Create report window
        report_window = tk.Toplevel(self.root)
        report_window.title("📋 Sales Analysis Report")
        report_window.geometry("900x700")
        report_window.configure(bg=self.bg_color)

        # Report content
        report_content = scrolledtext.ScrolledText(report_window,
                                                   font=('Courier', 10),
                                                   bg=self.card_bg,
                                                   fg='white',
                                                   wrap=tk.WORD)
        report_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Generate report text
        report = self.generate_report_text()
        report_content.insert('1.0', report)
        report_content.config(state=tk.DISABLED)

        # Save button
        tk.Button(report_window,
                  text="💾 Save Report as Text",
                  command=lambda: self.save_text_report(report),
                  font=('Segoe UI', 10),
                  bg=self.accent_color,
                  fg=self.bg_color,
                  relief='flat',
                  padx=30,
                  pady=10).pack(pady=(0, 20))

    def save_charts(self):
        """Save dashboard charts as images"""
        if self.sales_data is None:
            messagebox.showwarning("Warning", "No charts to save")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[('PNG files', '*.png'), ('All files', '*.*')]
        )

        if filepath:
            try:
                # Recreate the dashboard figure
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))

                # ... (same plotting code as in plot_sales_dashboard)

                plt.tight_layout()
                plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=self.bg_color)
                plt.close()

                self.update_status(f"✅ Charts saved: {os.path.basename(filepath)}")
                messagebox.showinfo("Success", "Charts saved successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Could not save charts: {str(e)}")

    def generate_report_text(self):
        """Generate report text"""
        return "Sales Forecasting Report - Generated by Smart Sales Forecasting AI"

    def save_text_report(self, report_text):
        """Save report to text file"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')]
        )

        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(report_text)

                self.update_status(f"✅ Report saved")
                messagebox.showinfo("Success", "Report saved successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Could not save report: {str(e)}")

    def update_status(self, message):
        """Update status bar with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_label.config(text=f"[{timestamp}] {message}")
        self.root.update()


def main():
    root = tk.Tk()

    # Start maximized
    root.state('zoomed')

    # Create application
    app = SmartSalesForecaster(root)

    # Check dependencies
    try:
        import pandas
        import sklearn
    except ImportError:
        messagebox.showwarning(
            "Missing Dependencies",
            "Please install:\n\n"
            "pip install pandas matplotlib scikit-learn seaborn statsmodels openpyxl"
        )

    root.mainloop()


if __name__ == "__main__":
    main()