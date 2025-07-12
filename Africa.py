import wbgapi as wb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class WestAfricaDebtAnalyzer:
    def __init__(self):
        # West African countries
        self.countries = {
            'BEN': 'Benin', 'BFA': 'Burkina Faso', 'CPV': 'Cape Verde',
            'CIV': 'Cote d\'Ivoire', 'GMB': 'Gambia', 'GHA': 'Ghana',
            'GIN': 'Guinea', 'GNB': 'Guinea-Bissau', 'LBR': 'Liberia',
            'MLI': 'Mali', 'MRT': 'Mauritania', 'NER': 'Niger',
            'NGA': 'Nigeria', 'SEN': 'Senegal', 'SLE': 'Sierra Leone',
            'TGO': 'Togo'
        }
        
        # Comprehensive debt and economic indicators
        self.indicators = {
            'DT.DOD.DECT.GN.ZS': 'External Debt (% of GNI)',
            'DT.TDS.DECT.EX.ZS': 'Debt Service (% of exports)',
            'DT.DOD.DECT.CD': 'Total External Debt (USD)',
            'NY.GDP.MKTP.CD': 'GDP (USD)',
            'NY.GDP.MKTP.KD.ZG': 'GDP Growth (%)',
            'NY.GNP.MKTP.CD': 'GNI (USD)',
            'BX.KLT.DINV.WD.GD.ZS': 'FDI (% of GDP)',
            'NE.EXP.GNFS.ZS': 'Exports (% of GDP)',
            'GC.REV.XGRT.GD.ZS': 'Government Revenue (% of GDP)',
            'SL.UEM.TOTL.ZS': 'Unemployment Rate (%)'
        }
        
        self.data = None
        
    def fetch_data(self, start_year=2010, end_year=2023):
        """Fetch comprehensive debt and economic data"""
        print("Fetching comprehensive debt and economic data...")
        
        all_data = []
        
        for code, name in self.indicators.items():
            print(f"Fetching {name}...")
            try:
                df = wb.data.DataFrame(
                    code,
                    economy=list(self.countries.keys()),
                    time=range(start_year, end_year + 1),
                    columns='series'
                ).reset_index()
                
                df.columns = ['Country', 'Year', name]
                df['Country_Name'] = df['Country'].map(self.countries)
                
                if len(all_data) == 0:
                    all_data = df
                else:
                    all_data = all_data.merge(df, on=['Country', 'Year', 'Country_Name'], how='outer')
                    
            except Exception as e:
                print(f"Error fetching {name}: {e}")
                continue
        
        # Calculate derived metrics
        all_data['Debt_to_GDP'] = (all_data['Total External Debt (USD)'] / all_data['GDP (USD)']) * 100
        all_data['Debt_per_capita'] = all_data['Total External Debt (USD)'] / (all_data['GDP (USD)'] / 1000)  # Approximate
        
        self.data = all_data
        print("Data fetching completed!")
        return all_data
    
    def categorize_debt_risk(self, df):
        """Automatically classify countries into debt risk levels"""
        def get_risk_level(row):
            debt_gni = row['External Debt (% of GNI)']
            debt_service = row['Debt Service (% of exports)']
            debt_gdp = row['Debt_to_GDP']
            
            # Handle NaN values
            if pd.isna(debt_gni) and pd.isna(debt_service) and pd.isna(debt_gdp):
                return 'No Data'
            
            # Risk scoring system
            risk_score = 0
            
            if not pd.isna(debt_gni):
                if debt_gni > 60: risk_score += 3
                elif debt_gni > 40: risk_score += 2
                elif debt_gni > 25: risk_score += 1
            
            if not pd.isna(debt_service):
                if debt_service > 25: risk_score += 3
                elif debt_service > 15: risk_score += 2
                elif debt_service > 10: risk_score += 1
            
            if not pd.isna(debt_gdp):
                if debt_gdp > 70: risk_score += 2
                elif debt_gdp > 50: risk_score += 1
            
            # Classify based on total risk score
            if risk_score >= 6: return 'Critical Risk'
            elif risk_score >= 4: return 'High Risk'
            elif risk_score >= 2: return 'Medium Risk'
            else: return 'Low Risk'
        
        df['Risk_Level'] = df.apply(get_risk_level, axis=1)
        return df
    
    def calculate_sustainability_metrics(self, df):
        """Calculate debt sustainability metrics"""
        df['Debt_Sustainability_Index'] = (
            (df['External Debt (% of GNI)'] / 60) * 0.4 +
            (df['Debt Service (% of exports)'] / 25) * 0.3 +
            (df['Debt_to_GDP'] / 70) * 0.3
        )
        
        df['Debt_Trend'] = df.groupby('Country')['External Debt (% of GNI)'].pct_change() * 100
        df['GDP_Debt_Ratio'] = df['GDP Growth (%)'] / df['External Debt (% of GNI)']
        
        return df
    
    def create_regional_benchmarks(self, df):
        """Create regional averages and benchmarks"""
        regional_stats = df.groupby('Year').agg({
            'External Debt (% of GNI)': ['mean', 'median', 'std'],
            'Debt Service (% of exports)': ['mean', 'median', 'std'],
            'Debt_to_GDP': ['mean', 'median', 'std'],
            'GDP Growth (%)': ['mean', 'median', 'std'],
            'Debt_Sustainability_Index': ['mean', 'median', 'std']
        }).round(2)
        
        # Flatten column names
        regional_stats.columns = ['_'.join(col).strip() for col in regional_stats.columns]
        regional_stats = regional_stats.reset_index()
        
        return regional_stats
    
    def create_debt_gdp_trends(self):
        """Create debt-to-GDP trends over time"""
        fig = px.line(
            self.data.dropna(subset=['Debt_to_GDP']),
            x='Year',
            y='Debt_to_GDP',
            color='Country_Name',
            title='Debt-to-GDP Trends Over Time (2010-2023)',
            labels={'Debt_to_GDP': 'Debt-to-GDP Ratio (%)'}
        )
        
        # Add benchmark lines
        fig.add_hline(y=60, line_dash="dash", line_color="red", 
                     annotation_text="Critical Threshold (60%)")
        fig.add_hline(y=40, line_dash="dash", line_color="orange", 
                     annotation_text="Warning Threshold (40%)")
        
        fig.update_layout(height=500, template='plotly_white')
        return fig
    
    def create_risk_heatmap(self):
        """Create current debt risk heatmap"""
        latest_year = self.data['Year'].max()
        latest_data = self.data[self.data['Year'] == latest_year]
        latest_data = self.categorize_debt_risk(latest_data)
        
        # Create risk matrix
        risk_matrix = latest_data.pivot_table(
            values='Debt_Sustainability_Index',
            index='Country_Name',
            columns='Risk_Level',
            aggfunc='mean'
        ).fillna(0)
        
        # Create heatmap using country data
        fig = px.imshow(
            latest_data[['External Debt (% of GNI)', 'Debt Service (% of exports)', 'Debt_to_GDP']].T,
            y=['External Debt (% of GNI)', 'Debt Service (% of exports)', 'Debt-to-GDP'],
            x=latest_data['Country_Name'],
            color_continuous_scale='RdYlGn_r',
            title='Current Debt Risk Heatmap by Country',
            aspect='auto'
        )
        
        fig.update_layout(height=400, template='plotly_white')
        return fig
    
    def create_sustainability_dashboard(self):
        """Create comprehensive debt sustainability dashboard"""
        latest_data = self.data[self.data['Year'] == self.data['Year'].max()]
        latest_data = self.categorize_debt_risk(latest_data)
        latest_data = self.calculate_sustainability_metrics(latest_data)
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Risk Level Distribution',
                'Debt Sustainability Index',
                'Debt Service Burden',
                'Economic Performance vs Debt',
                'Regional Risk Comparison',
                'Debt Trend Analysis'
            ]
        )
        
        # 1. Risk Level Distribution
        risk_counts = latest_data['Risk_Level'].value_counts()
        colors = {'Critical Risk': 'darkred', 'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'}
        
        fig.add_trace(
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                marker_color=[colors.get(risk, 'gray') for risk in risk_counts.index],
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Debt Sustainability Index
        sustainability_data = latest_data.dropna(subset=['Debt_Sustainability_Index'])
        fig.add_trace(
            go.Scatter(
                x=sustainability_data['Country_Name'],
                y=sustainability_data['Debt_Sustainability_Index'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=sustainability_data['Debt_Sustainability_Index'],
                    colorscale='RdYlGn_r',
                    showscale=False
                ),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Debt Service Burden
        debt_service_data = latest_data.dropna(subset=['Debt Service (% of exports)'])
        fig.add_trace(
            go.Bar(
                x=debt_service_data['Country_Name'],
                y=debt_service_data['Debt Service (% of exports)'],
                marker_color='lightcoral',
                showlegend=False
            ),
            row=1, col=3
        )
        
        # 4. Economic Performance vs Debt
        perf_data = latest_data.dropna(subset=['GDP Growth (%)', 'External Debt (% of GNI)'])
        fig.add_trace(
            go.Scatter(
                x=perf_data['GDP Growth (%)'],
                y=perf_data['External Debt (% of GNI)'],
                mode='markers+text',
                text=perf_data['Country_Name'],
                textposition='top center',
                marker=dict(size=10),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 5. Regional Risk Comparison
        regional_data = self.create_regional_benchmarks(latest_data)
        fig.add_trace(
            go.Bar(
                x=['Mean', 'Median'],
                y=[regional_data['External Debt (% of GNI)_mean'].iloc[-1], 
                   regional_data['External Debt (% of GNI)_median'].iloc[-1]],
                marker_color=['skyblue', 'lightgreen'],
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 6. Debt Trend Analysis
        trend_data = self.data.groupby('Year')['External Debt (% of GNI)'].mean()
        fig.add_trace(
            go.Scatter(
                x=trend_data.index,
                y=trend_data.values,
                mode='lines+markers',
                line=dict(color='red', width=3),
                showlegend=False
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title='Comprehensive Debt Sustainability Dashboard',
            height=800,
            template='plotly_white'
        )
        
        return fig
    
    def create_regional_comparison(self):
        """Create regional comparison charts"""
        regional_stats = self.create_regional_benchmarks(self.data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Regional Average Debt Trends',
                'Debt Service Burden Trends',
                'GDP Growth vs Debt Correlation',
                'Debt Sustainability Evolution'
            ]
        )
        
        # Regional debt trends
        fig.add_trace(
            go.Scatter(
                x=regional_stats['Year'],
                y=regional_stats['External Debt (% of GNI)_mean'],
                mode='lines+markers',
                name='Mean Debt',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=regional_stats['Year'],
                y=regional_stats['External Debt (% of GNI)_median'],
                mode='lines+markers',
                name='Median Debt',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Debt service trends
        fig.add_trace(
            go.Scatter(
                x=regional_stats['Year'],
                y=regional_stats['Debt Service (% of exports)_mean'],
                mode='lines+markers',
                name='Mean Service',
                line=dict(color='orange'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # GDP Growth correlation
        correlation_data = self.data.dropna(subset=['GDP Growth (%)', 'External Debt (% of GNI)'])
        fig.add_trace(
            go.Scatter(
                x=correlation_data['GDP Growth (%)'],
                y=correlation_data['External Debt (% of GNI)'],
                mode='markers',
                marker=dict(
                    color=correlation_data['Year'].astype('category').cat.codes,
                    colorscale='viridis',
                    showscale=False
                ),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Sustainability evolution
        fig.add_trace(
            go.Scatter(
                x=regional_stats['Year'],
                y=regional_stats['Debt_Sustainability_Index_mean'],
                mode='lines+markers',
                name='Sustainability Index',
                line=dict(color='purple'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Regional Debt Analysis Comparison',
            height=700,
            template='plotly_white'
        )
        
        return fig
    
    def create_economic_correlation_analysis(self):
        """Create economic performance vs debt correlation analysis"""
        corr_data = self.data.dropna(subset=['GDP Growth (%)', 'External Debt (% of GNI)', 'FDI (% of GDP)'])
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                'GDP Growth vs Debt Burden',
                'FDI vs Debt Levels',
                'Export Performance vs Debt Service'
            ]
        )
        
        # GDP Growth vs Debt
        fig.add_trace(
            go.Scatter(
                x=corr_data['GDP Growth (%)'],
                y=corr_data['External Debt (% of GNI)'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=corr_data['Year'].astype('category').cat.codes,
                    colorscale='viridis',
                    showscale=False
                ),
                text=corr_data['Country_Name'],
                showlegend=False
            ),
            row=1, col=1
        )
        
        # FDI vs Debt
        fdi_data = corr_data.dropna(subset=['FDI (% of GDP)'])
        fig.add_trace(
            go.Scatter(
                x=fdi_data['FDI (% of GDP)'],
                y=fdi_data['External Debt (% of GNI)'],
                mode='markers',
                marker=dict(size=8, color='red'),
                text=fdi_data['Country_Name'],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Export Performance vs Debt Service
        export_data = corr_data.dropna(subset=['Exports (% of GDP)', 'Debt Service (% of exports)'])
        fig.add_trace(
            go.Scatter(
                x=export_data['Exports (% of GDP)'],
                y=export_data['Debt Service (% of exports)'],
                mode='markers',
                marker=dict(size=8, color='green'),
                text=export_data['Country_Name'],
                showlegend=False
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            title='Economic Performance vs Debt Correlation Analysis',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def run_comprehensive_analysis(self):
        """Run the complete comprehensive debt analysis"""
        print("Starting Comprehensive West Africa Debt Analysis...")
        
        # Fetch data
        self.fetch_data()
        
        # Process data
        self.data = self.categorize_debt_risk(self.data)
        self.data = self.calculate_sustainability_metrics(self.data)
        
        # Create all visualizations
        print("Creating visualizations...")
        
        # 1. Debt-to-GDP Trends
        debt_gdp_fig = self.create_debt_gdp_trends()
        
        # 2. Risk Heatmap
        heatmap_fig = self.create_risk_heatmap()
        
        # 3. Sustainability Dashboard
        dashboard_fig = self.create_sustainability_dashboard()
        
        # 4. Regional Comparison
        regional_fig = self.create_regional_comparison()
        
        # 5. Economic Correlation
        correlation_fig = self.create_economic_correlation_analysis()
        
        print("Analysis completed!")
        
        return {
            'debt_gdp_trends': debt_gdp_fig,
            'risk_heatmap': heatmap_fig,
            'sustainability_dashboard': dashboard_fig,
            'regional_comparison': regional_fig,
            'economic_correlation': correlation_fig,
            'processed_data': self.data
        }

# Run the comprehensive analysis
if __name__ == "__main__":
    analyzer = WestAfricaDebtAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    # Display all visualizations
    print("Displaying visualizations...")
    
    results['debt_gdp_trends'].show()
    results['risk_heatmap'].show()
    results['sustainability_dashboard'].show()
    results['regional_comparison'].show()
    results['economic_correlation'].show()
    
    # Save all visualizations
    print("Saving visualizations...")
    results['debt_gdp_trends'].write_html("debt_gdp_trends.html")
    results['risk_heatmap'].write_html("debt_risk_heatmap.html")
    results['sustainability_dashboard'].write_html("debt_sustainability_dashboard.html")
    results['regional_comparison'].write_html("regional_comparison.html")
    results['economic_correlation'].write_html("economic_correlation.html")
    
    # Save processed data
    results['processed_data'].to_csv("comprehensive_debt_analysis.csv", index=False)
    
    print("Files saved successfully!")