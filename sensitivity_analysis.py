import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math

class BuildingEnergySimulation:
    def __init__(self, tes_volume_m3, heat_pump_cop, pv_panel_capacity_kwp, panel_count, 
                 hp_min_electrical_input_kw=5, acdc_loss_factor=0.03, tes_capacity_multiplier=1.0):
        self.heat_pump_cop = heat_pump_cop
        self.hp_min_electrical_input_kw = hp_min_electrical_input_kw
        self.acdc_loss_factor = acdc_loss_factor
        
        # PV configuration - simplified to single south-facing array
        self.pv_panel_capacity_kwp = pv_panel_capacity_kwp
        self.panel_count = panel_count
        self.total_pv_capacity_kwp = panel_count * pv_panel_capacity_kwp
        
        # TES parameters with capacity multiplier for sensitivity analysis
        self.tes_volume_m3 = tes_volume_m3
        self.tes_minimum_kwh = 1272.0
        self.tes_physical_max_kwh = 2090.0 * tes_capacity_multiplier
        self.tes_capacity = self.tes_physical_max_kwh - self.tes_minimum_kwh
        self.tes_energy_kwh = 0
        self.tes_loss_rate_per_hour = 0.005
        
        # Location parameters for Stockholm
        self.latitude_deg = 59.33
        
        # Summer tracking
        self.summer_total_demand_kwh = 0
        self.summer_grid_to_hp_thermal_kwh = 0
        self.summer_months = [5, 6, 7, 8, 9]
        self.summer_pv_generation_kwh = 0
        self.summer_pv_curtailed_kwh = 0

    def simplified_pv_generation(self, dni_wh_per_m2_hour, dhi_wh_per_m2_hour, day_of_year, hour_of_day):
        """Simplified PV generation calculation using basic solar geometry"""
        if self.total_pv_capacity_kwp == 0:
            return 0.0, 0.0
            
        # Calculate solar declination
        declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))
        
        # Calculate hour angle
        hour_angle = 15 * (hour_of_day - 12)
        
        # Calculate solar elevation angle
        elevation = math.asin(
            math.sin(math.radians(declination)) * math.sin(math.radians(self.latitude_deg)) +
            math.cos(math.radians(declination)) * math.cos(math.radians(self.latitude_deg)) * 
            math.cos(math.radians(hour_angle))
        )
        
        # If sun is below horizon, no generation
        if elevation <= 0:
            return 0.0, 0.0
        
        # Calculate solar zenith angle
        zenith = math.pi/2 - elevation
        
        # Calculate GHI from DNI and DHI
        ghi = dhi_wh_per_m2_hour + dni_wh_per_m2_hour * math.cos(zenith)
        
        # Simplified tilt factor for 27° south-facing panels
        tilt_factor = 1.15 if 6 <= hour_of_day <= 18 else 0.8
        
        # Monthly temperature derating factors
        temp_factors = [0.92, 0.94, 0.97, 1.0, 1.02, 1.0, 0.98, 0.99, 1.01, 1.0, 0.96, 0.93]
        month = ((day_of_year - 1) // 30) + 1
        month = min(12, max(1, month))
        temp_factor = temp_factors[month - 1]
        
        # Calculate DC power
        irradiance_factor = ghi / 1000.0  # Normalize to STC (1000 W/m²)
        dc_power_kw = self.total_pv_capacity_kwp * irradiance_factor * tilt_factor * temp_factor
        
        # Apply AC conversion losses
        ac_power_kw = dc_power_kw * (1 - self.acdc_loss_factor) * 0.96  # inverter efficiency
        
        return max(0, dc_power_kw), max(0, ac_power_kw)

    def simulate_hour(self, timestamp, hot_water_demand_kwh_hour, dni_wh_per_m2_hour, 
                     dhi_wh_per_m2_hour, electricity_price_sek_per_kwh):
        dt_hours = 1.0
        day_of_year = timestamp.timetuple().tm_yday
        hour_of_day = timestamp.hour
        
        is_summer_hour = timestamp.month in self.summer_months

        # TES heat loss
        tes_loss_kwh = self.tes_energy_kwh * self.tes_loss_rate_per_hour * dt_hours
        self.tes_energy_kwh = max(0, self.tes_energy_kwh - tes_loss_kwh)
        
        # PV generation
        pv_gross_kw, pv_net_kw = self.simplified_pv_generation(
            dni_wh_per_m2_hour, dhi_wh_per_m2_hour, day_of_year, hour_of_day
        )
        available_pv_kwh = pv_net_kw * dt_hours
        
        # Track summer PV generation
        if is_summer_hour:
            self.summer_pv_generation_kwh += available_pv_kwh
        
        # Meet demand from TES first
        tes_to_demand_kwh = min(hot_water_demand_kwh_hour, self.tes_energy_kwh)
        self.tes_energy_kwh -= tes_to_demand_kwh
        
        if is_summer_hour:
            self.summer_total_demand_kwh += hot_water_demand_kwh_hour

        unmet_demand_kwh = hot_water_demand_kwh_hour - tes_to_demand_kwh
        
        # Use PV for heat pump
        pv_to_hp_kwh = 0
        thermal_from_pv = 0
        
        if available_pv_kwh > 0:
            thermal_needed_for_demand = unmet_demand_kwh
            thermal_needed_for_storage = max(0, self.tes_physical_max_kwh - self.tes_energy_kwh)
            max_thermal_to_add = thermal_needed_for_demand + thermal_needed_for_storage
            max_thermal_from_pv = available_pv_kwh * self.heat_pump_cop
            thermal_from_pv = min(max_thermal_to_add, max_thermal_from_pv)
            pv_to_hp_kwh = thermal_from_pv / self.heat_pump_cop
            self.tes_energy_kwh += thermal_from_pv
        
        # Use grid for remaining needs
        remaining_unmet_demand = max(0, unmet_demand_kwh - thermal_from_pv)
        thermal_needed_for_minimum = max(0, self.tes_minimum_kwh - self.tes_energy_kwh)
        total_thermal_from_grid_needed = remaining_unmet_demand + thermal_needed_for_minimum
        
        if total_thermal_from_grid_needed > 0:
            grid_to_hp_kwh = total_thermal_from_grid_needed / self.heat_pump_cop
            if 0 < grid_to_hp_kwh < self.hp_min_electrical_input_kw:
                grid_to_hp_kwh = self.hp_min_electrical_input_kw
            thermal_from_grid = grid_to_hp_kwh * self.heat_pump_cop
            self.tes_energy_kwh += thermal_from_grid
            if is_summer_hour:
                self.summer_grid_to_hp_thermal_kwh += thermal_from_grid
        
        # Calculate curtailment
        pv_curtailed_kwh = max(0, available_pv_kwh - pv_to_hp_kwh)
        
        if is_summer_hour:
            self.summer_pv_curtailed_kwh += pv_curtailed_kwh

    def run_simulation(self, demand_values, dni_values, dhi_values, price_values):
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        time_index = [start_date + timedelta(hours=i) for i in range(8784)]
        for i in range(8784):
            timestamp = time_index[i]
            self.simulate_hour(
                timestamp,
                demand_values.iloc[i],
                dni_values.iloc[i],
                dhi_values.iloc[i],
                price_values.iloc[i]
            )

    def get_summer_demand_met_percentage(self):
        if self.summer_total_demand_kwh > 0:
            demand_met_by_pv_and_tes = self.summer_total_demand_kwh - self.summer_grid_to_hp_thermal_kwh
            percentage_met = (demand_met_by_pv_and_tes / self.summer_total_demand_kwh) * 100
            return max(0, percentage_met)
        else:
            return 0
    
    def get_summer_curtailment_percentage(self):
        if self.summer_pv_generation_kwh > 0:
            curtailment_percentage = (self.summer_pv_curtailed_kwh / self.summer_pv_generation_kwh) * 100
            return curtailment_percentage
        else:
            return 0

def find_optimal_panel_count(panel_counts, demand_percentages, curtailment_percentages):
    """Find optimal panel count using MCDA with 0.8 demand weight and 0.2 curtailment penalty"""
    optimal_score = -float('inf')
    optimal_panels = None
    optimal_demand = 0
    optimal_curtailment = 0
    
    demand_weight = 0.8
    curtailment_penalty = 0.2
    
    for i, (panels, demand_pct, curtail_pct) in enumerate(zip(panel_counts, demand_percentages, curtailment_percentages)):
        score = demand_weight * demand_pct - curtailment_penalty * curtail_pct
        
        if score > optimal_score:
            optimal_score = score
            optimal_panels = panels
            optimal_demand = demand_pct
            optimal_curtailment = curtail_pct
    
    return optimal_panels, optimal_demand, optimal_curtailment, optimal_score

def run_single_optimization(demand_values, dni_values, dhi_values, price_values, 
                           tes_capacity_multiplier=1.0, heat_pump_cop=2.5):
    """Run optimization for a single parameter set"""
    
    tes_volume_m3 = 36.0
    pv_panel_capacity_kwp = 0.45
    hp_min_electrical_input_kw = 10
    
    panel_counts = list(range(200, 401, 5))  # 200 to 400 panels, increment of 5
    summer_demand_met_percentages = []
    summer_curtailment_percentages = []
    
    for panel_count in panel_counts:
        simulation = BuildingEnergySimulation(
            tes_volume_m3=tes_volume_m3, 
            heat_pump_cop=heat_pump_cop,
            pv_panel_capacity_kwp=pv_panel_capacity_kwp, 
            panel_count=panel_count,
            hp_min_electrical_input_kw=hp_min_electrical_input_kw,
            tes_capacity_multiplier=tes_capacity_multiplier
        )
        simulation.run_simulation(demand_values, dni_values, dhi_values, price_values)
        summer_demand_met = simulation.get_summer_demand_met_percentage()
        summer_curtailment = simulation.get_summer_curtailment_percentage()
        
        summer_demand_met_percentages.append(summer_demand_met)
        summer_curtailment_percentages.append(summer_curtailment)
    
    # Find optimal panel count
    optimal_panels, optimal_demand, optimal_curtailment, optimal_score = find_optimal_panel_count(
        panel_counts, summer_demand_met_percentages, summer_curtailment_percentages
    )
    
    return (optimal_panels, optimal_demand, optimal_curtailment, optimal_score,
            panel_counts, summer_demand_met_percentages, summer_curtailment_percentages)

def run_comprehensive_sensitivity_analysis(demand_file, dni_dhi_file, price_file):
    """Run comprehensive sensitivity analysis with visual representation"""
    
    # Load data
    demand_df = pd.read_csv(demand_file)
    dni_dhi_df = pd.read_csv(dni_dhi_file) 
    price_df = pd.read_csv(price_file)
    demand_values = demand_df['energy demand (kWh)']
    dni_values = dni_dhi_df['ALLSKY_SFC_SW_DNI']    
    dhi_values = dni_dhi_df['ALLSKY_SFC_SW_DHI']    
    price_values = price_df['price']
    
    # Sensitivity factors
    factors = [-0.2, -0.1, 0.0, 0.1, 0.2]
    multipliers = [1 + f for f in factors]
    
    # Define parameter sets for analysis
    tes_multipliers = multipliers
    cop_values = [2.0, 2.25, 2.5, 2.75, 3.0]  # Different COP values instead of multipliers
    
    print("Running Comprehensive Sensitivity Analysis...")
    print("="*60)
    
    # Create subplots for comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    
    # Main optimization curves plot
    ax1 = plt.subplot(2, 3, (1, 2))  # Top row, spans 2 columns
    ax2 = plt.subplot(2, 3, (4, 5))  # Bottom row, spans 2 columns
    ax3 = plt.subplot(2, 3, 3)       # Top right
    ax4 = plt.subplot(2, 3, 6)       # Bottom right
    
    # Color schemes
    tes_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']  # Red to Purple
    cop_colors = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # Brown to Cyan
    
    # Storage for sensitivity results
    tes_sensitivity_results = []
    cop_sensitivity_results = []
    
    # TES Capacity Sensitivity Analysis
    print("TES Capacity Sensitivity:")
    for i, (multiplier, color) in enumerate(zip(tes_multipliers, tes_colors)):
        factor_pct = factors[i] * 100
        print(f"  TES: {factor_pct:+.0f}%", end=" ")
        
        results = run_single_optimization(
            demand_values, dni_values, dhi_values, price_values,
            tes_capacity_multiplier=multiplier, heat_pump_cop=2.5
        )
        
        optimal_panels, optimal_demand, optimal_curtailment, _, panel_counts, demand_pcts, curtail_pcts = results
        
        # Plot optimization curves
        label = f'TES {factor_pct:+.0f}% (Opt: {optimal_panels})'
        ax1.plot(panel_counts, demand_pcts, color=color, linewidth=2, alpha=0.8, label=label)
        ax2.plot(panel_counts, curtail_pcts, color=color, linewidth=2, alpha=0.8, label=label)
        
        # Mark optimal point
        ax1.scatter([optimal_panels], [optimal_demand], color=color, s=100, 
                   marker='o', edgecolor='white', linewidth=2, zorder=5)
        ax2.scatter([optimal_panels], [optimal_curtailment], color=color, s=100, 
                   marker='o', edgecolor='white', linewidth=2, zorder=5)
        
        tes_sensitivity_results.append({
            'factor_pct': factor_pct,
            'optimal_panels': optimal_panels,
            'demand_met': optimal_demand,
            'curtailment': optimal_curtailment
        })
        
        print(f"-> Optimal: {optimal_panels} panels ({optimal_demand:.1f}% demand, {optimal_curtailment:.1f}% curtail)")
    
    print("\nHeat Pump COP Sensitivity:")
    # Heat Pump COP Sensitivity Analysis  
    for i, (cop, color) in enumerate(zip(cop_values, cop_colors)):
        print(f"  COP: {cop:.2f}", end=" ")
        
        results = run_single_optimization(
            demand_values, dni_values, dhi_values, price_values,
            tes_capacity_multiplier=1.0, heat_pump_cop=cop
        )
        
        optimal_panels, optimal_demand, optimal_curtailment, _, panel_counts, demand_pcts, curtail_pcts = results
        
        # Plot optimization curves
        label = f'COP {cop:.2f} (Opt: {optimal_panels})'
        ax1.plot(panel_counts, demand_pcts, color=color, linewidth=2, alpha=0.8, 
                linestyle='--', label=label)
        ax2.plot(panel_counts, curtail_pcts, color=color, linewidth=2, alpha=0.8, 
                linestyle='--', label=label)
        
        # Mark optimal point
        ax1.scatter([optimal_panels], [optimal_demand], color=color, s=100, 
                   marker='s', edgecolor='white', linewidth=2, zorder=5)
        ax2.scatter([optimal_panels], [optimal_curtailment], color=color, s=100, 
                   marker='s', edgecolor='white', linewidth=2, zorder=5)
        
        cop_sensitivity_results.append({
            'cop': cop,
            'optimal_panels': optimal_panels,
            'demand_met': optimal_demand,
            'curtailment': optimal_curtailment
        })
        
        print(f"-> Optimal: {optimal_panels} panels ({optimal_demand:.1f}% demand, {optimal_curtailment:.1f}% curtail)")
    
    # Format main plots
    ax1.set_xlabel('Number of PV Panels', fontsize=12)
    ax1.set_ylabel('Summer Demand Met by PV & TES (%)', fontsize=12)
    ax1.set_title('Optimization Curves: Demand Coverage\n(○ = TES Sensitivity, □ = COP Sensitivity)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    ax2.set_xlabel('Number of PV Panels', fontsize=12)
    ax2.set_ylabel('Summer PV Curtailment (%)', fontsize=12)
    ax2.set_title('Optimization Curves: PV Curtailment\n(○ = TES Sensitivity, □ = COP Sensitivity)', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Create sensitivity summary plots
    tes_factors_plot = [r['factor_pct'] for r in tes_sensitivity_results]
    tes_panels_plot = [r['optimal_panels'] for r in tes_sensitivity_results]
    cop_values_plot = [r['cop'] for r in cop_sensitivity_results]
    cop_panels_plot = [r['optimal_panels'] for r in cop_sensitivity_results]
    
    # TES sensitivity plot
    ax3.plot(tes_factors_plot, tes_panels_plot, 'bo-', linewidth=3, markersize=10, 
             color='steelblue', markerfacecolor='lightblue', markeredgecolor='steelblue', markeredgewidth=2)
    ax3.set_xlabel('TES Capacity Change (%)', fontsize=11)
    ax3.set_ylabel('Optimal Panel Count', fontsize=11)
    ax3.set_title('TES Capacity\nSensitivity', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for x, y in zip(tes_factors_plot, tes_panels_plot):
        ax3.annotate(f'{int(y)}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # COP sensitivity plot
    ax4.plot(cop_values_plot, cop_panels_plot, 's-', linewidth=3, markersize=10, 
             color='darkorange', markerfacecolor='lightsalmon', markeredgecolor='darkorange', markeredgewidth=2)
    ax4.set_xlabel('Heat Pump COP', fontsize=11)
    ax4.set_ylabel('Optimal Panel Count', fontsize=11)
    ax4.set_title('Heat Pump COP\nSensitivity', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for x, y in zip(cop_values_plot, cop_panels_plot):
        ax4.annotate(f'{int(y)}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*80)
    
    print("\nTES Capacity Impact:")
    for result in tes_sensitivity_results:
        print(f"  {result['factor_pct']:+.0f}%: {result['optimal_panels']} panels "
              f"({result['demand_met']:.1f}% demand, {result['curtailment']:.1f}% curtail)")
    
    print("\nHeat Pump COP Impact:")
    for result in cop_sensitivity_results:
        print(f"  COP {result['cop']:.2f}: {result['optimal_panels']} panels "
              f"({result['demand_met']:.1f}% demand, {result['curtailment']:.1f}% curtail)")
    
    return tes_sensitivity_results, cop_sensitivity_results

if __name__ == "__main__":
    demand_file = 'energy_demand.csv'   
    dni_file = 'dni_dhi.csv'            
    price_file = 'electricity_price.csv' 
    
    # Run comprehensive sensitivity analysis
    tes_results, cop_results = run_comprehensive_sensitivity_analysis(demand_file, dni_file, price_file)