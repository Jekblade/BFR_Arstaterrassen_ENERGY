import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
import warnings
from pvlib import location, irradiance, solarposition

warnings.filterwarnings("ignore", message="The get_cmap function is deprecated")

# Load input data from CSV files - ensure these files are in the same directory
try:
    demand_df = pd.read_csv("energy_demand.csv")
    dni_df = pd.read_csv("dni_dhi.csv")  # Should include 'ALLSKY_SFC_SW_DNI' and 'ALLSKY_SFC_SW_DHI'
    price_df = pd.read_csv("electricity_price.csv")
except FileNotFoundError as e:
    print(f"Error loading data CSV files: {e}")
    print("Please ensure 'energy_demand.csv', 'dni_dhi.csv', and 'electricity_price.csv' are in the script's directory.")
    exit()

demand_values = demand_df['energy demand (kWh)']
dni_values = dni_df['ALLSKY_SFC_SW_DNI']
dhi_values = dni_df['ALLSKY_SFC_SW_DHI']
price_values = price_df['price']

class BuildingEnergySimulation:
    def __init__(self, tes_volume_m3, heat_pump_cop, hp_min_electrical_input_kw=15.0, south_27deg_panel_count=0):
        self.heat_pump_cop = heat_pump_cop
        self.hp_min_electrical_input_kw = hp_min_electrical_input_kw
        self.tes_volume_m3 = tes_volume_m3
        self.south_27deg_panel_count = south_27deg_panel_count
        self.water_density_kg_per_m3 = 1000
        self.water_specific_heat_kj_per_kg_k = 4.184
        self.tes_loss_rate_per_hour = 0.005
        self.tes_target_maintenance_kwh = 1272.0
        self.tes_physical_max_kwh = 2090.0
        self.tes_energy_kwh = self.tes_target_maintenance_kwh
        self.results = []
        self.total_solar_thermal_generation_kwh = 0
        self.total_solar_thermal_to_tes_kwh = 0
        self.total_solar_thermal_curtailed_kwh = 0
        self.total_hot_water_demand_kwh = 0
        self.total_tes_to_demand_kwh = 0
        self.total_grid_to_hp_kwh = 0
        self.total_grid_electricity_cost_sek = 0
        self.total_baseline_electricity_cost_sek = 0
        self.unmet_demand_kwh = 0
        
        # Summer-specific tracking
        self.summer_months = [5, 6, 7, 8, 9]
        self.summer_solar_thermal_generation_kwh = 0
        self.summer_solar_thermal_curtailed_kwh = 0

    def calculate_solar_thermal_generation(self, dni_wh_per_m2_hour, dhi_wh_per_m2_hour, timestamp):
        site = location.Location(59.33, 18.06, tz='Europe/Stockholm') # Stockholm
        times = pd.DatetimeIndex([timestamp])
        solar_pos = solarposition.get_solarposition(times, site.latitude, site.longitude)

        if solar_pos['zenith'].iloc[0] >= 90 or dni_wh_per_m2_hour <= 0 or self.south_27deg_panel_count == 0:
            return 0.0

        solar_zenith_deg = solar_pos['apparent_zenith'].iloc[0]
        # GHI needed for POA calculation if not directly available
        ghi = dhi_wh_per_m2_hour + dni_wh_per_m2_hour * np.cos(np.radians(solar_zenith_deg))
        dhi = dhi_wh_per_m2_hour # dhi is already diffuse horizontal

        # Modified collector configuration: only south-facing 27deg panels
        collector_configs = [
            {'tilt': 27, 'azimuth': 185, 'count': self.south_27deg_panel_count}
        ]
    
        collector_efficiency = 0.65
        collector_area_m2 = 2.35
        total_q_kwh = 0

        for config in collector_configs:
            if config['count'] == 0:
                continue
            
            poa = irradiance.get_total_irradiance(
                surface_tilt=config['tilt'],
                surface_azimuth=config['azimuth'],
                dni=dni_wh_per_m2_hour,
                ghi=ghi,
                dhi=dhi,
                solar_zenith=solar_pos['apparent_zenith'],
                solar_azimuth=solar_pos['azimuth']
            )
            poa_global_irradiance_wh_per_m2 = poa['poa_global'].iloc[0] if not poa.empty and pd.notna(poa['poa_global'].iloc[0]) else 0
            
            q_solar_wh = collector_efficiency * collector_area_m2 * config['count'] * poa_global_irradiance_wh_per_m2
            total_q_kwh += q_solar_wh / 1000

        return max(0, total_q_kwh)

    def simulate_hour(self, timestamp, demand_kwh, dni, dhi, price):
        dt = 1.0 # Hour
        is_summer_hour = timestamp.month in self.summer_months
        
        # TES losses
        tes_loss = self.tes_energy_kwh * self.tes_loss_rate_per_hour * dt
        self.tes_energy_kwh = max(0, self.tes_energy_kwh - tes_loss)

        # Meet demand from TES
        supplied_to_demand_from_tes = min(demand_kwh, self.tes_energy_kwh)
        self.tes_energy_kwh -= supplied_to_demand_from_tes
        unmet_this_hour = demand_kwh - supplied_to_demand_from_tes

        self.total_hot_water_demand_kwh += demand_kwh
        self.total_tes_to_demand_kwh += supplied_to_demand_from_tes
        self.unmet_demand_kwh += unmet_this_hour

        # Solar thermal generation
        q_solar_kwh = self.calculate_solar_thermal_generation(dni, dhi, timestamp)
        self.total_solar_thermal_generation_kwh += q_solar_kwh
        
        # Track summer solar thermal generation
        if is_summer_hour:
            self.summer_solar_thermal_generation_kwh += q_solar_kwh
        
        # Solar to TES - calculate how much can actually be stored
        available_tes_capacity = self.tes_physical_max_kwh - self.tes_energy_kwh
        thermal_from_solar_to_tes = min(q_solar_kwh, available_tes_capacity)
        
        # Calculate curtailed solar thermal energy
        solar_thermal_curtailed_this_hour = max(0, q_solar_kwh - thermal_from_solar_to_tes)
        
        self.tes_energy_kwh += thermal_from_solar_to_tes
        self.total_solar_thermal_to_tes_kwh += thermal_from_solar_to_tes
        self.total_solar_thermal_curtailed_kwh += solar_thermal_curtailed_this_hour
        
        # Track summer curtailment
        if is_summer_hour:
            self.summer_solar_thermal_curtailed_kwh += solar_thermal_curtailed_this_hour

        # Heat Pump operation to maintain TES target
        thermal_needed_for_tes_maintenance = max(0, self.tes_target_maintenance_kwh - self.tes_energy_kwh)
        hp_electrical_consumed_kwh = 0
        thermal_added_by_hp_to_tes_kwh = 0

        if thermal_needed_for_tes_maintenance > 0:
            required_hp_electrical_input = thermal_needed_for_tes_maintenance / self.heat_pump_cop
            
            if 0 < required_hp_electrical_input < self.hp_min_electrical_input_kw:
                hp_electrical_consumed_kwh = self.hp_min_electrical_input_kw
            elif required_hp_electrical_input >= self.hp_min_electrical_input_kw:
                hp_electrical_consumed_kwh = required_hp_electrical_input
            else:
                hp_electrical_consumed_kwh = 0

            if hp_electrical_consumed_kwh > 0:
                thermal_output_from_hp = hp_electrical_consumed_kwh * self.heat_pump_cop
                thermal_added_by_hp_to_tes_kwh = min(thermal_output_from_hp, self.tes_physical_max_kwh - self.tes_energy_kwh)
                self.tes_energy_kwh += thermal_added_by_hp_to_tes_kwh
                
                self.total_grid_to_hp_kwh += hp_electrical_consumed_kwh
                self.total_grid_electricity_cost_sek += hp_electrical_consumed_kwh * price
            else:
                 thermal_added_by_hp_to_tes_kwh = 0
        
        self.results.append({
            'timestamp': timestamp,
            'hot_water_demand_kwh': demand_kwh,
            'dni_wh_per_m2_hour': dni,
            'dhi_wh_per_m2_hour': dhi,
            'solar_thermal_generation_kwh': q_solar_kwh,
            'solar_thermal_to_tes_kwh': thermal_from_solar_to_tes,
            'solar_thermal_curtailed_kwh': solar_thermal_curtailed_this_hour,
            'grid_to_hp_kwh': hp_electrical_consumed_kwh,
            'thermal_added_by_hp_to_tes_kwh': thermal_added_by_hp_to_tes_kwh,
            'tes_to_demand_kwh': supplied_to_demand_from_tes,
            'unmet_demand_this_hour_kwh': unmet_this_hour,
            'tes_energy_kwh': self.tes_energy_kwh,
            'electricity_price_sek_per_kwh': price,
            'tes_loss_kwh': tes_loss
        })

    def get_summer_curtailment_percentage(self):
        """Calculate the percentage of summer solar thermal generation that gets curtailed"""
        if self.summer_solar_thermal_generation_kwh > 0:
            curtailment_percentage = (self.summer_solar_thermal_curtailed_kwh / self.summer_solar_thermal_generation_kwh) * 100
            return curtailment_percentage
        else:
            return 0.0

    def run_simulation(self, demand_values_sim, dni_values_sim, dhi_values_sim, price_values_sim):
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        time_index = pd.date_range(start=start_date, periods=8784, freq='h')

        if not (len(demand_values_sim) == len(dni_values_sim) == len(dhi_values_sim) == len(price_values_sim) == 8784):
            raise ValueError(f"Input data series must contain 8784 hourly records. Current lengths: Demand {len(demand_values_sim)}, DNI {len(dni_values_sim)}, DHI {len(dhi_values_sim)}, Price {len(price_values_sim)}")

        current_total_baseline_electricity_cost_sek = 0
        for i in range(8784):
            current_total_baseline_electricity_cost_sek += demand_values_sim.iloc[i] * price_values_sim.iloc[i]
            self.simulate_hour(
                time_index[i],
                demand_values_sim.iloc[i],
                dni_values_sim.iloc[i],
                dhi_values_sim.iloc[i],
                price_values_sim.iloc[i]
            )
        self.total_baseline_electricity_cost_sek = current_total_baseline_electricity_cost_sek
        self.results_df = pd.DataFrame(self.results)
        if not self.results_df.empty:
            self.results_df.set_index('timestamp', inplace=True)

def find_optimal_panel_count(panel_counts, demand_percentages, curtailment_percentages):
    """
    Find the optimal panel count that maximizes demand fulfillment while minimizing curtailment.
    Uses a weighted score: higher weight for demand fulfillment, penalty for curtailment.
    """
    optimal_score = -float('inf')
    optimal_panels = None
    optimal_demand = 0
    optimal_curtailment = 0
    
    # Weight factors - prioritize demand fulfillment over curtailment reduction
    demand_weight = 1.0
    curtailment_penalty = 0.3  # Lower weight for curtailment penalty
    
    for i, (panels, demand_pct, curtail_pct) in enumerate(zip(panel_counts, demand_percentages, curtailment_percentages)):
        # Calculate score: maximize demand fulfillment, minimize curtailment
        score = demand_weight * demand_pct - curtailment_penalty * curtail_pct
        
        if score > optimal_score:
            optimal_score = score
            optimal_panels = panels
            optimal_demand = demand_pct
            optimal_curtailment = curtail_pct
    
    return optimal_panels, optimal_demand, optimal_curtailment, optimal_score

if __name__ == "__main__":
    # Define the panel counts you want to iterate through
    panel_counts_to_iterate = np.unique(np.concatenate((np.arange(0, 721, 10), [725])))
    panel_counts_to_iterate.sort()

    plot_data_points = []

    print("Enhanced Solar Thermal Panel Optimization Analysis...")
    print("Panel Count | Summer Demand Met (%) | Summer Curtailment (%)")
    print("------------------------------------------------------------")

    for p_count in panel_counts_to_iterate:
        simulation_instance = BuildingEnergySimulation(
            tes_volume_m3=36.0,
            heat_pump_cop=0.95,
            hp_min_electrical_input_kw=15.0,
            south_27deg_panel_count=int(p_count)
        )
        simulation_instance.run_simulation(demand_values, dni_values, dhi_values, price_values)

        # --- Summer Analysis for this iteration ---
        if not hasattr(simulation_instance, 'results_df') or simulation_instance.results_df.empty:
            print(f"{int(p_count):11} | No simulation results. Skipping.")
            plot_data_points.append({
                'panel_count': int(p_count), 
                'summer_demand_met_solar_percent': 0,
                'summer_curtailment_percent': 0
            })
            continue

        summer_months = [5, 6, 7, 8, 9] # May, June, July, August, September
        summer_data_df = simulation_instance.results_df[simulation_instance.results_df.index.month.isin(summer_months)]

        if summer_data_df.empty:
            percent_summer_demand_met_by_solar = 0.0
        else:
            total_summer_demand_kwh = summer_data_df['hot_water_demand_kwh'].sum()
            summer_hp_electrical_from_grid_kwh = summer_data_df['grid_to_hp_kwh'].sum()
            
            summer_hp_thermal_from_grid_kwh = summer_hp_electrical_from_grid_kwh * simulation_instance.heat_pump_cop

            if total_summer_demand_kwh > 0:
                demand_met_by_solar_system_kwh = total_summer_demand_kwh - summer_hp_thermal_from_grid_kwh
                demand_met_by_solar_system_kwh = max(0, min(demand_met_by_solar_system_kwh, total_summer_demand_kwh))
                
                percent_summer_demand_met_by_solar = (demand_met_by_solar_system_kwh / total_summer_demand_kwh) * 100
            else:
                percent_summer_demand_met_by_solar = 0.0
        
        # Get summer curtailment percentage
        summer_curtailment_percent = simulation_instance.get_summer_curtailment_percentage()
        
        print(f"{int(p_count):11} | {percent_summer_demand_met_by_solar:17.1f} | {summer_curtailment_percent:18.1f}")
        plot_data_points.append({
            'panel_count': int(p_count), 
            'summer_demand_met_solar_percent': percent_summer_demand_met_by_solar,
            'summer_curtailment_percent': summer_curtailment_percent
        })

    # Convert simulation results to a DataFrame for plotting
    results_df_for_plot = pd.DataFrame(plot_data_points)
    results_df_for_plot.sort_values('panel_count', inplace=True)

    # Extract data for analysis
    panel_counts = results_df_for_plot['panel_count'].values
    demand_percentages = results_df_for_plot['summer_demand_met_solar_percent'].values
    curtailment_percentages = results_df_for_plot['summer_curtailment_percent'].values

    # --- Define Target Percentage ---
    target_percentage = 97.0

    # --- Calculate optimal_panels_for_target_percentage ---
    optimal_panels_for_target_percentage = None
    eligible_panels_for_plot = results_df_for_plot[
        results_df_for_plot['summer_demand_met_solar_percent'] >= target_percentage
    ]

    if not eligible_panels_for_plot.empty:
        optimal_panels_for_target_percentage = eligible_panels_for_plot['panel_count'].min()

    # Find overall optimal balance
    optimal_panels, optimal_demand, optimal_curtailment, optimal_score = find_optimal_panel_count(
        panel_counts, demand_percentages, curtailment_percentages
    )

    # --- Enhanced Plotting with dual y-axes ---
    fig, ax1 = plt.subplots(figsize=(14, 10))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of South-Facing 27° Solar Thermal Panels')
    ax1.set_ylabel('Summer Demand Met by Solar System (%)', color=color1)
    line1 = ax1.plot(panel_counts, demand_percentages, 'b-', linewidth=2, 
                     marker='o', markersize=4, label='Summer Demand Met by Solar System & TES')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=target_percentage, color='r', linestyle='--', alpha=0.7, 
                label=f'{target_percentage}% Demand Target')
    
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Summer Solar Thermal Curtailment (%)', color=color2)
    line2 = ax2.plot(panel_counts, curtailment_percentages, 'orange', linewidth=2, 
                     marker='s', markersize=4, label='Summer Solar Thermal Curtailment')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add vertical lines for key points
    if optimal_panels_for_target_percentage is not None:
        ax1.axvline(x=optimal_panels_for_target_percentage, color='g', linestyle='--', alpha=0.7, 
                   label=f'Min panels for ≥{target_percentage}%: {optimal_panels_for_target_percentage}')
    
    if optimal_panels is not None:
        ax1.axvline(x=optimal_panels, color='purple', linestyle=':', alpha=0.8, linewidth=2,
                   label=f'Optimal balance: {optimal_panels} panels')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Solar Thermal Panel Optimization: Summer Demand Coverage vs Curtailment Analysis', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 105)
    
    # Customize X-axis ticks for clarity
    if len(panel_counts_to_iterate) > 0:
        min_pc_tick = np.min(panel_counts_to_iterate)
        max_pc_tick = np.max(panel_counts_to_iterate)
        
        num_ticks = 10
        if max_pc_tick > min_pc_tick:
            tick_interval = max(1, int(np.ceil((max_pc_tick - min_pc_tick) / num_ticks)))
        else:
            tick_interval = 1
        
        ax1.set_xticks(np.arange(min_pc_tick, max_pc_tick + tick_interval, tick_interval))

    plt.tight_layout()
    plt.show()

    # --- Enhanced Final Output ---
    print("\n" + "="*80)
    print("ENHANCED SOLAR THERMAL OPTIMIZATION SUMMARY")
    print("="*80)
    
    if optimal_panels_for_target_percentage is not None:
        target_idx = results_df_for_plot[results_df_for_plot['panel_count'] == optimal_panels_for_target_percentage].index[0]
        demand_97 = results_df_for_plot.iloc[target_idx]['summer_demand_met_solar_percent']
        curtail_97 = results_df_for_plot.iloc[target_idx]['summer_curtailment_percent']
        
        print(f"Minimum panels for ≥{target_percentage}% summer demand: {optimal_panels_for_target_percentage}")
        print(f"  - Demand met: {demand_97:.1f}%")
        print(f"  - Curtailment: {curtail_97:.1f}%")
        print(f"  - Total collector area: {optimal_panels_for_target_percentage * 2.35:.1f} m²")
        print()
    else:
        max_met_percentage = results_df_for_plot['summer_demand_met_solar_percent'].max()
        print(f"The target of {target_percentage}% summer demand was not reached with up to {results_df_for_plot['panel_count'].max()} panels.")
        print(f"The maximum percentage achieved was {max_met_percentage:.2f}%.")
        print()
    
    if optimal_panels is not None:
        print(f"OPTIMAL BALANCE POINT: {optimal_panels} panels")
        print(f"  - Summer demand met by solar system: {optimal_demand:.1f}%")
        print(f"  - Summer solar thermal curtailment: {optimal_curtailment:.1f}%")
        print(f"  - Optimization score: {optimal_score:.2f}")
        print(f"  - Total collector area: {optimal_panels * 2.35:.1f} m²")
        print()
    
    # Additional analysis
    max_demand_met = max(demand_percentages)
    max_demand_idx = list(demand_percentages).index(max_demand_met)
    max_demand_panels = panel_counts[max_demand_idx]
    max_demand_curtailment = curtailment_percentages[max_demand_idx]
    
    print(f"Maximum demand achievable: {max_demand_met:.1f}% with {max_demand_panels} panels")
    print(f"  - Associated curtailment: {max_demand_curtailment:.1f}%")
    print(f"  - Total collector area: {max_demand_panels * 2.35:.1f} m²")