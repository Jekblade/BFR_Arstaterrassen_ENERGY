import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
from pvlib import solarposition, irradiance, pvsystem, location
from datetime import datetime, timedelta
import math

class BuildingEnergySimulation:
    def __init__(self, tes_volume_m3, heat_pump_cop, pv_panel_capacity_kwp, panel_count, hp_min_electrical_input_kw=5, acdc_loss_factor=0.03):
        self.heat_pump_cop = heat_pump_cop
        self.hp_min_electrical_input_kw = hp_min_electrical_input_kw
        self.acdc_loss_factor = acdc_loss_factor
        self.pv_config = {
            'South_27': {'azimuth': 185, 'slope': 27, 'count': panel_count},
            'South_23': {'azimuth': 185, 'slope': 23, 'count': 0},
            'East_23': {'azimuth': 94, 'slope': 23, 'count': 0},
            'West_23': {'azimuth': 275, 'slope': 23, 'count': 0}
        }
        self.pv_panel_capacity_kwp = pv_panel_capacity_kwp
        self.pv_installed_kwp = {}
        for key, config in self.pv_config.items():
            self.pv_installed_kwp[key] = config['count'] * self.pv_panel_capacity_kwp
        self.tes_volume_m3 = tes_volume_m3
        self.water_density_kg_per_m3 = 1000
        self.water_specific_heat_kj_per_kg_k = 4.184
        self.tes_loss_rate_per_hour = 0.005
        self.tes_room_temp_deg = 20.0
        self.tes_minimum_kwh = 1272.0
        self.tes_physical_max_kwh = 2090.0
        self.tes_capacity = self.tes_physical_max_kwh - self.tes_minimum_kwh
        self.tes_energy_kwh = 0
        self.results = []
        self.total_pv_generation_kwh_gross = 0
        self.total_pv_generation_kwh_net = 0
        self.total_hot_water_demand_kwh = 0
        self.total_tes_to_demand_kwh = 0
        self.total_pv_to_hp_kwh = 0
        self.total_grid_to_hp_kwh = 0
        self.total_grid_electricity_cost_sek = 0
        self.total_baseline_electricity_cost_sek = 0
        self.total_pv_curtailed_kwh = 0
        self.latitude_deg = 59.33
        self.longitude_deg = 18.06
        self.altitude = 44
        self.timezone = 'Europe/Stockholm'
        self.site = location.Location(
            latitude=self.latitude_deg,
            longitude=self.longitude_deg,
            tz=self.timezone,
            altitude=self.altitude,
            name='Stockholm'
        )
        self.temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        self.module_parameters = {'pdc0': 1.0, 'gamma_pdc': -0.004}
        self.inverter_parameters = {'pdc0': 1.0, 'eta_inv_nom': 0.96}

        self.summer_total_demand_kwh = 0
        self.summer_grid_to_hp_thermal_kwh = 0
        self.summer_months = [5, 6, 7, 8, 9]
        
        # Add summer-specific tracking for curtailment
        self.summer_pv_generation_kwh = 0
        self.summer_pv_curtailed_kwh = 0

    def calculate_pv_generation(self, dni_wh_per_m2_hour, dhi_wh_per_m2_hour, day_of_year, hour_of_day):
        date = datetime(2024, 1, 1) + timedelta(days=day_of_year-1, hours=hour_of_day)
        times = pd.DatetimeIndex([date])
        solar_position = solarposition.get_solarposition(times, self.latitude_deg, self.longitude_deg)
        if solar_position['zenith'].iloc[0] >= 90:
            return 0.0, 0.0
        solar_zenith = solar_position['apparent_zenith'].iloc[0]
        ghi_wh_per_m2_hour = dhi_wh_per_m2_hour + (dni_wh_per_m2_hour * np.cos(np.radians(solar_zenith)))
        stockholm_avg_temps = {
            1: -3,  2: -3,  3: 1, 4: 6,   5: 12,  6: 17,
            7: 20,  8: 19,  9: 14, 10: 8, 11: 3,  12: -1
        }
        month = date.month
        ambient_temp = stockholm_avg_temps.get(month)
        total_gross_power_kw = 0
        for key, config in self.pv_config.items():
            panel_tilt = config['slope']
            panel_azimuth = config['azimuth']
            group_capacity_kwp = self.pv_installed_kwp[key]
            if group_capacity_kwp == 0:
                continue
            poa_irradiance = irradiance.get_total_irradiance(
                surface_tilt=panel_tilt, surface_azimuth=panel_azimuth,
                dni=dni_wh_per_m2_hour, ghi=ghi_wh_per_m2_hour, dhi=dhi_wh_per_m2_hour,
                solar_zenith=solar_position['apparent_zenith'], solar_azimuth=solar_position['azimuth']
            )
            cell_temperature = pvsystem.temperature.sapm_cell(
                poa_global=poa_irradiance['poa_global'], temp_air=ambient_temp,
                wind_speed=1.0, **self.temperature_model_parameters
            )
            dc_power = pvsystem.pvwatts_dc(
                g_poa_effective=poa_irradiance['poa_global'], temp_cell=cell_temperature,
                pdc0=group_capacity_kwp, gamma_pdc=self.module_parameters['gamma_pdc']
            )
            if not dc_power.empty:
                total_gross_power_kw += dc_power.iloc[0]
        net_total_power_kw = total_gross_power_kw * (1 - self.acdc_loss_factor)
        return total_gross_power_kw, max(0, net_total_power_kw)

    def simulate_hour(self, timestamp, hot_water_demand_kwh_hour, dni_wh_per_m2_hour, dhi_wh_per_m2_hour, electricity_price_sek_per_kwh):
        dt_hours = 1.0
        day_of_year = timestamp.timetuple().tm_yday
        hour_of_day = timestamp.hour
        
        is_summer_hour = timestamp.month in self.summer_months

        tes_loss_kwh = self.tes_energy_kwh * self.tes_loss_rate_per_hour * dt_hours
        self.tes_energy_kwh = max(0, self.tes_energy_kwh - tes_loss_kwh)
        
        pv_gross_kw, pv_net_kw = self.calculate_pv_generation(dni_wh_per_m2_hour, dhi_wh_per_m2_hour, day_of_year, hour_of_day)
        available_pv_kwh = pv_net_kw * dt_hours
        self.total_pv_generation_kwh_gross += pv_gross_kw * dt_hours
        self.total_pv_generation_kwh_net += available_pv_kwh
        
        # Track summer PV generation
        if is_summer_hour:
            self.summer_pv_generation_kwh += available_pv_kwh
        
        tes_to_demand_kwh = min(hot_water_demand_kwh_hour, self.tes_energy_kwh)
        self.tes_energy_kwh -= tes_to_demand_kwh
        
        self.total_hot_water_demand_kwh += hot_water_demand_kwh_hour
        self.total_tes_to_demand_kwh += tes_to_demand_kwh
        
        if is_summer_hour:
            self.summer_total_demand_kwh += hot_water_demand_kwh_hour

        unmet_demand_kwh = hot_water_demand_kwh_hour - tes_to_demand_kwh
        
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
            self.total_pv_to_hp_kwh += pv_to_hp_kwh
        
        grid_to_hp_kwh = 0
        thermal_from_grid = 0
        remaining_unmet_demand = max(0, unmet_demand_kwh - thermal_from_pv)
        thermal_needed_for_minimum = max(0, self.tes_minimum_kwh - self.tes_energy_kwh)
        total_thermal_from_grid_needed = remaining_unmet_demand + thermal_needed_for_minimum
        
        if total_thermal_from_grid_needed > 0:
            grid_to_hp_kwh = total_thermal_from_grid_needed / self.heat_pump_cop
            if 0 < grid_to_hp_kwh < self.hp_min_electrical_input_kw:
                grid_to_hp_kwh = self.hp_min_electrical_input_kw
            thermal_from_grid = grid_to_hp_kwh * self.heat_pump_cop
            self.tes_energy_kwh += thermal_from_grid
            self.total_grid_to_hp_kwh += grid_to_hp_kwh
            self.total_grid_electricity_cost_sek += grid_to_hp_kwh * electricity_price_sek_per_kwh
            if is_summer_hour:
                self.summer_grid_to_hp_thermal_kwh += thermal_from_grid
        
        pv_curtailed_kwh = max(0, available_pv_kwh - pv_to_hp_kwh)
        self.total_pv_curtailed_kwh += pv_curtailed_kwh
        
        # Track summer curtailment
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
        self.results_df = pd.DataFrame(self.results)
        if not self.results_df.empty:
            self.results_df.set_index('timestamp', inplace=True)

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
    """
    Find the optimal panel count that maximizes demand fulfillment while minimizing curtailment.
    Uses a weighted score: higher weight for demand fulfillment, penalty for curtailment.
    """
    optimal_score = -float('inf')
    optimal_panels = None
    optimal_demand = 0
    optimal_curtailment = 0
    
    # Weight factors - prioritize demand fulfillment over curtailment reduction
    demand_weight = 0.8
    curtailment_penalty = 0.2  # Lower weight for curtailment penalty
    
    for i, (panels, demand_pct, curtail_pct) in enumerate(zip(panel_counts, demand_percentages, curtailment_percentages)):
        # Calculate score: maximize demand fulfillment, minimize curtailment
        score = demand_weight * demand_pct - curtailment_penalty * curtail_pct
        
        if score > optimal_score:
            optimal_score = score
            optimal_panels = panels
            optimal_demand = demand_pct
            optimal_curtailment = curtail_pct
    
    return optimal_panels, optimal_demand, optimal_curtailment, optimal_score

def run_panel_optimization(demand_file, dni_dhi_file, price_file):
    demand_df = pd.read_csv(demand_file)
    dni_dhi_df = pd.read_csv(dni_dhi_file) 
    price_df = pd.read_csv(price_file)
    demand_values = demand_df['energy demand (kWh)']
    dni_values = dni_dhi_df['ALLSKY_SFC_SW_DNI']    
    dhi_values = dni_dhi_df['ALLSKY_SFC_SW_DHI']    
    price_values = price_df['price']
    
    tes_volume_m3 = 36.0
    heat_pump_cop = 0.95
    pv_panel_capacity_kwp = 0.45
    hp_min_electrical_input_kw = 10
    
    panel_counts = list(range(0, 600 + 1, 5))
    summer_demand_met_percentages = []
    summer_curtailment_percentages = []
    
    print("Running enhanced panel optimization analysis...")
    print("Panel Count | Summer Demand Met (%) | Summer Curtailment (%)")
    print("-" * 65)
    
    for panel_count in panel_counts:
        simulation = BuildingEnergySimulation(
            tes_volume_m3=tes_volume_m3, heat_pump_cop=heat_pump_cop,
            pv_panel_capacity_kwp=pv_panel_capacity_kwp, panel_count=panel_count,
            hp_min_electrical_input_kw=hp_min_electrical_input_kw
        )
        simulation.run_simulation(demand_values, dni_values, dhi_values, price_values)
        summer_demand_met = simulation.get_summer_demand_met_percentage()
        summer_curtailment = simulation.get_summer_curtailment_percentage()
        
        summer_demand_met_percentages.append(summer_demand_met)
        summer_curtailment_percentages.append(summer_curtailment)
        
        print(f"{panel_count:11d} | {summer_demand_met:17.1f} | {summer_curtailment:18.1f}")
    
    # Find optimal panel count for different criteria
    optimal_panels_for_97_percent = None
    target_percentage = 97.0
    
    for i, (panels, percentage) in enumerate(zip(panel_counts, summer_demand_met_percentages)):
        if percentage >= target_percentage:
            optimal_panels_for_97_percent = panels
            break
    
    # Find overall optimal balance
    optimal_panels, optimal_demand, optimal_curtailment, optimal_score = find_optimal_panel_count(
        panel_counts, summer_demand_met_percentages, summer_curtailment_percentages
    )
    
    # Create the enhanced plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 10))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of PV Panels (27° slope, South-facing)')
    ax1.set_ylabel('Summer Demand Met by PV & TES (%)', color=color1)
    line1 = ax1.plot(panel_counts, summer_demand_met_percentages, 'b-', linewidth=2, 
                     marker='o', markersize=4, label='Summer Demand Met by PV & TES')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=target_percentage, color='r', linestyle='--', alpha=0.7, 
                label=f'{target_percentage}% Demand Target')
    
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Summer PV Curtailment (%)', color=color2)
    line2 = ax2.plot(panel_counts, summer_curtailment_percentages, 'orange', linewidth=2, 
                     marker='s', markersize=4, label='Summer PV Curtailment')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add vertical lines for key points
    if optimal_panels_for_97_percent is not None:
        ax1.axvline(x=optimal_panels_for_97_percent, color='g', linestyle='--', alpha=0.7, 
                   label=f'Min panels for ≥{target_percentage}%: {optimal_panels_for_97_percent}')
    
    if optimal_panels is not None:
        ax1.axvline(x=optimal_panels, color='purple', linestyle=':', alpha=0.8, linewidth=2,
                   label=f'Optimal balance: {optimal_panels} panels')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    ax1.grid(True, alpha=0.3)
    ax1.set_title('PV Panel Optimization: Summer Demand Coverage vs Curtailment Analysis', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    print("\n" + "="*80)
    print("ENHANCED OPTIMIZATION SUMMARY")
    print("="*80)
    
    if optimal_panels_for_97_percent is not None:
        demand_97 = summer_demand_met_percentages[panel_counts.index(optimal_panels_for_97_percent)]
        curtail_97 = summer_curtailment_percentages[panel_counts.index(optimal_panels_for_97_percent)]
        print(f"Minimum panels for ≥{target_percentage}% summer demand: {optimal_panels_for_97_percent}")
        print(f"  - Demand met: {demand_97:.1f}%")
        print(f"  - Curtailment: {curtail_97:.1f}%")
        print()
    
    if optimal_panels is not None:
        print(f"OPTIMAL BALANCE POINT: {optimal_panels} panels")
        print(f"  - Summer demand met by PV & TES: {optimal_demand:.1f}%")
        print(f"  - Summer PV curtailment: {optimal_curtailment:.1f}%")
        print(f"  - Optimization score: {optimal_score:.2f}")
        print(f"  - Total installed capacity: {optimal_panels * pv_panel_capacity_kwp:.1f} kWp")
        print()
    
    # Additional analysis
    max_demand_met = max(summer_demand_met_percentages)
    max_demand_panels = panel_counts[summer_demand_met_percentages.index(max_demand_met)]
    max_demand_curtailment = summer_curtailment_percentages[summer_demand_met_percentages.index(max_demand_met)]
    
    print(f"Maximum demand achievable: {max_demand_met:.1f}% with {max_demand_panels} panels")
    print(f"  - Associated curtailment: {max_demand_curtailment:.1f}%")
    
    plt.show()
    return panel_counts, summer_demand_met_percentages, summer_curtailment_percentages

if __name__ == "__main__":
    demand_file = 'energy_demand.csv'   
    dni_file = 'dni_dhi.csv'            
    price_file = 'electricity_price.csv' 

    panel_counts, demand_percentages, curtailment_percentages = run_panel_optimization(demand_file, dni_file, price_file)