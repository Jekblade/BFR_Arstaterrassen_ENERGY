import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
from pvlib import solarposition, irradiance, pvsystem, location
from datetime import datetime, timedelta
import math


class BuildingEnergySimulation:
    def __init__(self, tes_volume_m3, heat_pump_cop, pv_panel_capacity_kwp, hp_min_electrical_input_kw=5, acdc_loss_factor=0.03):
        # System parameters
        self.heat_pump_cop = heat_pump_cop
        self.hp_min_electrical_input_kw = hp_min_electrical_input_kw
        self.acdc_loss_factor = acdc_loss_factor

        # PV Panel Configuration
        self.pv_config = {
            'South_27': {'azimuth': 185, 'slope': 27, 'count': 546}, # 546 
            'South_23': {'azimuth': 185, 'slope': 23, 'count': 320}, # 320
            'East_23': {'azimuth': 94, 'slope': 23, 'count': 0},  # 167
            'West_23': {'azimuth': 275, 'slope': 23, 'count': 0}  # 171
        }
        self.pv_panel_capacity_kwp = pv_panel_capacity_kwp

        # Calculate total installed capacity per group
        self.pv_installed_kwp = {}
        for key, config in self.pv_config.items():
            self.pv_installed_kwp[key] = config['count'] * self.pv_panel_capacity_kwp

        # Thermal Energy Storage (TES) parameters
        self.tes_volume_m3 = tes_volume_m3
        self.water_density_kg_per_m3 = 1000
        self.water_specific_heat_kj_per_kg_k = 4.184
        self.tes_loss_rate_per_hour = 0.005
        self.tes_room_temp_deg = 20.0
        self.tes_minimum_kwh = 1072.0 #This in KWh for 36m2 tank is 40deg celsius. We can't go lower as the water will not be hot.
        self.tes_physical_max_kwh = 2090.0 # keeping max at 70 deg

        self.tes_capacity = self.tes_physical_max_kwh - self.tes_minimum_kwh # This is the total available capacity (max-min)
       
        self.tes_energy_kwh = 0 # Setting the current energy level to just room temp (delta t = 0)

        # District heating pricing (SEK per MWh)
        self.district_heating_prices = {
            'jan_dec': 1200,  # January and December
            'feb_mar_nov': 863,  # February, March, November
            'summer': 0  # Apr, May, Jun, Jul, Aug, Sep, Oct - no district heating in summer
        }

        # Tracking results
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
        
        # New tracking for district heating
        self.total_district_heating_kwh = 0
        self.total_district_heating_cost_sek = 0
        self.total_pv_savings_from_district_heating_sek = 0

        # Location: Stockholm, Sweden
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
        
        # PV system parameters
        self.temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        self.module_parameters = {'pdc0': 1.0, 'gamma_pdc': -0.004}
        self.inverter_parameters = {'pdc0': 1.0, 'eta_inv_nom': 0.96}

    def get_district_heating_price(self, month):
        """Get district heating price based on month"""
        if month in [1, 12]:  # January, December
            return self.district_heating_prices['jan_dec'] / 1000  # Convert to SEK per kWh
        elif month in [2, 3, 11]:  # February, March, November
            return self.district_heating_prices['feb_mar_nov'] / 1000  # Convert to SEK per kWh
        else:  # Summer months (Apr, May, Jun, Jul, Aug, Sep, Oct)
            return self.district_heating_prices['summer']

    def is_summer_month(self, month):
        """Check if month is considered summer (May-September)"""
        return month in [5, 6, 7, 8, 9]

    def calculate_pv_generation(self, dni_wh_per_m2_hour, dhi_wh_per_m2_hour, day_of_year, hour_of_day):
        """Calculate PV generation using actual DNI and DHI inputs"""
        date = datetime(2024, 1, 1) + timedelta(days=day_of_year-1, hours=hour_of_day)
        times = pd.DatetimeIndex([date])
        
        solar_position = solarposition.get_solarposition(times, self.latitude_deg, self.longitude_deg)
        
        # Only check solar position for generation potential
        if solar_position['zenith'].iloc[0] >= 90:
            return 0.0, 0.0
        
        # Calculate GHI using solar geometry
        solar_zenith = solar_position['apparent_zenith'].iloc[0]
        ghi_wh_per_m2_hour = dhi_wh_per_m2_hour + (dni_wh_per_m2_hour * np.cos(np.radians(solar_zenith)))
        
        stockholm_avg_temps = {
            1: -3,  2: -3,  3: 1,
            4: 6,   5: 12,  6: 17,
            7: 20,  8: 19,  9: 14,
            10: 8, 11: 3,  12: -1
        }

        month = date.month
        ambient_temp = stockholm_avg_temps.get(month) # Average temperatures
            
        total_gross_power_kw = 0
        
        for key, config in self.pv_config.items():
            panel_tilt = config['slope']
            panel_azimuth = config['azimuth']
            group_capacity_kwp = self.pv_installed_kwp[key]
            
            poa_irradiance = irradiance.get_total_irradiance(
                surface_tilt=panel_tilt,
                surface_azimuth=panel_azimuth,
                dni=dni_wh_per_m2_hour,
                ghi=ghi_wh_per_m2_hour,
                dhi=dhi_wh_per_m2_hour,
                solar_zenith=solar_position['apparent_zenith'],
                solar_azimuth=solar_position['azimuth']
            )
            
            cell_temperature = pvsystem.temperature.sapm_cell(
                poa_global=poa_irradiance['poa_global'],
                temp_air=ambient_temp,
                wind_speed=1.0,
                **self.temperature_model_parameters
            )
            
            dc_power = pvsystem.pvwatts_dc(
                g_poa_effective=poa_irradiance['poa_global'],
                temp_cell=cell_temperature,
                pdc0=group_capacity_kwp,
                gamma_pdc=self.module_parameters['gamma_pdc']
            )
            
            if not dc_power.empty:
                total_gross_power_kw += dc_power.iloc[0]
        
        net_total_power_kw = total_gross_power_kw * (1 - self.acdc_loss_factor)
        return total_gross_power_kw, max(0, net_total_power_kw)

    def simulate_hour(self, timestamp, hot_water_demand_kwh_hour, dni_wh_per_m2_hour, dhi_wh_per_m2_hour, electricity_price_sek_per_kwh):
        dt_hours = 1.0
        day_of_year = timestamp.timetuple().tm_yday
        hour_of_day = timestamp.hour
        month = timestamp.month
        
        # TES Standby Losses
        tes_loss_kwh = self.tes_energy_kwh * self.tes_loss_rate_per_hour * dt_hours
        self.tes_energy_kwh = max(0, self.tes_energy_kwh - tes_loss_kwh)
        
        # PV Generation
        pv_gross_kw, pv_net_kw = self.calculate_pv_generation(dni_wh_per_m2_hour, dhi_wh_per_m2_hour, day_of_year, hour_of_day)
        available_pv_kwh = pv_net_kw * dt_hours
        self.total_pv_generation_kwh_gross += pv_gross_kw * dt_hours
        self.total_pv_generation_kwh_net += available_pv_kwh
        
        # Meet hot water demand from TES first
        tes_to_demand_kwh = min(hot_water_demand_kwh_hour, self.tes_energy_kwh)
        self.tes_energy_kwh -= tes_to_demand_kwh
        
        # Track demand met
        self.total_hot_water_demand_kwh += hot_water_demand_kwh_hour
        self.total_tes_to_demand_kwh += tes_to_demand_kwh
        
        # Calculate unmet demand
        unmet_demand_kwh = hot_water_demand_kwh_hour - tes_to_demand_kwh
        
        # Use PV for heat pump - Convert electricity to thermal energy
        pv_to_hp_kwh = 0
        thermal_from_pv = 0
        
        if available_pv_kwh > 0:
            # Calculate thermal energy needed to:
            # 1. Meet unmet demand
            # 2. Fill TES up to maximum (if there's excess PV)
            thermal_needed_for_demand = unmet_demand_kwh
            thermal_needed_for_storage = max(0, self.tes_physical_max_kwh - self.tes_energy_kwh)
            
            # Total thermal energy we could add
            max_thermal_to_add = thermal_needed_for_demand + thermal_needed_for_storage
            
            # But we're limited by available PV electricity
            max_thermal_from_pv = available_pv_kwh * self.heat_pump_cop
            
            # Actual thermal energy to add
            thermal_from_pv = min(max_thermal_to_add, max_thermal_from_pv)
            
            # Calculate PV electricity needed
            pv_to_hp_kwh = thermal_from_pv / self.heat_pump_cop
            
            # Add thermal to TES
            self.tes_energy_kwh += thermal_from_pv
            
            # Track PV usage
            self.total_pv_to_hp_kwh += pv_to_hp_kwh
        
        # Handle remaining unmet demand based on season
        grid_to_hp_kwh = 0
        thermal_from_grid = 0
        district_heating_kwh = 0
        district_heating_cost_sek = 0
        pv_savings_from_district_heating_sek = 0
        
        # Check if we still have unmet demand after using PV
        remaining_unmet_demand = max(0, unmet_demand_kwh - thermal_from_pv)
        
        # Check if TES is below minimum
        thermal_needed_for_minimum = max(0, self.tes_minimum_kwh - self.tes_energy_kwh)
        
        # Total thermal needed
        total_thermal_needed = remaining_unmet_demand + thermal_needed_for_minimum
        
        if total_thermal_needed > 0:
            if self.is_summer_month(month):
                # Summer: Use grid electricity for heat pump as before
                grid_to_hp_kwh = total_thermal_needed / self.heat_pump_cop
                
                # Enforce minimum HP electrical input if running
                if 0 < grid_to_hp_kwh < self.hp_min_electrical_input_kw:
                    grid_to_hp_kwh = self.hp_min_electrical_input_kw
                
                # Calculate thermal energy produced
                thermal_from_grid = grid_to_hp_kwh * self.heat_pump_cop
                
                # Add to TES
                self.tes_energy_kwh += thermal_from_grid
                
                # Track grid usage
                self.total_grid_to_hp_kwh += grid_to_hp_kwh
                self.total_grid_electricity_cost_sek += grid_to_hp_kwh * electricity_price_sek_per_kwh
            else:
                # Non-summer months: Use district heating for unmet demand
                district_heating_kwh = total_thermal_needed
                district_heating_price = self.get_district_heating_price(month)
                district_heating_cost_sek = district_heating_kwh * district_heating_price
                
                # Add thermal energy to TES from district heating
                self.tes_energy_kwh += district_heating_kwh
                
                # Track district heating usage
                self.total_district_heating_kwh += district_heating_kwh
                self.total_district_heating_cost_sek += district_heating_cost_sek
                
                # Calculate savings: if we had used PV instead of district heating
                # This represents the value of PV generation that could offset district heating
                if available_pv_kwh > pv_to_hp_kwh:
                    # Calculate how much district heating could be offset by remaining PV
                    remaining_pv_kwh = available_pv_kwh - pv_to_hp_kwh
                    max_thermal_from_remaining_pv = remaining_pv_kwh * self.heat_pump_cop
                    potential_district_heating_offset = min(district_heating_kwh, max_thermal_from_remaining_pv)
                    pv_savings_from_district_heating_sek = potential_district_heating_offset * district_heating_price
                    self.total_pv_savings_from_district_heating_sek += pv_savings_from_district_heating_sek
        
        # Calculate curtailed PV
        pv_curtailed_kwh = max(0, available_pv_kwh - pv_to_hp_kwh)
        self.total_pv_curtailed_kwh += pv_curtailed_kwh
        
        # Determine operating modes
        modes = []
        if pv_to_hp_kwh > 0: modes.append("PV to HP")
        if grid_to_hp_kwh > 0: modes.append("Grid to HP")
        if district_heating_kwh > 0: modes.append("District Heating")
        if tes_to_demand_kwh > 0: modes.append("TES to Demand")
        if pv_curtailed_kwh > 1e-3: modes.append("PV Curtailed")
        
        if not modes: modes.append("Standby")
        operating_mode_str = " & ".join(sorted(list(set(modes))))
        
        # Calculate TES state of charge
        tes_soc_percent = (self.tes_energy_kwh / self.tes_physical_max_kwh) * 100 if self.tes_physical_max_kwh > 0 else 0
        tes_soc_percent = max(0, min(100, tes_soc_percent))
        
        # Record results
        self.results.append({
            'timestamp': timestamp,
            'hot_water_demand_kwh': hot_water_demand_kwh_hour,
            'dni_wh_per_m2_hour': dni_wh_per_m2_hour,
            'dhi_wh_per_m2_hour': dhi_wh_per_m2_hour,
            'pv_generation_kw_net': pv_net_kw, 
            'pv_to_hp_kwh': pv_to_hp_kwh,
            'thermal_from_pv_kwh': thermal_from_pv,
            'grid_to_hp_kwh': grid_to_hp_kwh,
            'district_heating_kwh': district_heating_kwh,
            'district_heating_cost_sek': district_heating_cost_sek,
            'hp_electrical_consumed_kwh': pv_to_hp_kwh + grid_to_hp_kwh,
            'thermal_added_by_hp_to_tes_kwh': thermal_from_pv + thermal_from_grid,
            'tes_to_demand_kwh': tes_to_demand_kwh,
            'tes_energy_kwh': self.tes_energy_kwh,
            'tes_soc_percent': tes_soc_percent,
            'grid_total_usage_kwh_hour': grid_to_hp_kwh,
            'electricity_price_sek_per_kwh': electricity_price_sek_per_kwh,
            'operating_mode': operating_mode_str,
            'tes_loss_kwh': tes_loss_kwh,
            'pv_curtailed_kwh': pv_curtailed_kwh,
            'pv_savings_from_district_heating_sek': pv_savings_from_district_heating_sek
        })

    def run_simulation(self, demand_values, dni_values, dhi_values, price_values):
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        time_index = [start_date + timedelta(hours=i) for i in range(8784)] # As 2024 is leap year there is 1 more day

        # Reset simulation state
        self.__init__(self.tes_volume_m3, self.heat_pump_cop, self.pv_panel_capacity_kwp,
                      self.hp_min_electrical_input_kw, self.acdc_loss_factor)

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

    def output_analysis(self):
        print("\n--- Annual System Performance Analysis ---")
        print(f"Total Net PV Electricity Produced Annually: {self.total_pv_generation_kwh_net:.2f} kWh")
        print(f"Total PV Electricity Used for HP: {self.total_pv_to_hp_kwh:.2f} kWh")
        if self.total_pv_generation_kwh_net > 0:
            print(f"PV Curtailment (losses): {self.total_pv_curtailed_kwh:.2f} kWh ({self.total_pv_curtailed_kwh/self.total_pv_generation_kwh_net*100:.1f}%)")
            print(f"PV Utilization Rate: {self.total_pv_to_hp_kwh/self.total_pv_generation_kwh_net*100:.1f}%")

        # District heating analysis
        print(f"\n--- District Heating vs PV System Analysis ---")
        print(f"Total District Heating Used (Non-Summer): {self.total_district_heating_kwh:.2f} kWh")
        print(f"Total District Heating Cost: {self.total_district_heating_cost_sek:.2f} SEK")
        print(f"Potential Savings from PV offsetting District Heating: {self.total_pv_savings_from_district_heating_sek:.2f} SEK")
        
        # Calculate total system cost vs baseline (district heating only)
        baseline_district_heating_cost = self.total_hot_water_demand_kwh * (self.district_heating_prices['jan_dec'] / 1000) * (2/12) + \
                                      self.total_hot_water_demand_kwh * (self.district_heating_prices['feb_mar_nov'] / 1000) * (3/12) + \
                                      self.total_hot_water_demand_kwh * 0 * (7/12)  # Summer months
        
        total_system_cost = self.total_grid_electricity_cost_sek + self.total_district_heating_cost_sek
        annual_savings = baseline_district_heating_cost - total_system_cost
        
        print(f"Baseline Annual Cost (District Heating Only): {baseline_district_heating_cost:.2f} SEK")
        print(f"Actual System Cost (PV+HP+District Heating): {total_system_cost:.2f} SEK")
        print(f"Annual Savings with PV System: {annual_savings:.2f} SEK")

        if not self.results_df.empty:
            print("\n--- Annual Operating Mode Distribution ---")
            annual_mode_counts = self.results_df['operating_mode'].value_counts()
            annual_mode_percentages = (annual_mode_counts / len(self.results_df)) * 100
            print(annual_mode_percentages.round(1).to_string())

            # Summer analysis (May-September)
            summer_months = [5, 6, 7, 8, 9] # May to Sep
            summer_data = self.results_df[self.results_df.index.month.isin(summer_months)]

            summer_pv_gen_net = summer_data['pv_generation_kw_net'].sum()
            summer_pv_to_hp = summer_data['pv_to_hp_kwh'].sum()
            summer_grid_to_hp = summer_data['grid_to_hp_kwh'].sum()
            summer_pv_curtailed = summer_data['pv_curtailed_kwh'].sum()
            
            summer_grid_cost = (summer_data['grid_to_hp_kwh'] * summer_data['electricity_price_sek_per_kwh']).sum()
            
            summer_total_demand = summer_data['hot_water_demand_kwh'].sum()
            
            # Calculate thermal energy provided by PV (via heat pump)
            summer_thermal_from_pv = summer_data['thermal_from_pv_kwh'].sum()
            
            # Summer self-sufficiency calculation
            summer_self_sufficiency = (summer_thermal_from_pv / summer_total_demand) * 100

            print("\n--- Summer System Performance Analysis (May-September) ---")
            print(f"Total summer hot water energy demand (thermal): {summer_total_demand:.2f} kWh")
            print(f"Net PV Electricity Produced: {summer_pv_gen_net:.2f} kWh")
            print(f"PV Electricity to HP: {summer_pv_to_hp:.2f} kWh")
            print(f"Thermal Energy from PV (via HP): {summer_thermal_from_pv:.2f} kWh")
            print(f"Summer Self-Sufficiency (PV+TES): {summer_self_sufficiency:.2f}%")
            print(f"Grid Electricity to HP (Summer): {summer_grid_to_hp:.2f} kWh")
            print(f"Summer PV Curtailment: {summer_pv_curtailed:.2f} kWh ({summer_pv_curtailed/summer_pv_gen_net*100:.1f}%)")
            print(f"Electricity Cost (Summer, from Grid): {summer_grid_cost:.2f} SEK")
            
            # Non-summer analysis
            non_summer_months = [1, 2, 3, 4, 10, 11, 12]
            non_summer_data = self.results_df[self.results_df.index.month.isin(non_summer_months)]
            
            non_summer_pv_gen = non_summer_data['pv_generation_kw_net'].sum()
            non_summer_pv_to_hp = non_summer_data['pv_to_hp_kwh'].sum()
            non_summer_district_heating = non_summer_data['district_heating_kwh'].sum()
            non_summer_district_heating_cost = non_summer_data['district_heating_cost_sek'].sum()
            non_summer_total_demand = non_summer_data['hot_water_demand_kwh'].sum()
            non_summer_thermal_from_pv = non_summer_data['thermal_from_pv_kwh'].sum()
            non_summer_pv_curtailed = non_summer_data['pv_curtailed_kwh'].sum()
            
            non_summer_self_sufficiency = (non_summer_thermal_from_pv / non_summer_total_demand) * 100
            
            print("\n--- Non-Summer System Performance Analysis (Oct-Apr) ---")
            print(f"Total non-summer hot water energy demand (thermal): {non_summer_total_demand:.2f} kWh")
            print(f"Net PV Electricity Produced: {non_summer_pv_gen:.2f} kWh")
            print(f"PV Electricity to HP: {non_summer_pv_to_hp:.2f} kWh")
            print(f"Thermal Energy from PV (via HP): {non_summer_thermal_from_pv:.2f} kWh")
            print(f"District Heating Used: {non_summer_district_heating:.2f} kWh")
            print(f"Non-Summer Self-Sufficiency (PV+TES): {non_summer_self_sufficiency:.2f}%")
            print(f"District Heating Cost: {non_summer_district_heating_cost:.2f} SEK")
            print(f"Non-Summer PV Curtailment: {non_summer_pv_curtailed:.2f} kWh ({non_summer_pv_curtailed/non_summer_pv_gen*100:.1f}%)")

    def plot_summer_operating_modes(self):
        summer_months = [5, 6, 7, 8, 9]
        summer_data = self.results_df[self.results_df.index.month.isin(summer_months)]
        mode_counts = summer_data['operating_mode'].value_counts()

        plt.figure(figsize=(10, 8))
        plt.pie(mode_counts, labels=mode_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Summer Operating Mode Distribution (May-September)')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def plot_annual_operating_modes(self):
        """Plot annual operating mode distribution"""
        mode_counts = self.results_df['operating_mode'].value_counts()

        plt.figure(figsize=(12, 8))
        plt.pie(mode_counts, labels=mode_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Annual Operating Mode Distribution')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def plot_weekly_performance(self, week_number):
        weekly_data = self.results_df[self.results_df.index.isocalendar().week == week_number].copy()

        fig, ax1 = plt.subplots(figsize=(15, 7))
        time_index_weekly = weekly_data.index
        hours_in_week = np.arange(len(weekly_data))

        ax1.plot(hours_in_week, weekly_data['hot_water_demand_kwh'] / 2.5, label='Hot Water Demand (kWh elec equiv)', color='blue')
        ax1.plot(hours_in_week, weekly_data['pv_generation_kw_net'], label='Net PV Generation (kWh elec)', color='orange')
        ax1.plot(hours_in_week, weekly_data['grid_total_usage_kwh_hour'], label='Grid to HP (kWh elec)', color='purple')
        ax1.plot(hours_in_week, weekly_data['district_heating_kwh'], label='District Heating (kWh thermal)', color='red')
        ax1.plot(hours_in_week, weekly_data['hp_electrical_consumed_kwh'], label='HP Electrical Input (kWh)', color='black', linestyle=':')
        ax1.plot(hours_in_week, weekly_data['pv_curtailed_kwh'], label='PV Curtailed (kWh)', color='orange', linestyle='--')
        
        ax1.set_xlabel('Hour in Week')
        ax1.set_ylabel('Power / Energy (kW or kWh)')
        ax1.tick_params(axis='y')
        
        ax2 = ax1.twinx()
        ax2.plot(hours_in_week, weekly_data['tes_soc_percent'], label='TES SoC (%)', color='darkred', linestyle='-')
        ax2.set_ylabel('TES State of Charge (%)')
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis='y')

        tes_minimum_soc_percent = (self.tes_minimum_kwh / self.tes_physical_max_kwh) * 100
        ax2.axhline(
            tes_minimum_soc_percent,
            color='goldenrod',
            linestyle=(0, (5, 5)),
            linewidth=2,
            label='Minimum TES energy (40deg)')

        # X-axis labels
        xticks_pos = hours_in_week[::24]
        xticks_labels = [time_index_weekly[i].strftime('%Y-%m-%d') for i in xticks_pos if i < len(time_index_weekly)]
        if len(xticks_pos) > len(xticks_labels) : xticks_pos = xticks_pos[:len(xticks_labels)]

        ax1.set_xticks(xticks_pos)
        ax1.set_xticklabels(xticks_labels, rotation=45, ha='right')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title(f'Energy System Performance - Week {week_number}')
        plt.tight_layout()
        plt.show()

    def plot_monthly_energy_flow(self):
        """Plot monthly energy flows and costs"""
        monthly_data = self.results_df.groupby(self.results_df.index.month).agg({
            'hot_water_demand_kwh': 'sum',
            'pv_generation_kw_net': 'sum',
            'pv_to_hp_kwh': 'sum',
            'thermal_from_pv_kwh': 'sum',
            'grid_to_hp_kwh': 'sum',
            'district_heating_kwh': 'sum',
            'district_heating_cost_sek': 'sum',
            'pv_curtailed_kwh': 'sum'
        })
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Monthly Energy Generation and Demand
        ax1.bar(months, monthly_data['hot_water_demand_kwh'], alpha=0.7, label='Hot Water Demand', color='blue')
        ax1.bar(months, monthly_data['pv_generation_kw_net'], alpha=0.7, label='PV Generation', color='green')
        ax1.set_ylabel('Energy (kWh)')
        ax1.set_title('Monthly Energy Generation vs Demand')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Monthly Energy Sources for Hot Water
        ax2.bar(months, monthly_data['thermal_from_pv_kwh'], alpha=0.7, label='PV via HP', color='green')
        ax2.bar(months, monthly_data['grid_to_hp_kwh'] * self.heat_pump_cop, 
                bottom=monthly_data['thermal_from_pv_kwh'], alpha=0.7, label='Grid via HP', color='red')
        ax2.bar(months, monthly_data['district_heating_kwh'], 
                bottom=monthly_data['thermal_from_pv_kwh'] + monthly_data['grid_to_hp_kwh'] * self.heat_pump_cop, 
                alpha=0.7, label='District Heating', color='brown')
        ax2.set_ylabel('Thermal Energy (kWh)')
        ax2.set_title('Monthly Energy Sources for Hot Water')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Monthly PV Utilization
        pv_utilization = (monthly_data['pv_to_hp_kwh'] / monthly_data['pv_generation_kw_net'] * 100).fillna(0)
        ax3.bar(months, pv_utilization, alpha=0.7, color='orange')
        ax3.set_ylabel('PV Utilization (%)')
        ax3.set_title('Monthly PV Utilization Rate')
        ax3.set_ylim(0, 100)
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Monthly Costs
        grid_costs = monthly_data['grid_to_hp_kwh'] * 1.5  # Approximate electricity price
        ax4.bar(months, monthly_data['district_heating_cost_sek'], alpha=0.7, label='District Heating Cost', color='brown')
        ax4.bar(months, grid_costs, bottom=monthly_data['district_heating_cost_sek'], 
                alpha=0.7, label='Grid Electricity Cost', color='red')
        ax4.set_ylabel('Cost (SEK)')
        ax4.set_title('Monthly Energy Costs')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# Main function updated for DHI input and district heating analysis
def run_full_simulation_process(demand_file, dni_dhi_file, price_file, tes_volume_m3, heat_pump_cop, 
                                pv_panel_capacity_kwp, hp_min_electrical_input_kw, plot_weekly=False, 
                                week_number=None, plot_monthly=True):
    
    demand_df = pd.read_csv(demand_file)
    dni_dhi_df = pd.read_csv(dni_dhi_file) 
    price_df = pd.read_csv(price_file)

    demand_values = demand_df['energy demand (kWh)']
    dni_values = dni_dhi_df['ALLSKY_SFC_SW_DNI']    
    dhi_values = dni_dhi_df['ALLSKY_SFC_SW_DHI']    
    price_values = price_df['price']            

    simulation = BuildingEnergySimulation(
        tes_volume_m3=tes_volume_m3,
        heat_pump_cop=heat_pump_cop,
        pv_panel_capacity_kwp=pv_panel_capacity_kwp,
        hp_min_electrical_input_kw=hp_min_electrical_input_kw
    )

    simulation.run_simulation(demand_values, dni_values, dhi_values, price_values)
    simulation.output_analysis()
    simulation.plot_summer_operating_modes()
    simulation.plot_annual_operating_modes()

    if plot_monthly:
        simulation.plot_monthly_energy_flow()

    if plot_weekly and week_number is not None:
        simulation.plot_weekly_performance(week_number)
    
    return simulation

if __name__ == "__main__":
    demand_file = 'energy_demand.csv'   
    dni_file = 'dni_dhi.csv'            
    price_file = 'electricity_price.csv' 

    # System parameters
    tes_volume_m3 = 36.0
    heat_pump_cop = 2.5
    pv_panel_capacity_kwp = 0.45
    hp_min_electrical_input_kw = 5

    simulation_result = run_full_simulation_process(
        demand_file,
        dni_file,
        price_file,
        tes_volume_m3,
        heat_pump_cop,
        pv_panel_capacity_kwp,
        hp_min_electrical_input_kw,
        plot_weekly=True,
        week_number=22,
        plot_monthly=True
    )
