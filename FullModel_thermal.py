import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib # Still needed for solar position and irradiance
from pvlib import solarposition, irradiance, location # Ensure all are imported
from datetime import datetime, timedelta
import math
import warnings

warnings.filterwarnings("ignore", message="The get_cmap function is deprecated")

class SolarThermalEnergySimulation:
    def __init__(self, tes_volume_m3, heat_pump_cop, hp_min_electrical_input_kw=5):
        # System parameters
        self.heat_pump_cop = heat_pump_cop
        self.hp_min_electrical_input_kw = hp_min_electrical_input_kw

        # Solar Thermal Collector Configuration (hardcoded as per example)
        self.collector_configs = [
            {'tilt': 27, 'azimuth': 185, 'count': 200, 'efficiency': 0.65, 'area_m2': 2.35}
        ]

        # Thermal Energy Storage (TES) parameters
        self.tes_volume_m3 = tes_volume_m3
        self.water_density_kg_per_m3 = 1000
        self.water_specific_heat_kj_per_kg_k = 4.184
        self.tes_loss_rate_per_hour = 0.005
        # self.tes_room_temp_deg = 20.0 # Kept for reference, but calculations use kWh
        self.tes_minimum_kwh = 1072.0 # This in KWh for 36m2 tank is 40deg celsius.
        self.tes_physical_max_kwh = 2090.0 # keeping max at 70 deg

        # self.tes_capacity = self.tes_physical_max_kwh - self.tes_minimum_kwh # Usable capacity range
        self.tes_energy_kwh = 0 # Initial energy level (relative to a baseline, needs to reach tes_minimum_kwh)

        # District heating pricing (SEK per MWh)
        self.district_heating_prices = {
            'jan_dec': 1200,
            'feb_mar_nov': 863,
            'summer': 0 # No district heating cost in summer (Apr-Oct)
        }

        # Tracking results
        self.results = []
        self.total_solar_thermal_generation_kwh = 0
        self.total_solar_thermal_to_tes_kwh = 0
        self.total_solar_thermal_curtailed_kwh = 0
        self.total_hot_water_demand_kwh = 0
        self.total_tes_to_demand_kwh = 0
        self.total_grid_to_hp_kwh = 0 # Electricity from grid to HP
        self.total_grid_electricity_cost_sek = 0
        # self.total_baseline_electricity_cost_sek = 0 # Baseline will be calculated in output_analysis

        # District heating tracking
        self.total_district_heating_kwh = 0
        self.total_district_heating_cost_sek = 0

        # Location: Stockholm, Sweden (same as FullModel_FINAL.py)
        self.latitude_deg = 59.33
        self.longitude_deg = 18.06
        self.altitude = 44
        self.timezone = 'Europe/Stockholm'
        self.site_location = location.Location( # Renamed from 'site' to avoid conflict if pvlib.location is also named 'site'
            latitude=self.latitude_deg,
            longitude=self.longitude_deg,
            tz=self.timezone,
            altitude=self.altitude,
            name='Stockholm'
        )

    def get_district_heating_price(self, month):
        if month in [1, 12]:
            return self.district_heating_prices['jan_dec'] / 1000
        elif month in [2, 3, 11]:
            return self.district_heating_prices['feb_mar_nov'] / 1000
        else: # Apr, May, Jun, Jul, Aug, Sep, Oct
            return self.district_heating_prices['summer'] # Which is 0

    def is_summer_month(self, month):
        # Summer defined as May-September for HP usage instead of DH
        # DH pricing has summer from Apr-Oct (where price is 0)
        # For HP vs DH decision, sticking to FullModel_FINAL's May-Sep
        return month in [5, 6, 7, 8, 9]

    def calculate_solar_thermal_generation(self, dni_wh_per_m2_hour, dhi_wh_per_m2_hour, timestamp):
        """Calculate solar thermal generation using DNI, DHI, and timestamp."""
        times = pd.DatetimeIndex([timestamp])
        solar_pos = solarposition.get_solarposition(times, self.site_location.latitude, self.site_location.longitude)

        if solar_pos['zenith'].iloc[0] >= 90 or dni_wh_per_m2_hour <= 0:
            return 0.0

        solar_zenith_deg = solar_pos['apparent_zenith'].iloc[0]
        # GHI is needed for get_total_irradiance if not directly provided by it based on DNI,DHI,Zenith
        ghi_wh_per_m2_hour = dhi_wh_per_m2_hour + dni_wh_per_m2_hour * np.cos(np.radians(solar_zenith_deg))

        total_q_thermal_kwh = 0

        for config in self.collector_configs:
            poa = irradiance.get_total_irradiance(
                surface_tilt=config['tilt'],
                surface_azimuth=config['azimuth'],
                dni=dni_wh_per_m2_hour,
                ghi=ghi_wh_per_m2_hour, # Pass calculated GHI
                dhi=dhi_wh_per_m2_hour,
                solar_zenith=solar_pos['apparent_zenith'], # Series expected
                solar_azimuth=solar_pos['azimuth']      # Series expected
            )
            # poa_global is a series, get the first (only) value
            incident_irradiance_on_poa = poa['poa_global'].iloc[0] if not poa.empty and not pd.isna(poa['poa_global'].iloc[0]) else 0

            q_collector_wh = config['efficiency'] * config['area_m2'] * config['count'] * incident_irradiance_on_poa
            total_q_thermal_kwh += q_collector_wh / 1000 # Convert Wh to kWh

        return max(0, total_q_thermal_kwh)

    def simulate_hour(self, timestamp, hot_water_demand_kwh_hour, dni_wh_per_m2_hour, dhi_wh_per_m2_hour, electricity_price_sek_per_kwh):
        dt_hours = 1.0
        month = timestamp.month

        # 1. TES Standby Losses
        tes_loss_kwh = self.tes_energy_kwh * self.tes_loss_rate_per_hour * dt_hours
        self.tes_energy_kwh = max(0, self.tes_energy_kwh - tes_loss_kwh)

        # 2. Meet hot water demand from TES first
        tes_to_demand_kwh = min(hot_water_demand_kwh_hour, self.tes_energy_kwh)
        self.tes_energy_kwh -= tes_to_demand_kwh
        self.total_hot_water_demand_kwh += hot_water_demand_kwh_hour
        self.total_tes_to_demand_kwh += tes_to_demand_kwh
        
        unmet_demand_after_tes_kwh = hot_water_demand_kwh_hour - tes_to_demand_kwh

        # 3. Calculate Solar Thermal Generation
        q_solar_thermal_kwh = self.calculate_solar_thermal_generation(dni_wh_per_m2_hour, dhi_wh_per_m2_hour, timestamp)
        self.total_solar_thermal_generation_kwh += q_solar_thermal_kwh

        # 4. Add Solar Thermal to TES
        solar_thermal_to_tes_kwh = min(q_solar_thermal_kwh, self.tes_physical_max_kwh - self.tes_energy_kwh)
        self.tes_energy_kwh += solar_thermal_to_tes_kwh
        self.total_solar_thermal_to_tes_kwh += solar_thermal_to_tes_kwh
        
        solar_thermal_curtailed_kwh = q_solar_thermal_kwh - solar_thermal_to_tes_kwh
        self.total_solar_thermal_curtailed_kwh += solar_thermal_curtailed_kwh

        # 5. Handle remaining unmet demand and TES minimum energy level
        grid_to_hp_kwh = 0
        thermal_from_grid_hp_kwh = 0 # Renamed for clarity
        district_heating_kwh = 0
        district_heating_cost_sek = 0

        # Calculate thermal energy needed to cover remaining unmet demand AND ensure TES is at minimum
        thermal_needed_for_minimum_tes = max(0, self.tes_minimum_kwh - self.tes_energy_kwh)
        total_thermal_needed = unmet_demand_after_tes_kwh + thermal_needed_for_minimum_tes
        
        if total_thermal_needed > 0:
            if self.is_summer_month(month): # May - Sep: Use Grid HP
                # Electrical energy needed by HP
                grid_to_hp_kwh = total_thermal_needed / self.heat_pump_cop
                
                # Enforce minimum HP electrical input if HP runs
                if 0 < grid_to_hp_kwh < self.hp_min_electrical_input_kw:
                    grid_to_hp_kwh = self.hp_min_electrical_input_kw
                
                thermal_from_grid_hp_kwh = grid_to_hp_kwh * self.heat_pump_cop
                
                # Add to TES (ensure not to overfill, though total_thermal_needed should account for this via tes_physical_max_kwh indirectly)
                actual_thermal_added_from_grid_hp = min(thermal_from_grid_hp_kwh, self.tes_physical_max_kwh - self.tes_energy_kwh)
                self.tes_energy_kwh += actual_thermal_added_from_grid_hp
                # If actual_thermal_added_from_grid_hp < thermal_from_grid_hp_kwh, it means TES filled up exactly to max.
                # Adjust grid_to_hp_kwh if less thermal was added than planned due to TES max capacity.
                if thermal_from_grid_hp_kwh > 0 : # to avoid division by zero if thermal_from_grid_hp_kwh was 0
                    grid_to_hp_kwh = (actual_thermal_added_from_grid_hp / thermal_from_grid_hp_kwh) * grid_to_hp_kwh


                self.total_grid_to_hp_kwh += grid_to_hp_kwh
                self.total_grid_electricity_cost_sek += grid_to_hp_kwh * electricity_price_sek_per_kwh
            
            else: # Non-summer months (Oct - Apr according to is_summer_month def): Use District Heating
                district_heating_kwh = total_thermal_needed # DH directly provides thermal energy
                dh_price_sek_per_kwh = self.get_district_heating_price(month)
                district_heating_cost_sek = district_heating_kwh * dh_price_sek_per_kwh
                
                # Add to TES (ensure not to overfill)
                actual_thermal_added_from_dh = min(district_heating_kwh, self.tes_physical_max_kwh - self.tes_energy_kwh)
                self.tes_energy_kwh += actual_thermal_added_from_dh
                # Adjust district_heating_kwh and cost if less thermal was added
                if district_heating_kwh > 0:
                    cost_adjustment_factor = actual_thermal_added_from_dh / district_heating_kwh
                    district_heating_kwh = actual_thermal_added_from_dh
                    district_heating_cost_sek *= cost_adjustment_factor


                self.total_district_heating_kwh += district_heating_kwh
                self.total_district_heating_cost_sek += district_heating_cost_sek
        
        # Ensure TES energy does not exceed physical maximum (should be handled by min functions above)
        self.tes_energy_kwh = min(self.tes_energy_kwh, self.tes_physical_max_kwh)

        # Determine operating modes
        modes = []
        if solar_thermal_to_tes_kwh > 0: modes.append("Solar Thermal to TES")
        if grid_to_hp_kwh > 0: modes.append("Grid to HP")
        if district_heating_kwh > 0: modes.append("District Heating")
        if tes_to_demand_kwh > 0: modes.append("TES to Demand")
        if solar_thermal_curtailed_kwh > 1e-3: modes.append("Solar Thermal Curtailed")
        
        if not modes: modes.append("Standby")
        operating_mode_str = " & ".join(sorted(list(set(modes))))
        
        tes_soc_percent = (self.tes_energy_kwh / self.tes_physical_max_kwh) * 100 if self.tes_physical_max_kwh > 0 else 0
        tes_soc_percent = max(0, min(100, tes_soc_percent))
        
        self.results.append({
            'timestamp': timestamp,
            'hot_water_demand_kwh': hot_water_demand_kwh_hour,
            'dni_wh_per_m2_hour': dni_wh_per_m2_hour,
            'dhi_wh_per_m2_hour': dhi_wh_per_m2_hour,
            'solar_thermal_generation_kwh': q_solar_thermal_kwh,
            'solar_thermal_to_tes_kwh': solar_thermal_to_tes_kwh,
            'solar_thermal_curtailed_kwh': solar_thermal_curtailed_kwh,
            'grid_to_hp_kwh': grid_to_hp_kwh, # Electrical energy to HP from grid
            'thermal_from_grid_hp_kwh': thermal_from_grid_hp_kwh, # Thermal energy from HP (grid powered)
            'district_heating_kwh': district_heating_kwh,
            'district_heating_cost_sek': district_heating_cost_sek,
            'hp_electrical_consumed_kwh': grid_to_hp_kwh, # Total HP electrical consumption (only grid here)
            'tes_to_demand_kwh': tes_to_demand_kwh,
            'tes_energy_kwh': self.tes_energy_kwh,
            'tes_soc_percent': tes_soc_percent,
            'electricity_price_sek_per_kwh': electricity_price_sek_per_kwh,
            'operating_mode': operating_mode_str,
            'tes_loss_kwh': tes_loss_kwh,
        })

    def run_simulation(self, demand_values, dni_values, dhi_values, price_values):
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        # Assuming 2024 is a leap year for 8784 hours
        time_index = [start_date + timedelta(hours=i) for i in range(len(demand_values))] # Use length of input data

        # Reset simulation state using current parameters
        # Re-call __init__ with the originally passed parameters for a clean run
        # This specific way of re-init might need adjustment based on how parameters are stored if changed post-init
        # For now, assuming parameters like tes_volume_m3, heat_pump_cop, hp_min_electrical_input_kw are fixed for the instance lifetime
        # If run_simulation is called multiple times on the same instance for sensitivity, this reset is crucial.
        # A better way for full reset is:
        current_tes_volume = self.tes_volume_m3
        current_hp_cop = self.heat_pump_cop
        current_hp_min_input = self.hp_min_electrical_input_kw
        self.__init__(current_tes_volume, current_hp_cop, current_hp_min_input)


        for i in range(len(demand_values)): # Iterate for 8784 hours
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
        print("\n--- Annual System Performance Analysis (Solar Thermal) ---")
        print(f"Total Solar Thermal Energy Generated Annually: {self.total_solar_thermal_generation_kwh:.2f} kWh")
        print(f"Total Solar Thermal Energy Stored in TES: {self.total_solar_thermal_to_tes_kwh:.2f} kWh")
        if self.total_solar_thermal_generation_kwh > 0:
            curtailment_percentage = (self.total_solar_thermal_curtailed_kwh / self.total_solar_thermal_generation_kwh) * 100
            utilization_rate = (self.total_solar_thermal_to_tes_kwh / self.total_solar_thermal_generation_kwh) * 100
            print(f"Solar Thermal Curtailment (losses): {self.total_solar_thermal_curtailed_kwh:.2f} kWh ({curtailment_percentage:.1f}%)")
            print(f"Solar Thermal Utilization Rate (to TES): {utilization_rate:.1f}%")
        else:
            print("No solar thermal energy generated.")

        print(f"\nTotal Hot Water Demand: {self.total_hot_water_demand_kwh:.2f} kWh")
        print(f"Total Demand Met by TES: {self.total_tes_to_demand_kwh:.2f} kWh")
        if self.total_hot_water_demand_kwh > 0:
             print(f"Overall percentage of demand met by TES: {(self.total_tes_to_demand_kwh/self.total_hot_water_demand_kwh)*100:.1f}%")
        
        unmet_demand_total = self.total_hot_water_demand_kwh - self.total_tes_to_demand_kwh # This is demand not met by TES *after* all sources contributed to TES
        # Note: 'unmet_demand_after_tes_kwh' in simulate_hour is before ST/HP/DH for the hour.
        # The true system unmet demand would be if tes_to_demand was less than demand_kwh and HP/DH couldn't cover it.
        # The current logic aims to always meet demand + TES min via HP/DH, so unmet should be near zero unless TES max capacity is limiting.
        # Let's re-evaluate "unmet". If demand is always met by TES, which is topped up, then unmet should be 0.
        # The calculation `unmet_demand_after_tes_kwh` is an intermediate step.
        # A better measure of "unmet" would be if `total_thermal_needed` could not be fully satisfied.
        # For now, we assume the system is sized to meet demand if energy is available.

        print(f"\n--- Energy Consumption & Costs ---")
        print(f"Total Grid Electricity Used for HP: {self.total_grid_to_hp_kwh:.2f} kWh")
        print(f"Total Grid Electricity Cost for HP: {self.total_grid_electricity_cost_sek:.2f} SEK")
        
        print(f"Total District Heating Used: {self.total_district_heating_kwh:.2f} kWh")
        print(f"Total District Heating Cost: {self.total_district_heating_cost_sek:.2f} SEK")

        # Baseline cost calculation (District Heating Only for all demand)
        # Weighted average DH price (example if needed, or simplified sum)
        # This baseline matches the FullModel_FINAL.py approach
        baseline_dh_cost = 0
        # Simple sum of what DH would have cost for the demand each month (excluding summer price = 0)
        # This requires monthly demand totals. For simplicity, using FullModel_FINAL's annual proration:
        # Prorating annual demand by price categories.
        # (2 months high price, 3 months mid price, 7 months zero price for DH)
        if self.total_hot_water_demand_kwh > 0 :
            baseline_dh_cost = (
                (self.total_hot_water_demand_kwh * (2/12)) * (self.district_heating_prices['jan_dec'] / 1000) +
                (self.total_hot_water_demand_kwh * (3/12)) * (self.district_heating_prices['feb_mar_nov'] / 1000) +
                (self.total_hot_water_demand_kwh * (7/12)) * (self.district_heating_prices['summer'] / 1000) # summer price is 0
            )
        
        total_system_operational_cost = self.total_grid_electricity_cost_sek + self.total_district_heating_cost_sek
        annual_savings = baseline_dh_cost - total_system_operational_cost
        
        print(f"\nBaseline Annual Cost (District Heating Only for all demand): {baseline_dh_cost:.2f} SEK")
        print(f"Actual System Operational Cost (Grid HP + District Heating): {total_system_operational_cost:.2f} SEK")
        print(f"Annual Operational Savings with Solar Thermal System: {annual_savings:.2f} SEK")

        if hasattr(self, 'results_df') and not self.results_df.empty:
            print("\n--- Annual Operating Mode Distribution ---")
            annual_mode_counts = self.results_df['operating_mode'].value_counts()
            annual_mode_percentages = (annual_mode_counts / len(self.results_df)) * 100
            print(annual_mode_percentages.round(1).to_string())

            # Summer analysis (May-September)
            summer_months_ops = [5, 6, 7, 8, 9] # For HP operation vs DH
            summer_data = self.results_df[self.results_df.index.month.isin(summer_months_ops)]

            if not summer_data.empty:
                summer_total_demand = summer_data['hot_water_demand_kwh'].sum()
                summer_solar_thermal_gen = summer_data['solar_thermal_generation_kwh'].sum()
                summer_solar_thermal_to_tes = summer_data['solar_thermal_to_tes_kwh'].sum()
                summer_grid_to_hp = summer_data['grid_to_hp_kwh'].sum()
                summer_solar_thermal_curtailed = summer_data['solar_thermal_curtailed_kwh'].sum()
                summer_grid_cost = (summer_data['grid_to_hp_kwh'] * summer_data['electricity_price_sek_per_kwh']).sum()
                
                # Summer self-sufficiency: (Thermal demand met by solar thermal) / Total summer thermal demand
                # Assume solar_thermal_to_tes contributes to meeting demand.
                summer_self_sufficiency = 0
                if summer_total_demand > 0:
                    # A portion of tes_to_demand_kwh comes from solar_thermal_to_tes_kwh.
                    # Simplification: what fraction of demand was covered by ST (directly or via TES) vs Grid HP
                    # Energy provided by ST to TES is a good proxy for ST contribution
                    summer_self_sufficiency = (summer_solar_thermal_to_tes / summer_total_demand) * 100
                    summer_self_sufficiency = min(summer_self_sufficiency, 100) # Cap at 100%

                print("\n--- Summer System Performance Analysis (May-September) ---")
                print(f"Total summer hot water energy demand (thermal): {summer_total_demand:.2f} kWh")
                print(f"Solar Thermal Generated: {summer_solar_thermal_gen:.2f} kWh")
                print(f"Solar Thermal to TES: {summer_solar_thermal_to_tes:.2f} kWh")
                print(f"Summer Self-Sufficiency (from Solar Thermal): {summer_self_sufficiency:.2f}%")
                print(f"Grid Electricity to HP (Summer): {summer_grid_to_hp:.2f} kWh")
                if summer_solar_thermal_gen > 0:
                    summer_curtailment_perc = (summer_solar_thermal_curtailed / summer_solar_thermal_gen) * 100
                    print(f"Summer Solar Thermal Curtailment: {summer_solar_thermal_curtailed:.2f} kWh ({summer_curtailment_perc:.1f}%)")
                else:
                    print(f"Summer Solar Thermal Curtailment: {summer_solar_thermal_curtailed:.2f} kWh")
                print(f"Electricity Cost (Summer, from Grid for HP): {summer_grid_cost:.2f} SEK")
            else:
                print("\nNo data for summer months (May-September) analysis.")

            # Non-summer analysis (Oct-Apr)
            non_summer_months_ops = [1, 2, 3, 4, 10, 11, 12]
            non_summer_data = self.results_df[self.results_df.index.month.isin(non_summer_months_ops)]
            
            if not non_summer_data.empty:
                non_summer_total_demand = non_summer_data['hot_water_demand_kwh'].sum()
                non_summer_solar_thermal_gen = non_summer_data['solar_thermal_generation_kwh'].sum()
                non_summer_solar_thermal_to_tes = non_summer_data['solar_thermal_to_tes_kwh'].sum()
                non_summer_district_heating = non_summer_data['district_heating_kwh'].sum()
                non_summer_district_heating_cost = non_summer_data['district_heating_cost_sek'].sum()
                non_summer_solar_thermal_curtailed = non_summer_data['solar_thermal_curtailed_kwh'].sum()

                non_summer_self_sufficiency = 0
                if non_summer_total_demand > 0:
                    non_summer_self_sufficiency = (non_summer_solar_thermal_to_tes / non_summer_total_demand) * 100
                    non_summer_self_sufficiency = min(non_summer_self_sufficiency, 100)

                print("\n--- Non-Summer System Performance Analysis (Oct-Apr) ---")
                print(f"Total non-summer hot water energy demand (thermal): {non_summer_total_demand:.2f} kWh")
                print(f"Solar Thermal Generated: {non_summer_solar_thermal_gen:.2f} kWh")
                print(f"Solar Thermal to TES: {non_summer_solar_thermal_to_tes:.2f} kWh")
                print(f"Non-Summer Self-Sufficiency (from Solar Thermal): {non_summer_self_sufficiency:.2f}%")
                print(f"District Heating Used: {non_summer_district_heating:.2f} kWh")
                print(f"District Heating Cost: {non_summer_district_heating_cost:.2f} SEK")
                if non_summer_solar_thermal_gen > 0:
                    non_summer_curtailment_perc = (non_summer_solar_thermal_curtailed / non_summer_solar_thermal_gen) * 100
                    print(f"Non-Summer Solar Thermal Curtailment: {non_summer_solar_thermal_curtailed:.2f} kWh ({non_summer_curtailment_perc:.1f}%)")
                else:
                    print(f"Non-Summer Solar Thermal Curtailment: {non_summer_solar_thermal_curtailed:.2f} kWh")
            else:
                print("\nNo data for non-summer months (Oct-Apr) analysis.")
        else:
            print("No results data frame to analyze.")


    def plot_summer_operating_modes(self):
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No data to plot summer operating modes.")
            return
        summer_months = [5, 6, 7, 8, 9]
        summer_data = self.results_df[self.results_df.index.month.isin(summer_months)]
        if summer_data.empty:
            print("No summer data to plot operating modes.")
            return
            
        mode_counts = summer_data['operating_mode'].value_counts()
        if mode_counts.empty:
            print("No operating modes recorded in summer data.")
            return

        plt.figure(figsize=(10, 8))
        plt.pie(mode_counts, labels=mode_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Summer Operating Mode Distribution (May-September)')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def plot_annual_operating_modes(self):
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No data to plot annual operating modes.")
            return
        mode_counts = self.results_df['operating_mode'].value_counts()
        if mode_counts.empty:
            print("No operating modes recorded in annual data.")
            return

        plt.figure(figsize=(12, 8))
        plt.pie(mode_counts, labels=mode_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Annual Operating Mode Distribution')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def plot_weekly_performance(self, week_number):
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print(f"No simulation results to plot week {week_number}.")
            return
        
        weekly_data = self.results_df[self.results_df.index.isocalendar().week == week_number].copy()
        if weekly_data.empty:
            print(f"No data found for week {week_number}")
            return

        fig, ax1 = plt.subplots(figsize=(15, 7))
        time_index_weekly = weekly_data.index
        hours_in_week = np.arange(len(weekly_data))

        # Plotting thermal demand directly
        ax1.plot(hours_in_week, weekly_data['hot_water_demand_kwh'], label='Hot Water Demand (kWh thermal)', color='blue', linestyle='--')
        ax1.plot(hours_in_week, weekly_data['solar_thermal_generation_kwh'], label='Solar Thermal Generation (kWh thermal)', color='green')
        ax1.plot(hours_in_week, weekly_data['grid_to_hp_kwh'], label='Grid to HP (kWh elec)', color='red') # This is 'grid_total_usage_kwh_hour' effectively
        ax1.plot(hours_in_week, weekly_data['district_heating_kwh'], label='District Heating (kWh thermal)', color='brown')
        ax1.plot(hours_in_week, weekly_data['hp_electrical_consumed_kwh'], label='HP Electrical Input (kWh elec)', color='purple', linestyle=':')
        ax1.plot(hours_in_week, weekly_data['solar_thermal_curtailed_kwh'], label='Solar Thermal Curtailed (kWh thermal)', color='orange', linestyle='-.')
        
        ax1.set_xlabel('Hour in Week')
        ax1.set_ylabel('Energy (kWh or kW for elec components if applicable)')
        ax1.tick_params(axis='y')
        
        ax2 = ax1.twinx()
        ax2.plot(hours_in_week, weekly_data['tes_soc_percent'], label='TES SoC (%)', color='darkred', linestyle='-')
        ax2.set_ylabel('TES State of Charge (%)')
        ax2.set_ylim(0, 105) # Allow slight overfill visually if it happens due to rounding
        ax2.tick_params(axis='y')

        tes_minimum_soc_percent = (self.tes_minimum_kwh / self.tes_physical_max_kwh) * 100
        ax2.axhline(
            tes_minimum_soc_percent,
            color='goldenrod',
            linestyle=(0, (5, 5)), # Dashed line
            linewidth=2,
            label=f'Minimum TES energy ({self.tes_minimum_kwh:.0f} kWh)'
        )

        xticks_pos = hours_in_week[::24] # Tick every 24 hours
        xticks_labels = [time_index_weekly[i].strftime('%Y-%m-%d (%a)') for i in xticks_pos if i < len(time_index_weekly)]
        if len(xticks_pos) > len(xticks_labels) : xticks_pos = xticks_pos[:len(xticks_labels)] # Ensure lists are same length

        ax1.set_xticks(xticks_pos)
        ax1.set_xticklabels(xticks_labels, rotation=45, ha='right')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title(f'Energy System Performance (Solar Thermal) - Week {week_number}')
        plt.tight_layout()
        plt.show()

    def plot_monthly_energy_flow(self):
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No data to plot monthly energy flow.")
            return

        monthly_agg = {
            'hot_water_demand_kwh': 'sum',
            'solar_thermal_generation_kwh': 'sum',
            'solar_thermal_to_tes_kwh': 'sum',
            'grid_to_hp_kwh': 'sum',
            'thermal_from_grid_hp_kwh': 'sum', # Thermal output of HP
            'district_heating_kwh': 'sum',
            'district_heating_cost_sek': 'sum',
            'solar_thermal_curtailed_kwh': 'sum'
            # Add electricity cost for HP if needed for cost plot
        }
        # Add electricity price to calculate cost if not directly summed in results
        # For cost plot, we'll use grid_to_hp_kwh and an average price or sum hourly costs if available
        # self.results_df already has electricity_price_sek_per_kwh
        
        # Calculate monthly grid electricity cost for HP
        monthly_results_df = self.results_df.copy()
        monthly_results_df['grid_hp_cost_sek'] = monthly_results_df['grid_to_hp_kwh'] * monthly_results_df['electricity_price_sek_per_kwh']
        monthly_agg['grid_hp_cost_sek'] = 'sum'

        monthly_data = monthly_results_df.groupby(monthly_results_df.index.month).agg(monthly_agg)
        
        months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Ensure monthly_data is indexed 1-12 and reindex if necessary for full year plot
        monthly_data = monthly_data.reindex(range(1, 13)) # Fill missing months with NaN
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(17, 14)) # Increased size
        
        # Plot 1: Monthly Energy Generation and Demand
        ax1.bar(months_labels, monthly_data['hot_water_demand_kwh'].fillna(0), alpha=0.7, label='Hot Water Demand (Thermal)', color='blue')
        ax1.bar(months_labels, monthly_data['solar_thermal_generation_kwh'].fillna(0), alpha=0.7, label='Solar Thermal Generation (Thermal)', color='green')
        ax1.set_ylabel('Energy (kWh)')
        ax1.set_title('Monthly Energy Generation vs Demand')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Monthly Energy Sources for Hot Water (all thermal kWh)
        # Stacked bar: Solar Thermal to TES, Thermal from Grid HP, District Heating
        st_to_tes = monthly_data['solar_thermal_to_tes_kwh'].fillna(0)
        th_from_grid_hp = monthly_data['thermal_from_grid_hp_kwh'].fillna(0) # This is already thermal
        dh_kwh = monthly_data['district_heating_kwh'].fillna(0)

        ax2.bar(months_labels, st_to_tes, alpha=0.7, label='Solar Thermal to TES', color='green')
        ax2.bar(months_labels, th_from_grid_hp, bottom=st_to_tes, alpha=0.7, label='Thermal from Grid HP', color='red')
        ax2.bar(months_labels, dh_kwh, bottom=st_to_tes + th_from_grid_hp, alpha=0.7, label='District Heating', color='brown')
        ax2.set_ylabel('Thermal Energy Supplied (kWh)')
        ax2.set_title('Monthly Energy Sources for Hot Water / TES Charging')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Monthly Solar Thermal Utilization
        # Utilization = (Solar Thermal to TES) / (Solar Thermal Generation)
        utilization_num = monthly_data['solar_thermal_to_tes_kwh'].fillna(0)
        utilization_den = monthly_data['solar_thermal_generation_kwh'].fillna(0)
        solar_thermal_utilization = (utilization_num / utilization_den * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        ax3.bar(months_labels, solar_thermal_utilization, alpha=0.7, color='orange')
        ax3.set_ylabel('Solar Thermal Utilization (%)')
        ax3.set_title('Monthly Solar Thermal Utilization Rate (Energy to TES / Generated)')
        ax3.set_ylim(0, 100)
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Monthly Costs
        dh_costs = monthly_data['district_heating_cost_sek'].fillna(0)
        grid_hp_costs = monthly_data['grid_hp_cost_sek'].fillna(0) # Calculated above

        ax4.bar(months_labels, grid_hp_costs, alpha=0.7, label='Grid Electricity Cost (HP)', color='red')
        ax4.bar(months_labels, dh_costs, bottom=grid_hp_costs, alpha=0.7, label='District Heating Cost', color='brown')
        ax4.set_ylabel('Cost (SEK)')
        ax4.set_title('Monthly Energy Costs')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


# Main function adapted for Solar Thermal Simulation
def run_solar_thermal_simulation_process(demand_file, dni_dhi_file, price_file, 
                                         tes_volume_m3, heat_pump_cop, 
                                         hp_min_electrical_input_kw, 
                                         plot_weekly=False, week_number=None, 
                                         plot_monthly=True):
    
    try:
        demand_df = pd.read_csv(demand_file)
        dni_dhi_df = pd.read_csv(dni_dhi_file) 
        price_df = pd.read_csv(price_file)
    except FileNotFoundError as e:
        print(f"Error loading input file: {e}")
        return None

    # Basic validation of data length (expecting 8784 for a leap year like 2024)
    # Or more generally, ensure all inputs have the same length
    if not (len(demand_df) == len(dni_dhi_df) == len(price_df)):
        print("Error: Input CSV files have inconsistent lengths.")
        return None
    if len(demand_df) != 8784 and len(demand_df) != 8760 : # common year lengths
        print(f"Warning: Input data length is {len(demand_df)}, not a typical full year (8760 or 8784 hours). Simulation will run for available data.")


    demand_values = demand_df['energy demand (kWh)']
    dni_values = dni_dhi_df['ALLSKY_SFC_SW_DNI']    
    dhi_values = dni_dhi_df['ALLSKY_SFC_SW_DHI']    
    price_values = price_df['price']            

    simulation = SolarThermalEnergySimulation( # Use the new class
        tes_volume_m3=tes_volume_m3,
        heat_pump_cop=heat_pump_cop,
        hp_min_electrical_input_kw=hp_min_electrical_input_kw
    )

    print("Starting Solar Thermal Simulation...")
    simulation.run_simulation(demand_values, dni_values, dhi_values, price_values)
    print("Simulation Complete. Generating Analysis...")
    simulation.output_analysis()
    
    if hasattr(simulation, 'results_df') and not simulation.results_df.empty:
        simulation.plot_summer_operating_modes()
        simulation.plot_annual_operating_modes()

        if plot_monthly:
            simulation.plot_monthly_energy_flow()

        if plot_weekly and week_number is not None:
            try:
                week_number_int = int(week_number)
                if 1 <= week_number_int <= 53:
                    simulation.plot_weekly_performance(week_number_int)
                else:
                    print(f"Invalid week number: {week_number}. Must be between 1 and 53.")
            except ValueError:
                print(f"Invalid week number format: {week_number}. Must be an integer.")
    else:
        print("Skipping plots as no simulation results are available.")
        
    return simulation

if __name__ == "__main__":
    # Define file paths (ensure these files are in the same directory as the script or provide full paths)
    demand_file = 'energy_demand.csv'   
    dni_dhi_file = 'dni_dhi.csv' # Changed from dni_file to reflect content        
    price_file = 'electricity_price.csv' 

    # System parameters from FullModel_FINAL.py main example
    tes_volume_m3_main = 36.0
    heat_pump_cop_main = 0.95
    # pv_panel_capacity_kwp is not needed for solar thermal model
    hp_min_electrical_input_kw_main = 5.0 # kW

    print(f"Attempting to load data from: \nDemand: {demand_file}\nDNI/DHI: {dni_dhi_file}\nPrice: {price_file}")

    simulation_result_st = run_solar_thermal_simulation_process(
        demand_file,
        dni_dhi_file, # Corrected variable name
        price_file,
        tes_volume_m3=tes_volume_m3_main,
        heat_pump_cop=heat_pump_cop_main,
        hp_min_electrical_input_kw=hp_min_electrical_input_kw_main,
        plot_weekly=True,
        week_number=22, # Example week
        plot_monthly=True
    )

    if simulation_result_st is None:
        print("Simulation process failed to run.")
    else:
        print("Solar Thermal Simulation Process Finished.")