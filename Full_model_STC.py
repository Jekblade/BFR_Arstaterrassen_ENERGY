import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
import warnings
from pvlib import location, irradiance, solarposition

warnings.filterwarnings("ignore", message="The get_cmap function is deprecated")

# Load input data from CSV files
demand_df = pd.read_csv("energy_demand.csv")
dni_df = pd.read_csv("Dni.csv")  # Should include columns 'ALLSKY_SFC_SW_DNI' and 'ALLSKY_SFC_SW_DHI'
price_df = pd.read_csv("electricity_price.csv")

demand_values = demand_df['energy demand (kWh)']
dni_values = dni_df['ALLSKY_SFC_SW_DNI']
dhi_values = dni_df['ALLSKY_SFC_SW_DHI']
price_values = price_df['price']

class BuildingEnergySimulation:
    def __init__(self, tes_volume_m3, heat_pump_cop, hp_min_electrical_input_kw=10):
        self.heat_pump_cop = heat_pump_cop
        self.hp_min_electrical_input_kw = hp_min_electrical_input_kw
        self.tes_volume_m3 = tes_volume_m3
        self.water_density_kg_per_m3 = 1000
        self.water_specific_heat_kj_per_kg_k = 4.184
        self.tes_loss_rate_per_hour = 0.005
        self.tes_target_maintenance_kwh = 1272.0
        self.tes_physical_max_kwh = 2090.0
        self.tes_energy_kwh = self.tes_target_maintenance_kwh
        self.results = []
        self.total_solar_thermal_generation_kwh = 0
        self.total_solar_thermal_to_tes_kwh = 0
        self.total_hot_water_demand_kwh = 0
        self.total_tes_to_demand_kwh = 0
        self.total_grid_to_hp_kwh = 0
        self.total_grid_electricity_cost_sek = 0
        self.total_baseline_electricity_cost_sek = 0
        self.unmet_demand_kwh = 0

    def calculate_solar_thermal_generation(self, dni_wh_per_m2_hour, dhi_wh_per_m2_hour, timestamp):
        site = location.Location(59.33, 18.06, tz='Europe/Stockholm')
        times = pd.DatetimeIndex([timestamp])
        solar_pos = solarposition.get_solarposition(times, site.latitude, site.longitude)

        if solar_pos['zenith'].iloc[0] >= 90 or dni_wh_per_m2_hour <= 0:
            return 0.0

        solar_zenith_deg = solar_pos['apparent_zenith'].iloc[0]
        ghi = dhi_wh_per_m2_hour + dni_wh_per_m2_hour * np.cos(np.radians(solar_zenith_deg))
        dhi = dhi_wh_per_m2_hour

        collector_configs = [
            {'tilt': 27, 'azimuth': 185, 'count': 200},
            {'tilt': 25, 'azimuth': 185, 'count': 0}]
    

        collector_efficiency = 0.65
        collector_area_m2 = 2.35
        total_q_kwh = 0

        for config in collector_configs:
            poa = irradiance.get_total_irradiance(
                surface_tilt=config['tilt'],
                surface_azimuth=config['azimuth'],
                dni=dni_wh_per_m2_hour,
                ghi=ghi,
                dhi=dhi,
                solar_zenith=solar_pos['apparent_zenith'],
                solar_azimuth=solar_pos['azimuth']
            )
            dni_adjusted_on_poa = poa['poa_global'].iloc[0] if not poa.empty else 0
            q_solar_wh = collector_efficiency * collector_area_m2 * config['count'] * dni_adjusted_on_poa
            total_q_kwh += q_solar_wh / 1000

        return max(0, total_q_kwh)

    def simulate_hour(self, timestamp, demand_kwh, dni, dhi, price):
        dt = 1.0
        tes_loss = self.tes_energy_kwh * self.tes_loss_rate_per_hour * dt
        self.tes_energy_kwh = max(0, self.tes_energy_kwh - tes_loss)

        thermal_needed = max(0, self.tes_target_maintenance_kwh - self.tes_energy_kwh)
        hp_electricity = 0
        hp_thermal = 0

        if thermal_needed > 0:
            hp_electricity = thermal_needed / self.heat_pump_cop
            if 0 < hp_electricity < self.hp_min_electrical_input_kw:
                hp_electricity = self.hp_min_electrical_input_kw
            hp_thermal = hp_electricity * self.heat_pump_cop
            added = min(hp_thermal, self.tes_physical_max_kwh - self.tes_energy_kwh)
            self.tes_energy_kwh += added
            self.total_grid_to_hp_kwh += hp_electricity
            self.total_grid_electricity_cost_sek += hp_electricity * price

        supplied = min(demand_kwh, self.tes_energy_kwh)
        self.tes_energy_kwh -= supplied
        unmet = demand_kwh - supplied

        self.total_hot_water_demand_kwh += demand_kwh
        self.total_tes_to_demand_kwh += supplied
        self.unmet_demand_kwh += unmet

        q_solar_kwh = self.calculate_solar_thermal_generation(dni, dhi, timestamp)
        self.total_solar_thermal_generation_kwh += q_solar_kwh
        thermal_from_solar = min(q_solar_kwh, self.tes_physical_max_kwh - self.tes_energy_kwh)
        self.tes_energy_kwh += thermal_from_solar
        self.total_solar_thermal_to_tes_kwh += thermal_from_solar

        

        modes = []
        if thermal_from_solar > 0: modes.append("Solar Collector to TES")
        if hp_electricity > 0: modes.append("Grid to HP")
        if supplied > 0: modes.append("TES to Demand")
        if q_solar_kwh - thermal_from_solar > 1e-3: modes.append("Solar Collector Excess/Curtailed")
        if not modes: modes.append("Standby")

        self.results.append({
            'timestamp': timestamp,
            'hot_water_demand_kwh': demand_kwh,
            'dni_wh_per_m2_hour': dni,
            'dhi_wh_per_m2_hour': dhi,
            'solar_thermal_generation_kwh': q_solar_kwh,
            'solar_thermal_to_tes_kwh': thermal_from_solar,
            'grid_to_hp_kwh': hp_electricity,
            'hp_electrical_consumed_kwh': hp_electricity,
            'thermal_added_by_hp_to_tes_kwh': hp_thermal,
            'tes_to_demand_kwh': supplied,
            'unmet_demand_this_hour_kwh': unmet,
            'tes_energy_kwh': self.tes_energy_kwh,
            'tes_soc_percent': (self.tes_energy_kwh / self.tes_physical_max_kwh) * 100,
            'electricity_price_sek_per_kwh': price,
            'operating_mode': " & ".join(sorted(set(modes))),
            'tes_loss_kwh': tes_loss
        })

    def run_simulation(self, demand_values, dni_values, dhi_values, price_values):
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        time_index = [start_date + timedelta(hours=i) for i in range(8784)]

        if not (len(demand_values) == len(dni_values) == len(dhi_values) == len(price_values) == 8784):
            raise ValueError("Input data series must contain 8784 hourly records")

        for i in range(8784):
            self.total_baseline_electricity_cost_sek += demand_values.iloc[i] * price_values.iloc[i]
            self.simulate_hour(
                time_index[i],
                demand_values.iloc[i],
                dni_values.iloc[i],
                dhi_values.iloc[i],
                price_values.iloc[i]
            )

        self.results_df = pd.DataFrame(self.results)
        self.results_df.set_index('timestamp', inplace=True)

    def output_analysis(self):
        print("\n--- Annual System Performance Analysis ---")
        print(f"Total Potential Solar Thermal Energy Generated: {self.total_solar_thermal_generation_kwh:.2f} kWh")
        print(f"Total Solar Thermal Energy Used by TES: {self.total_solar_thermal_to_tes_kwh:.2f} kWh")
        curtailed = ((self.total_solar_thermal_generation_kwh - self.total_solar_thermal_to_tes_kwh)/self.total_solar_thermal_generation_kwh)*100
        print(f"Total Solar Thermal Energy Curtailed/Excess: {curtailed:.2f} %")
        print(f"Total Grid Electricity Used by Boiler for TES: {self.total_grid_to_hp_kwh:.2f} kWh")
        print(f"Total Hot Water Demand: {self.total_hot_water_demand_kwh:.2f} kWh")
        print(f"Demand Met by TES: {self.total_tes_to_demand_kwh:.2f} kWh")
        print(f"Unmet Demand: {self.unmet_demand_kwh:.2f} kWh")
        if self.total_hot_water_demand_kwh > 0:
            print(f"Percentage of Demand Met: {(self.total_tes_to_demand_kwh / self.total_hot_water_demand_kwh) * 100:.2f}%")
        print(f"\nTotal Electricity Cost (baseline scenario - direct electric heating): {self.total_baseline_electricity_cost_sek:.2f} SEK")
        print(f"Total Electricity Cost (with system - HP from grid): {self.total_grid_electricity_cost_sek:.2f} SEK")
        print(f"Total Money Saved Annually: {self.total_baseline_electricity_cost_sek - self.total_grid_electricity_cost_sek:.2f} SEK")

        if hasattr(self, 'results_df') and not self.results_df.empty:
            print("\n--- Annual Operating Mode Distribution ---")
            if 'operating_mode' in self.results_df.columns:
                counts = self.results_df['operating_mode'].value_counts()
                percentages = (counts / len(self.results_df)) * 100
                print(percentages.round(1).to_string())
            else:
                print("Operating mode data not available.")

    def plot_weekly_performance(self, week_number):
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print(f"No simulation results to plot week {week_number}.")
            return

        weekly_data = self.results_df[self.results_df.index.isocalendar().week == week_number].copy()
        if weekly_data.empty:
            print(f"No data found for week {week_number}")
            return

        fig, ax1 = plt.subplots(figsize=(15, 7))
        hours_in_week = np.arange(len(weekly_data))

        ax1.plot(hours_in_week, weekly_data['hot_water_demand_kwh'],  color='blue', linestyle='--')
        ax1.plot(hours_in_week, weekly_data['solar_thermal_generation_kwh'], color='green')
        ax1.plot(hours_in_week, weekly_data['grid_to_hp_kwh'],  color='red')
        ax1.plot(hours_in_week, weekly_data['hp_electrical_consumed_kwh'], color='purple', linestyle=':')

        ax1.set_xlabel('Hour in Week')
        ax1.set_ylabel('Power / Energy (kWh)')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        ax2.plot(hours_in_week, weekly_data['tes_soc_percent'],  color='orange')
        ax2.set_ylabel('TES State of Charge (%)')
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis='y')

        ax1.set_xticks(hours_in_week[::24])
        xtick_labels_list = [weekly_data.index[i].strftime('%Y-%m-%d') for i in hours_in_week[::24] if i < len(weekly_data.index)]
        ax1.set_xticklabels(xtick_labels_list, rotation=45, ha='right')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title(f'Energy System Performance - Week {week_number}')
        plt.tight_layout()
        plt.show()

    def output_summer_analysis(self):
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("\nNo simulation results available for summer analysis.")
            return

        summer_months = [5, 6, 7, 8, 9]
        summer_data = self.results_df[self.results_df.index.month.isin(summer_months)]

        if summer_data.empty:
            print("\nNo data found for summer months (May to September).")
            return

        total_demand = summer_data['hot_water_demand_kwh'].sum()
        total_supplied = summer_data['tes_to_demand_kwh'].sum()
        total_unmet = summer_data['unmet_demand_this_hour_kwh'].sum()
        total_solar = summer_data['solar_thermal_generation_kwh'].sum()
        total_solar_to_tes = summer_data['solar_thermal_to_tes_kwh'].sum()
        total_grid_to_hp = summer_data['grid_to_hp_kwh'].sum()
        total_cost_with_hp = (summer_data['hp_electrical_consumed_kwh'] * summer_data['electricity_price_sek_per_kwh']).sum()
        baseline_cost = (summer_data['hot_water_demand_kwh'] * summer_data['electricity_price_sek_per_kwh']).sum()

        print("\n--- Summer System Performance Analysis (May–September) ---")
        print(f"Total Summer Hot Water Demand: {total_demand:.2f} kWh")
        print(f"Demand Met by TES: {total_supplied:.2f} kWh")
        print(f"Unmet Demand: {total_unmet:.2f} kWh")
        if total_demand > 0:
            print(f"Percentage of Demand Met: {(total_supplied / total_demand) * 100:.2f}%")
        print(f"Total Solar Thermal Generation: {total_solar:.2f} kWh")
        print(f"Solar Thermal Stored in TES: {total_solar_to_tes:.2f} kWh")
        curtailed = ((total_solar - total_solar_to_tes)/total_solar)*100
        print(f"Total Solar Thermal Energy Curtailed/Excess: {curtailed:.2f} %")
        print(f"Electricity Used by Boiler: {total_grid_to_hp:.2f} kWh")
        print(f"Total Electricity Cost (Boiler only): {total_cost_with_hp:.2f} SEK")
        print(f"Baseline Electricity Cost (direct electric heating): {baseline_cost:.2f} SEK")
        print(f"Total Money Saved in Summer: {64840 - total_cost_with_hp:.2f} SEK")


    def plot_summer_operating_modes(self):
        if not hasattr(self, 'results_df') or self.results_df.empty or 'operating_mode' not in self.results_df.columns:
            print("No operating mode data to plot for summer.")
            return
        summer_months = [5, 6, 7, 8, 9]
        summer_data = self.results_df[self.results_df.index.month.isin(summer_months)]
        if summer_data.empty or 'operating_mode' not in summer_data.columns:
            print("No data for summer months to plot operating modes.")
            return

        mode_counts = summer_data['operating_mode'].value_counts()
        if mode_counts.empty:
            print("No operating mode data for summer months to plot.")
            return

        plt.figure(figsize=(10, 8))
        plt.pie(mode_counts, labels=mode_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Summer Operating Mode Distribution (May-September)')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    simulation = BuildingEnergySimulation(tes_volume_m3=36.0, heat_pump_cop=0.95, hp_min_electrical_input_kw=10.0)
    simulation.run_simulation(demand_values, dni_values, dhi_values, price_values)
    simulation.output_analysis()
    simulation.output_summer_analysis() 
    simulation.plot_summer_operating_modes()
    simulation.plot_weekly_performance(36)
    simulation.plot_weekly_performance(2)


