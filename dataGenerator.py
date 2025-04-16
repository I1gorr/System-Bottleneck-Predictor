import csv
import random
from datetime import datetime, timedelta

# Process templates
process_templates = [
    {'name': 'gnome-shell', 'base_mem': 170000, 'mem_var': 50000, 'base_cpu': 3.0, 'cpu_var': 2.0, 'weight': 15},
    {'name': 'Xorg', 'base_mem': 80000, 'mem_var': 20000, 'base_cpu': 1.5, 'cpu_var': 1.0, 'weight': 5},
    {'name': 'android-studio', 'base_mem': 1500000, 'mem_var': 500000, 'base_cpu': 25.0, 'cpu_var': 15.0, 'weight': 20},
    {'name': 'code', 'base_mem': 400000, 'mem_var': 150000, 'base_cpu': 8.0, 'cpu_var': 5.0, 'weight': 18},
    {'name': 'emulator', 'base_mem': 2000000, 'mem_var': 800000, 'base_cpu': 40.0, 'cpu_var': 25.0, 'weight': 12},
    {'name': 'gradle', 'base_mem': 500000, 'mem_var': 200000, 'base_cpu': 15.0, 'cpu_var': 10.0, 'weight': 10},
    {'name': 'firefox', 'base_mem': 800000, 'mem_var': 400000, 'base_cpu': 10.0, 'cpu_var': 8.0, 'weight': 25},
    {'name': 'chrome', 'base_mem': 1000000, 'mem_var': 500000, 'base_cpu': 15.0, 'cpu_var': 10.0, 'weight': 25},
    {'name': 'steam', 'base_mem': 500000, 'mem_var': 300000, 'base_cpu': 20.0, 'cpu_var': 15.0, 'weight': 18},
    {'name': 'csgo_linux', 'base_mem': 1800000, 'mem_var': 600000, 'base_cpu': 50.0, 'cpu_var': 30.0, 'weight': 12},
    {'name': 'discord', 'base_mem': 300000, 'mem_var': 150000, 'base_cpu': 7.0, 'cpu_var': 5.0, 'weight': 15},
    {'name': 'spotify', 'base_mem': 200000, 'mem_var': 100000, 'base_cpu': 4.0, 'cpu_var': 3.0, 'weight': 10},
]

occasional_processes = [
    {'name': 'zoom', 'base_mem': 350000, 'mem_var': 150000, 'base_cpu': 12.0, 'cpu_var': 8.0},
    {'name': 'postman', 'base_mem': 250000, 'mem_var': 100000, 'base_cpu': 6.0, 'cpu_var': 4.0},
    {'name': 'minecraft', 'base_mem': 1200000, 'mem_var': 400000, 'base_cpu': 45.0, 'cpu_var': 25.0},
    {'name': 'obs', 'base_mem': 400000, 'mem_var': 200000, 'base_cpu': 20.0, 'cpu_var': 15.0},
]

def generate_process_data_for_30_days(output_file='system_usage_30days.csv'):
    processes = process_templates.copy()
    weights = [p['weight'] for p in processes]

    start_date = datetime(2025, 4, 1)
    days = 30

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Time', 'ProcessName', 'MemoryUsage', 'CPU'])

        for day_offset in range(days):
            current_day = start_date + timedelta(days=day_offset)
            current_time = datetime(current_day.year, current_day.month, current_day.day, 0, 0, 0)
            end_time = current_time.replace(hour=23, minute=59)

            while current_time <= end_time:
                if random.random() < 0.03:
                    new_proc = random.choice(occasional_processes).copy()
                    new_proc['weight'] = 8
                    processes.append(new_proc)
                    weights.append(new_proc['weight'])

                selected = random.choices(processes, weights=weights, k=1)[0]

                memory = max(10000, selected['base_mem'] + random.randint(-selected['mem_var'], selected['mem_var']))
                memory_str = f"{memory} kB"

                evening_boost = 1.3 if 18 <= current_time.hour < 23 else 1.0
                cpu_base = selected['base_cpu'] * evening_boost
                cpu = max(0.1, cpu_base + random.uniform(-selected['cpu_var'], selected['cpu_var']))
                cpu = round(cpu, 1)

                writer.writerow([
                    current_time.strftime('%Y-%m-%d'),
                    current_time.strftime('%H:%M:%S'),
                    selected['name'],
                    memory_str,
                    cpu
                ])

                current_time += timedelta(minutes=random.randint(1, 15))

if __name__ == '__main__':
    generate_process_data_for_30_days()
