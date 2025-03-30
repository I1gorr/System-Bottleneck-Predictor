import csv
import random
from datetime import datetime, timedelta

# Process templates with base values and behavior parameters
process_templates = [
    # System processes
    {'name': 'gnome-shell', 'base_mem': 170000, 'mem_var': 50000, 'base_cpu': 3.0, 'cpu_var': 2.0, 'weight': 15, 'fixed_pid': 2500},
    {'name': 'Xorg', 'base_mem': 80000, 'mem_var': 20000, 'base_cpu': 1.5, 'cpu_var': 1.0, 'weight': 5, 'fixed_pid': 1234},
    
    # Development tools
    {'name': 'android-studio', 'base_mem': 1500000, 'mem_var': 500000, 'base_cpu': 25.0, 'cpu_var': 15.0, 'weight': 20},
    {'name': 'code', 'base_mem': 400000, 'mem_var': 150000, 'base_cpu': 8.0, 'cpu_var': 5.0, 'weight': 18},
    {'name': 'emulator', 'base_mem': 2000000, 'mem_var': 800000, 'base_cpu': 40.0, 'cpu_var': 25.0, 'weight': 12},
    {'name': 'gradle', 'base_mem': 500000, 'mem_var': 200000, 'base_cpu': 15.0, 'cpu_var': 10.0, 'weight': 10},
    
    # Browsers
    {'name': 'firefox', 'base_mem': 800000, 'mem_var': 400000, 'base_cpu': 10.0, 'cpu_var': 8.0, 'weight': 25},
    {'name': 'chrome', 'base_mem': 1000000, 'mem_var': 500000, 'base_cpu': 15.0, 'cpu_var': 10.0, 'weight': 25},
    
    # Gaming
    {'name': 'steam', 'base_mem': 500000, 'mem_var': 300000, 'base_cpu': 20.0, 'cpu_var': 15.0, 'weight': 18},
    {'name': 'csgo_linux', 'base_mem': 1800000, 'mem_var': 600000, 'base_cpu': 50.0, 'cpu_var': 30.0, 'weight': 12},
    
    # Communication & Media
    {'name': 'discord', 'base_mem': 300000, 'mem_var': 150000, 'base_cpu': 7.0, 'cpu_var': 5.0, 'weight': 15},
    {'name': 'spotify', 'base_mem': 200000, 'mem_var': 100000, 'base_cpu': 4.0, 'cpu_var': 3.0, 'weight': 10},
]

# Additional occasional processes
occasional_processes = [
    {'name': 'zoom', 'base_mem': 350000, 'mem_var': 150000, 'base_cpu': 12.0, 'cpu_var': 8.0},
    {'name': 'postman', 'base_mem': 250000, 'mem_var': 100000, 'base_cpu': 6.0, 'cpu_var': 4.0},
    {'name': 'minecraft', 'base_mem': 1200000, 'mem_var': 400000, 'base_cpu': 45.0, 'cpu_var': 25.0},
    {'name': 'obs', 'base_mem': 400000, 'mem_var': 200000, 'base_cpu': 20.0, 'cpu_var': 15.0},
]

def generate_process_data(num_entries=1000):
    processes = process_templates.copy()
    weights = [p['weight'] for p in processes]
    pid_map = {}
    current_date = datetime(2025, 3, 18)
    
    with open('system_usage.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'PID', 'ProcessName', 'Memory', 'CPU'])  # Removed Usage column
        
        for _ in range(num_entries):
            # Occasionally add new temporary processes
            if random.random() < 0.03:
                new_proc = random.choice(occasional_processes).copy()
                new_proc['weight'] = 8
                processes.append(new_proc)
                weights.append(new_proc['weight'])
            
            # Select process with weighted probability
            selected = random.choices(processes, weights=weights, k=1)[0]
            
            # Manage PID lifecycle
            proc_name = selected['name']
            if 'fixed_pid' in selected:
                pid = selected['fixed_pid']
                active = True  # System processes never close
            else:
                if proc_name not in pid_map or random.random() < 0.1:
                    # Start new instance
                    pid = random.randint(10000, 99999)
                    pid_map[proc_name] = {'pid': pid, 'active': True, 'start_time': current_date}
                else:
                    # Use existing instance
                    pid_info = pid_map[proc_name]
                    pid = pid_info['pid']
                    # Random chance to close process
                    if random.random() < 0.05:
                        del pid_map[proc_name]
            
            # Generate memory with fluctuation
            memory = max(10000, selected['base_mem'] + random.randint(-selected['mem_var'], selected['mem_var']))
            memory_str = f"{memory} kB"
            
            # Generate CPU with spikes
            cpu_base = selected['base_cpu'] * (1.3 if current_date.hour > 18 else 1)  # Evening boost
            cpu = max(0.1, cpu_base + random.uniform(-selected['cpu_var'], selected['cpu_var']))
            cpu = round(cpu, 1)
            
            # Write entry
            writer.writerow([
                current_date.strftime('%Y-%m-%d'),
                pid,
                proc_name,
                memory_str,
                cpu
            ])  # Removed Usage value
            
            # Progress time (0-60 minute intervals)
            current_date += timedelta(minutes=random.randint(0, 60))

if __name__ == '__main__':
    generate_process_data(2000)