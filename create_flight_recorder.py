import json

def process_notebook():
    with open('../final_run.ipynb', 'r') as f:
        nb = json.load(f)

    # We need to find the cell that has: all_results = {}
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'all_results = {}' in ''.join(cell['source']):
            source = cell['source']
            new_source = []
            for line in source:
                new_source.append(line)
                if 'all_metrics.append(metrics)' in line:
                    # Inject our saving logic here
                    new_source.extend([
                        "\n",
                        "        # --- FLIGHT RECORDER: SAVE TRAJECTORY ---\n",
                        "        df_run = pd.DataFrame({\n",
                        "            'time_s': np.arange(len(result['pump'])),\n",
                        "            'pump_power_W': result['pump'],\n",
                        "            'battery_power_W': power_profile[:len(result['pump'])]\n",
                        "        })\n",
                        "        for z in range(NUM_ZONES):\n",
                        "            df_run[f'zone_{z+1}_temp_C'] = result['temps'][:, z]\n",
                        "            df_run[f'zone_{z+1}_flow_norm'] = result['flows'][:, z]\n",
                        "        \n",
                        "        # Clean the name to ensure valid filename\n",
                        "        clean_name = name.replace(' ', '_').replace('/', '_')\n",
                        "        save_path = f'results/{clean_name}_run.csv'\n",
                        "        df_run.to_csv(save_path, index=False)\n",
                        "        print(f'   💾 Saved full trajectory to {save_path}')\n",
                        "        # ----------------------------------------\n"
                    ])
            cell['source'] = new_source

    with open('../final_flight_recorder.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)

process_notebook()
print("Created final_flight_recorder.ipynb")
